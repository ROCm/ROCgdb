/* PDB symbol reader.

   Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* Read module/global symbols and provide computed variable locations.

   References:
     - microsoft-pdb cvinfo.h: https://github.com/microsoft/microsoft-pdb  */

#include "symtab.h"
#include "gdbtypes.h"
#include "objfiles.h"
#include "buildsym.h"
#include "complaints.h"
#include "source.h"
#include "block.h"
#include "cp-support.h"
#include "pdb/pdb-internal.h"
#include "frame.h"
#include "value.h"
#include "gdbarch.h"
#include "gdbcore.h"
#include "extract-store-integer.h"
#include "namespace.h"
#include "typeprint.h"
#include "event-top.h"
#include "symfile.h"
#include "gdbsupport/scoped_restore.h"

#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

namespace pdb
{

/* Parser for a CodeView symbol stream.  */

class pdb_sym_parser
{
public:
  pdb_sym_parser (pdb_per_objfile *pdb_, buildsym_compunit *cu_,
		  pdb_sym_flags flags_, pdb_range_pair_vec *func_ranges_,
		  enum language lang_, pdb_module_info *mod_info_,
		  pdb_symbol_lines *sym_lines_ = nullptr)
    : pdb (pdb_),
      cu (cu_),
      flags (flags_),
      func_ranges (func_ranges_),
      m_lang (lang_),
      m_mod_info (mod_info_),
      m_sym_lines (sym_lines_),
      /* Reentrant parser.  */
      m_scoped_parser (make_scoped_restore (&pdb_->cur_parser, this))
  {
    /* Use the architecture default until S_FRAMEPROC supplies a base.  */
    gdbarch *gdbarch = pdb->objfile->arch ();
    cur_frame_gdb_regnum = gdbarch_codeview_default_frame_regnum_p (gdbarch)
			     ? gdbarch_codeview_default_frame_regnum (gdbarch)
			     : -1;
    cur_param_frame_gdb_regnum = cur_frame_gdb_regnum;
  }

  DISABLE_COPY_AND_ASSIGN (pdb_sym_parser);

  /* Local-frame register, also used to resolve CV_REG_VFRAME.  */
  int cur_frame_gdb_regnum = -1;

  /* Parameter-frame register from S_FRAMEPROC.  */
  int cur_param_frame_gdb_regnum = -1;

  /* Select the frame register for SYM's parameter or local location.  */

  int frame_regnum_for (const symbol *sym) const
  {
    return (sym != nullptr && sym->is_argument ())
	     ? cur_param_frame_gdb_regnum
	     : cur_frame_gdb_regnum;
  }

  enum language lang () const
  {
    return m_lang;
  }

  /* Queue SYM to receive the line recorded at PC, along with the file
     that line counts within.  Only the C13 line table carries this;
     CodeView records no declaration line for data or types.  */
  void queue_line_for (symbol *sym, CORE_ADDR pc, bool with_line = false)
  {
    if (m_sym_lines == nullptr)
      return;
    auto it = m_sym_lines->at_pc.find (pc);
    if (it == m_sym_lines->at_pc.end ())
      return;
    m_sym_lines->to_apply.push_back ({ sym, it->second.subfile,
                                       it->second.line, with_line });
  }

  /* CodeView record handlers.  */
  void handle_func_sym (gdb_byte *rec_data, size_t body_sz, uint16_t rectype);
  void handle_pub_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_var_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_local_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_regrel_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_register_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_const_sym (gdb_byte *rec_data, uint16_t rectype,
			 uint16_t reclen);
  void handle_label_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_udt_sym (gdb_byte *rec_data, uint16_t rectype,
		       bool from_globals);
  void handle_block_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_thunk_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_inlinesite_sym (gdb_byte *rec_data, size_t body_size,
			      uint16_t rectype);
  void handle_scope_end_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_defrange_regrel (gdb_byte *rec_data, uint16_t reclen);
  void handle_defrange_reg (gdb_byte *rec_data, uint16_t reclen);
  void handle_defrange_fprel (gdb_byte *rec_data, uint16_t reclen);
  void handle_defrange_fprel_fullscope (gdb_byte *rec_data);
  void handle_procref_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_dataref_sym (gdb_byte *rec_data, uint16_t rectype);
  void handle_frameproc_sym (gdb_byte *rec_data);
  void handle_unamespace_sym (const gdb_byte *rec_data);

  /* Register NS_NAME as a namespace, including qualified names.  */

  void register_namespace (const char *ns_name);

  /* For each "::"-separated prefix of QNAME (e.g. "ns_a" and
     "ns_a::ns_b" for "ns_a::ns_b::foo"), register a namespace via
     register_namespace, unless the prefix is a class or a function.

     Background: PDB has no structural namespace records: namespaces
     live only as string prefixes on qualified names.

     TYPE_INDEX is the record's own type index.  When it names a tagged
     type, the tag's CV_prop_t answers the question outright: CV_PROP_SCOPED
     means a function body encloses it, so none of QNAME names a namespace,
     and neither that nor CV_PROP_ISNESTED means nothing encloses it but
     namespaces, so every prefix is one.  Only a tag whose enclosing scope
     is a class needs the prefixes classified one at a time, as does any
     record whose type index is not a tag at all — a variable's type index
     is its own type and says nothing about who declares it.  */
  void ensure_namespaces_for (const char *qname, uint32_t type_index);

  /* End the current S_LOCAL/S_DEFRANGE sequence.  */

  void clear_last_local ()
  {
    m_last_local = nullptr;
  }

  /* True when a local symbol named NAME was already created in the current
     scope.  MSVC emits both S_LOCAL+S_DEFRANGE* and the legacy
     S_REGREL32 / S_REGISTER for the same variable; the S_LOCAL+DEFRANGE
     location is strictly more precise, so the legacy record is skipped when a
     symbol of the same name already exists.  */
  bool local_already_defined (const char *name)
  {
    if (cu == nullptr || name == nullptr)
      return false;
    for (const symbol *s : cu->get_local_symbols ())
      if (s->linkage_name () != nullptr
	  && strcmp (s->linkage_name (), name) == 0)
	return true;
    return false;
  }

  /* Expose open scopes for end-of-stream recovery.  */

  pdb_scope_stack &scope_stack ()
  {
    return m_scope_stack;
  }

private:
  pdb_per_objfile *pdb;
  buildsym_compunit *cu;
  pdb_sym_flags flags;
  pdb_range_pair_vec *func_ranges;
  enum language m_lang;
  pdb_module_info *m_mod_info;

  /* Function declaration lines from the C13 line table, or nullptr.  */
  pdb_symbol_lines *m_sym_lines = nullptr;

  scoped_restore_tmpl<pdb_sym_parser *> m_scoped_parser;

  pdb_scope_stack m_scope_stack;
  symbol *m_last_local = nullptr;
  int m_remaining_params = 0;

  /* Section/offset of the enclosing procedure — the base that an
     S_INLINESITE binary-annotation code offset is measured from.  */
  uint16_t m_cur_proc_sect = 0;
  uint32_t m_cur_proc_offs = 0;

  /* Namespace names already registered by this parser.  */
  std::unordered_set<std::string> m_namespaces_seen;

  /* Prefixes already identified as tagged types.  */
  std::unordered_set<std::string> m_tag_prefixes_seen;
};

/* There are about 200 different symbol types in microsoft-pdb cvinfo.h.
   Below are the supported record types.  */

/* S_GPROC32 / S_LPROC32 - Global/local functions [cvinfo.h: PROCSYM32]
       uint32_t pParent
       uint32_t pEnd
       uint32_t pNext
       uint32_t len          procedure code size
       uint32_t DbgStart
       uint32_t DbgEnd
       uint32_t typind       type index
       uint32_t off          offset in section
       uint16_t seg          section number
       uint8_t  flags
       char     name[]  */
inline constexpr auto PDB_SYMBOL_FUNC_TYPE_OFFS = 24;
inline constexpr auto PDB_SYMBOL_FUNC_CODE_SIZE_OFFS = 12;
/* Offset of the first instruction after the prologue, relative to the
   procedure start.  */
inline constexpr auto PDB_SYMBOL_FUNC_DBGSTART_OFFS = 16;
inline constexpr auto PDB_SYMBOL_FUNC_SECTION_OFFS_OFFS = 28;
inline constexpr auto PDB_SYMBOL_FUNC_SECTION_NUM_OFFS = 32;
inline constexpr auto PDB_SYMBOL_FUNC_FLAGS_OFFS = 34;
/* Fixed fields plus room for an empty name.  */
inline constexpr auto PDB_SYMBOL_FUNC_MIN_SIZE = PDB_SYMBOL_FUNC_NAME_OFFS + 1;

/* S_GDATA32 / S_LDATA32 - Global/local data/variable [cvinfo.h: DATASYM32]
       uint32_t typind      TPI type index
       uint32_t offset      offset within section
       uint16_t segment     section number
       char     name[]  */
inline constexpr auto PDB_SYMBOL_VAR_TYPE_OFFS = 0;
inline constexpr auto PDB_SYMBOL_VAR_SECTION_OFFS_OFFS = 4;
inline constexpr auto PDB_SYMBOL_VAR_SECTION_NUM_OFFS = 8;

/* S_PUB32 - Public symbol (name + address, no type info) [cvinfo.h: PUBSYM32]
       uint32_t flags       public flags
       uint32_t offset      offset within section
       uint16_t segment     section number
       char     name[]  */
inline constexpr auto PDB_SYMBOL_PUB_FLAGS_OFFS = 0;
inline constexpr auto PDB_SYMBOL_PUB_SECT_OFFS_OFFS = 4;
inline constexpr auto PDB_SYMBOL_PUB_SECT_NUM_OFFS = 8;
inline constexpr auto PDB_SYMBOL_PUB_NAME_OFFS = 10;

/* S_PUB32 flag bits (CV_PUBSYMFLAGS in cvinfo.h).  */
inline constexpr uint32_t CV_PUBSYMFLAGS_FUNCTION = 0x02;

/* S_LOCAL / S_LOCAL32 - Local variable [cvinfo.h: LOCALSYM]
      uint32_t typind       TPI type index
      uint16_t flags        CV_LVARFLAGS bitfield (param, optimized out, etc.)
      char     name[]  */
inline constexpr auto PDB_SYMBOL_LOCAL_TYPE_OFFS = 0;
inline constexpr auto PDB_SYMBOL_LOCAL_FLAGS_OFFS = 4;
inline constexpr auto PDB_SYMBOL_LOCAL_NAME_OFFS = 6;

/* CV_LVARFLAGS bitfield (S_LOCAL flags field)  [cvinfo.h: CV_LVARFLAGS]  */
/* Variable is a parameter.  */
inline constexpr auto CV_LVARFLAG_IsParam = 0x0001;
/* Variable optimized out.  */
inline constexpr auto CV_LVARFLAG_OptOut = 0x0100;

/* S_UDT - Type-name binding [cvinfo.h: UDTSYM]
       uint32_t typind      TPI type index
       char     name[]      name  */
inline constexpr auto PDB_SYMBOL_UDT_TYPE_OFFS = 0;

/*  S_BLOCK32 - Block scope marker [cvinfo.h: BLOCKSYM32]
       uint32_t pParent    offset to parent scope record
       uint32_t pEnd       offset to S_END record for this block
       uint32_t len        length of code in bytes
       uint32_t offs       offset within section
       uint16_t seg        section number
       char     name[]  */

/* [cvinfo.h: BLOCKSYM32]
   pParent    — unsigned long pParent (exact)
   pEnd       — unsigned long pEnd (exact)
   len        — unsigned long len (exact)
   offs       — CV_uoff32_t off (renamed off→offs)
   seg        — unsigned short seg (exact)
   name       — unsigned char name[1]  */
inline constexpr auto PDB_SYMBOL_BLOCK_CODE_SIZE_OFFS = 8;
inline constexpr auto PDB_SYMBOL_BLOCK_SECTION_OFFS_OFFS = 12;
inline constexpr auto PDB_SYMBOL_BLOCK_SECTION_NUM_OFFS = 16;
inline constexpr auto PDB_SYMBOL_BLOCK_NAME_OFFS = 18;

/*  S_INLINESITE - Inlined function site, opens scope [cvinfo.h: INLINESITESYM]
     Marks where a function was inlined, variable size.
       uint32_t pParent              offset to parent scope
       uint32_t pEnd                 offset to S_INLINESITE_END
       uint32_t inlinee              IPI item id of the inlined function
       uint8_t  binaryAnnotations[]  binary annotation data  */
inline constexpr auto PDB_SYMBOL_INLINESITE_INLINEE_IDX_OFFS = 8;
inline constexpr auto PDB_SYMBOL_INLINESITE_BINANNOT_OFFS = 12;

/*  S_INLINESITE2 - Inlined function site with an invocation count
    [cvinfo.h: INLINESITESYM2].  Identical to S_INLINESITE but with an extra
    uint32_t between the inlinee id and the annotations.
       uint32_t pParent              offset to parent scope
       uint32_t pEnd                 offset to S_INLINESITE_END
       uint32_t inlinee              IPI item id of the inlined function
       uint32_t invocations          invocation count (ignored)
       uint8_t  binaryAnnotations[]  binary annotation data  */
inline constexpr auto PDB_SYMBOL_INLINESITE2_BINANNOT_OFFS = 16;

/*  S_REGISTER - Register variable [cvinfo.h: REGSYM]
       uint32_t typind     TPI type index
       uint16_t reg        CV register
       char     name[]  */
inline constexpr auto PDB_SYMBOL_REG_TYPE_OFFS = 0;
inline constexpr auto PDB_SYMBOL_REG_REG_OFFS = 4;
inline constexpr auto PDB_SYMBOL_REG_NAME_OFFS = 6;

/*   S_REGREL32 record layout [cvinfo.h: REGREL32]
       uint32_t off        offset from register
       uint32_t typind     TPI type index
       uint16_t reg        CV register ID
       char     name[]  */
inline constexpr auto PDB_SYMBOL_REGREL_OFFS_OFFS = 0;
inline constexpr auto PDB_SYMBOL_REGREL_TYPE_OFFS = 4;
inline constexpr auto PDB_SYMBOL_REGREL_REG_OFFS = 8;
inline constexpr auto PDB_SYMBOL_REGREL_NAME_OFFS = 10;

/*   S_CONSTANT - Constant value.  [cvinfo.h: CONSTSYM]
       uint32_t typind      TPI type index
       uint16_t value       numeric leaf containing value
       char     name[]  */
inline constexpr auto PDB_SYMBOL_CONST_TYPE_OFFS = 0;

/*  S_LABEL32 - Code label [cvinfo.h: LABELSYM32]
       uint32_t offs        offset within section
       uint16_t seg         section number
       uint8_t  flags       CV_PROCFLAGS
       char     name[]  */
inline constexpr auto PDB_SYMBOL_LABEL_OFFS_OFFS = 0;
inline constexpr auto PDB_SYMBOL_LABEL_SEG_OFFS = 4;
inline constexpr auto PDB_SYMBOL_LABEL_NAME_OFFS = 7;

/*  S_FRAMEPROC - Frame procedure info [cvinfo.h: FRAMEPROCSYM]
       uint32_t cbFrame        count of bytes of total frame of procedure
       uint32_t cbPad          count of bytes of padding in the frame
       uint32_t offPad         offset (FP relative) to padding start
       uint32_t cbSaveRegs     count of bytes of callee save registers
       uint32_t offExHdlr      offset of exception handler
       uint16_t sectExHdlr     section id of exception handler
       uint32_t flags          bit fields:
	       bits 14-15 = encodedLocalBasePointer
			      (0=none, 1=RSP, 2=RBP, 3=R13)
	       bits 16-17 = encodedParamBasePointer  */
inline constexpr auto PDB_SYMBOL_FRAMEPROC_FRAME_SIZE_OFFS = 0;
inline constexpr auto PDB_SYMBOL_FRAMEPROC_FLAGS_OFFS = 22;
inline constexpr auto PDB_FRAMEPROC_LOCAL_BP_SHIFT = 14;
inline constexpr auto PDB_FRAMEPROC_PARAM_BP_SHIFT = 16;
inline constexpr auto PDB_FRAMEPROC_BP_MASK = 0x3;

/*  S_THUNK32 - Thunk record for indirect calls [cvinfo.h: THUNKSYM32]
       uint32_t pParent    pointer to the parent
       uint32_t pEnd       pointer to this blocks end
       uint32_t pNext      pointer to next symbol
       uint32_t off        offset within section
       uint16_t seg        section number
       uint16_t len        length of thunk
       uint8_t  ord        type of thunk
       char     name[]    */
inline constexpr auto PDB_SYMBOL_THUNK_SECTION_OFFS_OFFS = 12;
inline constexpr auto PDB_SYMBOL_THUNK_SECTION_NUM_OFFS = 16;
inline constexpr auto PDB_SYMBOL_THUNK_CODE_SIZE_OFFS = 18;
inline constexpr auto PDB_SYMBOL_THUNK_NAME_OFFS = 21;

/*  S_PROCREF / S_LPROCREF / S_DATAREF - Symbol reference [cvinfo.h: REFSYM2]
       uint32_t sumName     checksum of symbol name
       uint32_t ibSym       symbol offset in module stream
       uint16_t imod        module index
       char     name[]  */
inline constexpr auto PDB_SYMBOL_REF_SYM_OFFSET_OFFS = 4;

/*  S_DEFRANGE_REGISTER - S_LOCAL reg location [cvinfo.h: DEFRANGESYMREGISTER]
       uint16_t reg        CV register ID
       uint16_t attr       attributes
       Followed by variable-length range and gap record.  */
inline constexpr auto PDB_SYMBOL_DEFRANGE_REG_REG_OFFS = 0;
inline constexpr auto PDB_SYMBOL_DEFRANGE_REG_ATTR_OFFS = 2;
inline constexpr auto PDB_SYMBOL_DEFRANGE_REG_RANGE_OFFS = 4;
inline constexpr auto PDB_SYMBOL_DEFRANGE_REG_GAPS_OFFS = 12;

/*  S_DEFRANGE_REGISTER_REL - Register-relative location for S_LOCAL
       uint16_t reg        CV register ID
       uint16_t flags      bit 0 = spilledUdtMember, bits 4-15 = offsetParent
       int32_t  off        offset from register
       Followed by variable-length range and gap records.  */
inline constexpr auto PDB_SYMBOL_DEFRANGE_REGREL_REG_OFFS = 0;
inline constexpr auto PDB_SYMBOL_DEFRANGE_REGREL_FLAGS_OFFS = 2;
inline constexpr auto PDB_SYMBOL_DEFRANGE_REGREL_OFFSET_OFFS = 4;
inline constexpr auto PDB_SYMBOL_DEFRANGE_REGREL_RANGE_OFFS = 8;
inline constexpr auto PDB_SYMBOL_DEFRANGE_REGREL_GAPS_OFFS = 16;
inline constexpr auto PDB_DEFRANGE_REGREL_SPILLED_MEMBER = 0x1;

/*  S_DEFRANGE_FRAMEPOINTER_REL - FP-relative location for S_LOCAL
       int32_t  off          offset from frame pointer
       uint32_t offStart     section-relative start PC
       uint16_t isectStart   section index
       uint16_t cbRange      byte length of range
       uint16_t gaps[]       variable-length gaps array  */
inline constexpr auto PDB_SYMBOL_DEFRANGE_FPREL_OFFSET_OFFS = 0;
inline constexpr auto PDB_SYMBOL_DEFRANGE_FPREL_OFFSTART_OFFS = 4;
inline constexpr auto PDB_SYMBOL_DEFRANGE_FPREL_GAPS_OFFS = 12;

/* S_DEFRANGE* address range [cvinfo.h: CV_lvar_addr_range]
       uint32_t offStart       section-relative start offset
       uint16_t isectStart     section index
       uint16_t cbRange        byte length of range  */
inline constexpr auto CV_RANGE_OFF_START_OFFS = 0;
inline constexpr auto CV_RANGE_ISECT_OFFS = 4;
inline constexpr auto CV_RANGE_CBRANGE_OFFS = 6;

/*  S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE - FP-relative, entire function
       int32_t  offs          offset from frame pointer  */
inline constexpr auto PDB_SYMBOL_DEFRANGE_FPREL_FULLSCOPE_OFFSET_OFFS = 0;

/* S_END / S_PROC_ID_END / S_INLINESITE_END have no body (reclen = 2).  */

/* Below are the name field offsets for unsupported symbols for which
   we want just to show the name.  */

/* S_OBJNAME: sig(4) name(...)    [cvinfo.h: OBJNAMESYM]  */
inline constexpr auto PDB_SYMBOL_OBJNAME_NAME_OFFS = 4;

/* S_BPREL32: off(4) typind(4) name(...)   [cvinfo.h: BPRELSYM32]  */
inline constexpr auto PDB_SYMBOL_BPREL_NAME_OFFS = 8;

/* S_LTHREAD32 / S_GTHREAD32: same as S_GDATA32 [cvinfo.h: THREADSYM32]
   (use PDB_SYMBOL_VAR_* offsets).  */

/* S_UNAMESPACE: name(...) [cvinfo.h: UNAMESPACE]  */
inline constexpr auto PDB_SYMBOL_UNAMESPACE_NAME_OFFS = 0;

/* S_EXPORT: ordinal(2) flags(2) name(...)  [cvinfo.h: EXPORTSYM]  */
inline constexpr auto PDB_SYMBOL_EXPORT_NAME_OFFS = 4;

/* S_SECTION: sectnum(2) align(1) pad(1) rva(4) len(4) chars(4) name()  */
inline constexpr auto PDB_SYMBOL_SECTION_NAME_OFFS = 16;

/* S_COFFGROUP: len(4) chars(4) off(4) seg(2) name(...)  */
inline constexpr auto PDB_SYMBOL_COFFGROUP_NAME_OFFS = 14;

/* S_FILESTATIC: typind(4) modfileoffs(4) flags(2) name(...)  */
inline constexpr auto PDB_SYMBOL_FILESTATIC_NAME_OFFS = 10;

/* Virtual frame pointer (resolved via S_FRAMEPROC or FPO data).
   Shared across all CodeView architectures.  */
inline constexpr auto CV_REG_VFRAME = 30006;

/* Convert a CodeView register ID to a GDB register number.

   CodeView has its own per-architecture register numbering.
   The mapping is delegated to the architecture via
   the gdbarch_codeview_reg_to_regnum hook.

   CV_REG_VFRAME is a generic frame pointer placeholder that resolves
   at runtime to the function's actual frame register, communicated
   via S_FRAMEPROC.  The currently tracked frame regnum lives on the
   active pdb_sym_parser (reachable via pdb->cur_parser), and is kept
   up to date by the S_FRAMEPROC handler.

   Returns -1 if the architecture has no CodeView mapping, or if the
   register is not recognized.  Errors out via pdb_error if the
   architecture does not implement the hook.  */

static int
cv_reg_to_gdb_regnum (pdb_per_objfile *pdb, uint16_t cv_reg)
{
  if (cv_reg == CV_REG_VFRAME)
    {
      gdb_assert (pdb->cur_parser != nullptr);
      return pdb->cur_parser->cur_frame_gdb_regnum;
    }

  gdbarch *gdbarch = pdb->objfile->arch ();
  if (!gdbarch_codeview_reg_to_regnum_p (gdbarch))
    pdb_error (
      "PDB: unsupported architecture %s for CodeView register mapping",
      gdbarch_bfd_arch_info (gdbarch)->printable_name);

  return gdbarch_codeview_reg_to_regnum (gdbarch, cv_reg);
}

/* Return a value of the given type marked entirely unavailable.  */

static value *
value_unavailable (type *type)
{
  value *val = value::allocate (type);
  val->mark_bytes_unavailable (0, type->length ());
  return val;
}

/* See pdb-internal.h.  */

const pdb_loc_entry *
pdb_loclist_select (const pdb_loclist_baton *baton, CORE_ADDR pc,
		    bool *gapped)
{
  const pdb_loc_entry *found = nullptr;
  const pdb_loc_entry *full_scope = nullptr;

  *gapped = false;

  /* We can have both a FULL_SCOPE entry and narrower ranges.  In that case a
     narrower range applies.  This is also valid for gaps - a gap in a narrower
     range means the variable is unavailable at this PC regardless of
     FULL_SCOPE.  A gap only speaks for the range holding it, so another range
     covering the same PC still answers.  */

  for (auto *e = baton->entries; e != nullptr; e = e->next)
    {
      /* DEFRANGE_FULL_SCOPE doesn't set the (full) range but this flag */
      if (e->is_full_scope)
	{
	  if (full_scope == nullptr)
	    full_scope = e;

	  continue;
	}

      /* The entry holds linked addresses; bring it to where the section
	 was loaded before comparing.  */
      CORE_ADDR reloc = baton->pdb->section_reloc (e->section);

      if (pc < e->start + reloc || pc >= e->end + reloc)
	continue;

      /* Check gaps.  */
      bool in_gap = false;
      for (int i = 0; i < e->num_gaps; i++)
	{
	  if (pc >= e->gaps[i].start + reloc && pc < e->gaps[i].end + reloc)
	    {
	      in_gap = true;
	      break;
	    }
	}

      if (in_gap)
	{
	  *gapped = true;
	  continue;
	}

      found = e;
    }

  if (found != nullptr)
    return found;

  return *gapped ? nullptr : full_scope;
}

/* Mark V as frame-dependent.  Callers that create a watchpoint use this to
   learn they must capture the frame the value was read in.  */

static value *
pdb_frame_scoped (value *v)
{
  v->set_scope (v->scope () | LOCATION_SCOPE_FRAME);
  return v;
}

/* symbol_computed_ops: read a variable described by a pdb_loclist_baton.
   Walks the parsed location entries to find the one matching current PC.  */

static value *
pdb_loclist_read_variable (symbol *symbol, const frame_info_ptr &frame)
{
  gdb_assert (symbol != nullptr);

  auto *baton = (pdb_loclist_baton *) SYMBOL_LOCATION_BATON (symbol);

  if (baton == nullptr || baton->entries == nullptr)
    return pdb_frame_scoped (value::allocate_optimized_out (symbol->type ()));

  /* GDB asks with no frame to learn whether one is needed.  Answering from
     the selected frame says none is, and the read that follows then loses
     the frame the caller meant.  */
  if (frame == nullptr)
    throw_error (DWARF2_FRAME_CONTEXT_MISSING,
		 _("No frame selected for PDB location of \"%s\"."),
		 symbol->natural_name ());

  frame_info_ptr frame_info = frame;

  CORE_ADDR pc = get_frame_address_in_block (frame_info);
  bool gapped;
  const pdb_loc_entry *e = pdb_loclist_select (baton, pc, &gapped);

  if (e == nullptr)
    return pdb_frame_scoped (value::allocate_optimized_out (symbol->type ()));

  int gdb_regnum = e->gdb_regnum;
  int32_t offset = e->offset;
  bool is_register = e->is_register;

  if (gdb_regnum < 0)
    return pdb_frame_scoped (value_unavailable (symbol->type ()));

  if (is_register && !baton->deref)
    return pdb_frame_scoped (value_from_register (symbol->type (), gdb_regnum,
						  frame_info, 0, 0));

  CORE_ADDR regval;
  try
    {
      regval = get_frame_register_unsigned (frame_info, gdb_regnum);
    }
  catch (const gdb_exception_error &ex)
    {
      if (ex.error == NOT_AVAILABLE_ERROR)
	return pdb_frame_scoped (value_unavailable (symbol->type ()));
      throw;
    }

  CORE_ADDR addr = is_register ? regval : regval + offset;

  if (baton->deref)
    {
      gdbarch *arch = symbol->arch ();
      addr = read_memory_unsigned_integer (addr,
					   gdbarch_ptr_bit (arch) / 8,
					   gdbarch_byte_order (arch));
    }

  return pdb_frame_scoped (value_at_lazy (symbol->type (), addr));
}

static void
pdb_loclist_describe_location (symbol *symbol, CORE_ADDR, ui_file *stream)
{
  const auto *baton
    = (const pdb_loclist_baton *) SYMBOL_LOCATION_BATON (symbol);

  if (baton == nullptr || baton->entries == nullptr)
    gdb_printf (stream, "optimized out");
  else if (baton->entries->next != nullptr)
    gdb_printf (stream, "location list");
  else if (baton->entries->is_register)
    gdb_printf (stream, "a register");
  else
    gdb_printf (stream, "register + %d", (int) baton->entries->offset);
}

static void
pdb_loclist_tracepoint_var_ref (symbol * /*symbol*/, agent_expr * /*ax*/,
				axs_value * /*value*/)
{
  /* Not supported for PDB yet.  */
}

static void
pdb_loclist_generate_c_location (symbol * /*symbol*/, string_file * /*stream*/,
				 gdbarch * /*gdbarch*/,
				 std::vector<bool> & /*registers_used*/,
				 CORE_ADDR /*pc*/,
				 const char * /*result_name*/)
{
  /* Not supported for PDB yet.  */
}

static const symbol_computed_ops pdb_loclist_funcs =
{
  pdb_loclist_read_variable,
  /* read_variable_at_entry: CodeView carries no entry values.  */
  nullptr,
  pdb_loclist_describe_location,
  /* location_has_loclist */
  0,
  pdb_loclist_tracepoint_var_ref,
  pdb_loclist_generate_c_location
};

/* LOC_COMPUTED implementation index, registered during file init.  */
static int pdb_loclist_index;

static void
pdb_init_loclist (void)
{
  pdb_loclist_index = register_symbol_computed_impl (LOC_COMPUTED,
						     &pdb_loclist_funcs);
}

/* See pdb-internal.h.  */

bool
pdb_is_pdb_location (const symbol *sym)
{
  return (sym->loc_class () == LOC_COMPUTED
	  && sym->computed_ops () == &pdb_loclist_funcs);
}

/* Parse a symbol record; REC must contain a complete fixed header.  */

std::optional<pdb_sym_record_hdr>
pdb_parse_sym_record_hdr (const gdb_byte *rec, const gdb_byte *end)
{
  pdb_sym_record_hdr hdr;
  hdr.len = read_u16 (rec + PDB_RECORD_LEN_OFFS);
  hdr.type = read_u16 (rec + PDB_RECORD_TYPE_OFFS);

  if (hdr.len < 2 || rec + hdr.rec_size () > end)
    return std::nullopt;

  return hdr;
}

/* Create a full-scope location baton (S_REGREL32, S_REGISTER).  */

static void
pdb_set_symbol_location (symbol *sym, pdb_per_objfile *pdb, int gdb_regnum,
			 int32_t offset, bool is_register)
{
  objfile *objfile = pdb->objfile;
  pdb_loclist_baton *baton = OBSTACK_ZALLOC (&objfile->objfile_obstack,
					     pdb_loclist_baton);
  baton->pdb = pdb;

  auto *entry = OBSTACK_ZALLOC (&objfile->objfile_obstack, pdb_loc_entry);
  entry->gdb_regnum = gdb_regnum;
  entry->offset = offset;
  entry->is_register = is_register;
  entry->is_full_scope = true;

  baton->entries = entry;
  SYMBOL_LOCATION_BATON (sym) = baton;
  sym->set_loc_class_index (pdb_loclist_index);
}

/* Prepend a location to SYM's existing baton; START/END are linked PCs.  */

static void
pdb_add_loc_entry (symbol *sym, pdb_per_objfile *pdb, CORE_ADDR start,
		   CORE_ADDR end, uint16_t section, int gdb_regnum,
		   int32_t offset, bool is_register, bool is_full_scope,
		   const std::vector<pdb_defrange_gap> &gaps)
{
  auto *baton = (pdb_loclist_baton *) SYMBOL_LOCATION_BATON (sym);
  auto num_gaps = static_cast<int> (gaps.size ());

  /* Store gaps immediately after the entry on the objfile obstack.  */
  size_t alloc_size = sizeof (pdb_loc_entry) + num_gaps * sizeof (pdb_loc_gap);
  auto *entry = XOBNEWVAR (&pdb->objfile->objfile_obstack, pdb_loc_entry,
			   alloc_size);

  entry->start = start;
  entry->end = end;
  entry->section = section;
  entry->gdb_regnum = gdb_regnum;
  entry->offset = offset;
  entry->is_register = is_register;
  entry->is_full_scope = is_full_scope;
  entry->num_gaps = num_gaps;

  /* Resolve gap offsets to absolute CORE_ADDR.  */
  for (uint32_t i = 0; i < num_gaps; i++)
    {
      entry->gaps[i].start = start + gaps[i].offset;
      entry->gaps[i].end = entry->gaps[i].start + gaps[i].length;
    }

  /* Prepend to linked list.  */
  entry->next = baton->entries;
  baton->entries = entry;
}

/* Give SYM a full-scope register location and add it to the locals.
   Unknown registers produce unavailable values.  */

static void
pdb_set_register_location (symbol *sym, pdb_per_objfile *pdb, uint16_t cv_reg,
			   int32_t offset, bool is_register,
			   buildsym_compunit *cu)
{
  int gdb_regnum = cv_reg_to_gdb_regnum (pdb, cv_reg);
  if (gdb_regnum < 0)
    {
      pdb_warning ("unsupported CV register %u for '%s'", cv_reg,
		   sym->natural_name ());
      gdb_regnum = -1;
    }

  /* S_REGREL32 and S_REGISTER each introduce a new symbol with a self-contained
     location, so the symbol is always inserted here.  */
  pdb_set_symbol_location (sym, pdb, gdb_regnum, offset, is_register);
  add_symbol_to_list (sym, cu->get_local_symbols ());
}

/* Parsed symbol fields shared by the record-specific builders.  */
struct pdb_sym
{
  uint16_t rectype;
  const char *name = nullptr;
  uint32_t type_index = 0;
  pdb_per_objfile *pdb;
  gdb_byte *rec_data;
  buildsym_compunit *cu;
  symbol *sym = nullptr;

protected:
  pdb_sym (gdb_byte *rec_data_in, uint16_t rectype_in, pdb_per_objfile *pdb_in,
	   buildsym_compunit *cu_in = nullptr)
    : rectype (rectype_in),
      pdb (pdb_in),
      rec_data (rec_data_in),
      cu (cu_in)
  {
  }

  /* Set name from a byte offset into rec_data.  */
  void set_name (uint32_t offset)
  {
    name = pdb_canonical_name (pdb, CSTR (rec_data + offset));
  }

  /* Set type_index from a byte offset into rec_data.  */
  void set_type (uint32_t offset)
  {
    type_index = read_u32 (rec_data + offset);
  }

  /* Allocate a symbol and register its inferred namespaces.  */

  void allocate_symbol (domain_enum domain)
  {
    sym = pdb->objfile->new_symbol<symbol> ();
    gdb_assert (pdb->cur_parser != nullptr);
    sym->set_language (pdb->cur_parser->lang (),
		       &pdb->objfile->objfile_obstack);
    sym->compute_and_set_names (name, true, pdb->objfile->per_bfd);
    sym->set_domain (domain);

    gdb_assert (pdb->cur_parser != nullptr);
    pdb->cur_parser->ensure_namespaces_for (name, type_index);
  }

  /* pub_mangled_names is keyed unrelocated, so undo any load bias in ADDR.  */
  CORE_ADDR unrelocated_pub_key (CORE_ADDR addr) const
  {
    return addr - pdb->objfile->text_section_offset ();
  }

  /* Name the symbol using its mangled name as found in S_PUB32.
     If S_PUB32 for the symbol is not available use its NATURAL name
     as recorded in the CodeView record.  */

  void set_symbol_names (CORE_ADDR addr, const char *natural)
  {
    natural = pdb_canonical_name (pdb, natural);

    auto it = pdb->pub_mangled_names.find (unrelocated_pub_key (addr));
    if (it != pdb->pub_mangled_names.end ())
      {
	sym->compute_and_set_names (it->second, true, pdb->objfile->per_bfd);
	/* Set the demangled name from the CodeView record.  */
	const char *demangled
	  = obstack_strdup (&pdb->objfile->objfile_obstack, natural);
	sym->set_demangled_name (demangled, &pdb->objfile->objfile_obstack);
      }
    else
      sym->compute_and_set_names (natural, true, pdb->objfile->per_bfd);
  }

  /* Allocate an addressed symbol and infer namespaces from NATURAL.  */

  void allocate_symbol_named (domain_enum domain, CORE_ADDR addr,
			      const char *natural)
  {
    sym = pdb->objfile->new_symbol<symbol> ();
    gdb_assert (pdb->cur_parser != nullptr);
    sym->set_language (pdb->cur_parser->lang (),
		       &pdb->objfile->objfile_obstack);
    set_symbol_names (addr, natural);
    sym->set_domain (domain);

    gdb_assert (pdb->cur_parser != nullptr);
    pdb->cur_parser->ensure_namespaces_for (natural, type_index);
  }

  /* Set a register location for the symbol using pdb/cu from the class.  */
  void set_register_location (uint16_t reg, int32_t offset, bool is_register)
  {
    pdb_set_register_location (sym, pdb, reg, offset, is_register, cu);
  }

  /* Return a printable type name, or an empty string on failure.  */

  std::string type_name ()
  {
    if (type_index == 0)
      return {};
    try
      {
	return type_to_string (&pdb_tpi_resolve_type (pdb, type_index));
      }
    catch (const gdb_exception &)
      {
	return {};
      }
  }

  /* Set symbol type via TPI resolution.  */
  void sym_set_type ()
  {
    sym->set_type (&pdb_tpi_resolve_type (pdb, type_index));
  }

  /* Helper to set type on a given symbol via TPI resolution.  */
  void set_gdb_sym_type (symbol *sym_arg)
  {
    sym_arg->set_type (&pdb_tpi_resolve_type (pdb, type_index));
  }

public:
  /* Base dump() — prints type_index, type name, and symbol name.  */
  void dump ()
  {
    auto tn = type_name ();
    gdb_printf ("  ti=%04X", type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" `%s`\n", name);
  }
};

/* See pdb-internal.h.  */

const char *
pdb_per_objfile::get_section_name (uint16_t sect_num) const
{
  if (sect_num == 0)
    return "(none)";

  bfd *abfd = objfile->obfd.get ();
  uint16_t idx = 0;
  for (auto *sect = abfd->sections; sect != nullptr; sect = sect->next)
    {
      if (++idx == sect_num)
	return bfd_section_name (sect);
    }

  return "(unknown)";
}

/* S_GPROC32 / S_LPROC32 / S_GPROC32_ID / S_LPROC32_ID.  */
struct pdb_func_sym final : pdb_sym
{
  uint32_t code_sz;
  uint32_t sect_offs;
  uint16_t sect_num;

  pdb_func_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
		buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    set_type (PDB_SYMBOL_FUNC_TYPE_OFFS);
    set_name (PDB_SYMBOL_FUNC_NAME_OFFS);

    /* Resolve _ID signatures through IPI when a matching record exists.  */
    if (rectype == S_GPROC32_ID || rectype == S_LPROC32_ID)
      {
	uint32_t sig_ti;
	if (pdb_ipi_lookup_inlinee (pdb, type_index, nullptr, &sig_ti))
	  type_index = sig_ti;
      }

    code_sz = read_u32 (rec_data + PDB_SYMBOL_FUNC_CODE_SIZE_OFFS);
    sect_offs = read_u32 (rec_data + PDB_SYMBOL_FUNC_SECTION_OFFS_OFFS);
    sect_num = read_u16 (rec_data + PDB_SYMBOL_FUNC_SECTION_NUM_OFFS);
    m_flags = read_u8 (rec_data + PDB_SYMBOL_FUNC_FLAGS_OFFS);
  }

  symbol *create_gdb_sym ()
  {
    const auto *of = pdb->objfile;

    /* PDB stores a C++ function name without its parameter list
       (e.g. "ns_a::ns_a_func"), and GDB expects the natural name of a C++
       function to carry one so overloads stay distinct.  C has no
       overloads and its natural name is the bare identifier, so a list
       there would corrupt every frame line and symbol print.  */
    gdb_assert (pdb->cur_parser != nullptr);
    enum language lang = pdb->cur_parser->lang ();

    type *func_type = &pdb_tpi_resolve_type (pdb, type_index);

    const char *natural;
    if (lang == language_cplus
	&& (check_typedef (func_type)->code () == TYPE_CODE_FUNC
	    || check_typedef (func_type)->code () == TYPE_CODE_METHOD))
      {
	string_file buf;
	buf.puts (name);
	c_type_print_args (func_type, &buf, 1, lang,
			   &type_print_raw_options);
	buf.puts (pdb_method_ref_qualifier (pdb, func_type));
	natural = obstack_strdup (&pdb->objfile->objfile_obstack,
				  buf.string ().c_str ());
      }
    else
      natural = name;

    /* The mangled (linkage) name, if any, lives in the matching S_PUB32
       public symbol, found by address.  */
    CORE_ADDR addr = pdb->map_section_offset_to_pc (sect_num, sect_offs);
    allocate_symbol_named (FUNCTION_DOMAIN, addr, natural);

    pdb->cur_parser->queue_line_for (sym, addr);
    sym->set_loc_class_index (LOC_BLOCK);
    sym->set_type (func_type);
    sym->set_section_index (SECT_OFF_TEXT (of));
    bool is_global = (rectype == S_GPROC32 || rectype == S_GPROC32_ID);
    auto &list = is_global ? cu->get_global_symbols ()
			   : cu->get_file_symbols ();
    add_symbol_to_list (sym, list);
    return sym;
  }

  void dump ()
  {
    auto tn = type_name ();
    auto sect_name = pdb->get_section_name (sect_num);
    uint32_t end_addr = sect_offs + code_sz - 1;
    gdb_printf ("  %s:%08X-%08X [sz:%u] ti=%04X", sect_name, sect_offs,
		end_addr, code_sz, type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" `%s`\n", name);
  }

private:
  uint8_t m_flags;
};

/* S_GDATA32 / S_LDATA32.  */
struct pdb_var_sym final : pdb_sym
{
  pdb_var_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
	       buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    set_type (PDB_SYMBOL_VAR_TYPE_OFFS);
    m_sect_offs = read_u32 (rec_data + PDB_SYMBOL_VAR_SECTION_OFFS_OFFS);
    m_sect_num = read_u16 (rec_data + PDB_SYMBOL_VAR_SECTION_NUM_OFFS);
    set_name (PDB_SYMBOL_VAR_NAME_OFFS);
  }

  symbol *create_gdb_sym (bool in_scope = false)
  {
    if (!pdb->section_valid (m_sect_num))
      {
	pdb_complaint ("PDB: bad section %u for '%s'", m_sect_num, name);
	return nullptr;
      }

    CORE_ADDR addr = pdb->map_section_offset_to_pc (m_sect_num, m_sect_offs);

    allocate_symbol_named (VAR_DOMAIN, addr, name);
    sym->set_loc_class_index (LOC_STATIC);
    set_gdb_sym_type (sym);
    sym->set_value_address (addr);

    sym->set_section_index (m_sect_num - 1);
    bool is_global = (rectype == S_GDATA32);

    /* A static declared in a function body outlives the call but is named
       only from inside it, so it belongs to that block.  */
    std::vector<symbol *> *list;
    if (is_global)
      list = &cu->get_global_symbols ();
    else if (in_scope)
      list = &cu->get_local_symbols ();
    else
      list = &cu->get_file_symbols ();

    add_symbol_to_list (sym, *list);
    return sym;
  }

  void dump ()
  {
    auto tn = type_name ();
    auto sect_name = pdb->get_section_name (m_sect_num);
    gdb_printf ("  %s:%08X ti=%04X", sect_name, m_sect_offs, type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" `%s`\n", name);
  }

private:
  uint32_t m_sect_offs;
  uint16_t m_sect_num;
};

/* S_PUB32 — public symbol (name + address, no type info).  */
struct pdb_pub_sym final : pdb_sym
{
  pdb_pub_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
	       buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    m_pub_flags = read_u32 (rec_data + PDB_SYMBOL_PUB_FLAGS_OFFS);
    m_sect_offs = read_u32 (rec_data + PDB_SYMBOL_PUB_SECT_OFFS_OFFS);
    m_sect_num = read_u16 (rec_data + PDB_SYMBOL_PUB_SECT_NUM_OFFS);
    set_name (PDB_SYMBOL_PUB_NAME_OFFS);
  }

  void dump ()
  {
    auto sect_name = pdb->get_section_name (m_sect_num);
    gdb_printf ("  %s:%08X flags=%08X `%s`\n", sect_name, m_sect_offs,
		m_pub_flags, name);
  }

private:
  uint32_t m_pub_flags;
  uint32_t m_sect_offs;
  uint16_t m_sect_num;
};

/* S_LOCAL / S_LOCAL32.  */
struct pdb_local_sym final : pdb_sym
{
  uint16_t flags;

  pdb_local_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
		 buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    set_type (PDB_SYMBOL_LOCAL_TYPE_OFFS);
    set_name (PDB_SYMBOL_LOCAL_NAME_OFFS);
    flags = read_u16 (rec_data + PDB_SYMBOL_LOCAL_FLAGS_OFFS);
  }

  symbol *create_gdb_sym ()
  {
    allocate_symbol (VAR_DOMAIN);
    set_gdb_sym_type (sym);

    if (flags & CV_LVARFLAG_IsParam)
      {
	sym->set_is_argument (true);

	/* Win64 passes anything wider than 8 bytes indirectly, and clang-cl
	   records that by giving the parameter a reference type.  C has no
	   references, so the indirection is the ABI's, not the program's:
	   show the value the way the source declared it.  */
	if (m_deref_param && sym->type () != nullptr
	    && sym->type ()->code () == TYPE_CODE_REF
	    && sym->type ()->target_type () != nullptr)
	  sym->set_type (sym->type ()->target_type ());
	else
	  m_deref_param = false;
      }
    else
      m_deref_param = false;

    /* S_LOCAL carries type + name only; actual location comes from
       subsequent S_DEFRANGE* records.  Default to optimized-out until
       DEFRANGE records attach a real location via pdb_add_loc_entry.  */
    sym->set_loc_class_index (LOC_OPTIMIZED_OUT);
    return sym;
  }

  /* True when the location records give the address of the value.  */
  bool deref_param () const
  { return m_deref_param; }

  void set_deref_param (bool v)
  { m_deref_param = v; }

  void dump ()
  {
    auto tn = type_name ();
    gdb_printf ("  ti=%04X", type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" flags=%04X `%s`\n", flags, name);
  }

private:
  bool m_deref_param = false;
};

/* S_REGREL32 — register-relative local variable.  */
struct pdb_regrel_sym final : pdb_sym
{
  pdb_regrel_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
		  buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    set_type (PDB_SYMBOL_REGREL_TYPE_OFFS);
    set_name (PDB_SYMBOL_REGREL_NAME_OFFS);
    m_offset = read_i32 (rec_data + PDB_SYMBOL_REGREL_OFFS_OFFS);
    m_cv_reg = read_u16 (rec_data + PDB_SYMBOL_REGREL_REG_OFFS);
  }

  symbol *create_gdb_sym ()
  {
    allocate_symbol (VAR_DOMAIN);
    sym_set_type ();
    set_register_location (m_cv_reg, m_offset, false);
    return sym;
  }

  void dump ()
  {
    auto tn = type_name ();
    gdb_printf ("  reg=%u+%d ti=%04X", m_cv_reg, m_offset, type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" `%s`\n", name);
  }

private:
  int32_t m_offset;
  uint16_t m_cv_reg;
};

/* S_REGISTER — variable lives directly in a register.  */
struct pdb_register_sym final : pdb_sym
{
  pdb_register_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
		    buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    set_type (PDB_SYMBOL_REG_TYPE_OFFS);
    set_name (PDB_SYMBOL_REG_NAME_OFFS);
    m_cv_reg = read_u16 (rec_data + PDB_SYMBOL_REG_REG_OFFS);
  }

  symbol *create_gdb_sym ()
  {
    allocate_symbol (VAR_DOMAIN);
    sym_set_type ();

    set_register_location (m_cv_reg, 0, true);
    return sym;
  }

  void dump ()
  {
    auto tn = type_name ();
    gdb_printf ("  reg=%u ti=%04X", m_cv_reg, type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" `%s`\n", name);
  }

private:
  uint16_t m_cv_reg;
};

/* S_CONSTANT — compile-time constant value.  */
struct pdb_constant_sym final : pdb_sym
{
  pdb_constant_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
		    buildsym_compunit *cu, uint16_t reclen)
    : pdb_sym (rec_data, rectype, pdb, cu),
      m_reclen (reclen)
  {
    set_type (PDB_SYMBOL_CONST_TYPE_OFFS);

    /* RECLEN counts the 2-byte record kind before the body.  */
    uint32_t body = m_reclen >= 2 ? m_reclen - 2 : 0;
    uint32_t max_len = body > PDB_SYMBOL_CONST_VALUE_OFFS
			 ? body - PDB_SYMBOL_CONST_VALUE_OFFS
			 : 0;
    uint32_t consumed = pdb_cv_read_numeric (rec_data
					       + PDB_SYMBOL_CONST_VALUE_OFFS,
					     max_len, &m_value);
    const char *value_name
      = (consumed > 0
	 ? pdb_extract_string (rec_data + PDB_SYMBOL_CONST_VALUE_OFFS
				 + consumed, rec_data + body)
	 : nullptr);
    name = (value_name != nullptr
	    ? pdb_canonical_name (pdb, value_name) : "");
  }

  symbol *create_gdb_sym (bool local)
  {
    if (cu == nullptr)
      return nullptr;
    allocate_symbol (VAR_DOMAIN);
    sym_set_type ();
    sym->set_loc_class_index (LOC_CONST);
    sym->set_value_longest ((LONGEST) m_value);
    auto &list = local ? cu->get_local_symbols () : cu->get_global_symbols ();
    add_symbol_to_list (sym, list);
    return sym;
  }

  void dump ()
  {
    auto tn = type_name ();
    gdb_printf ("  ti=%04X", type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" val=%" PRIu64 " `%s`\n", m_value, name);
  }

private:
  uint64_t m_value = 0;
  uint16_t m_reclen;
};

/* S_LABEL32 — code label.  */
struct pdb_label_sym final : pdb_sym
{
  pdb_label_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
		 buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    m_sect_offs = read_u32 (rec_data + PDB_SYMBOL_LABEL_OFFS_OFFS);
    m_sect_num = read_u16 (rec_data + PDB_SYMBOL_LABEL_SEG_OFFS);
    set_name (PDB_SYMBOL_LABEL_NAME_OFFS);
  }

  symbol *create_gdb_sym ()
  {
    if (!pdb->section_valid (m_sect_num))
      {
	pdb_complaint ("PDB: bad section %u for label '%s'", m_sect_num, name);
	return nullptr;
      }
    allocate_symbol (LABEL_DOMAIN);
    sym->set_loc_class_index (LOC_LABEL);
    CORE_ADDR addr = pdb->map_section_offset_to_pc (m_sect_num, m_sect_offs);
    sym->set_value_address (addr);
    sym->set_section_index (SECT_OFF_TEXT (pdb->objfile));
    add_symbol_to_list (sym, cu->get_local_symbols ());
    return sym;
  }

  void dump ()
  {
    auto sect_name = pdb->get_section_name (m_sect_num);
    gdb_printf ("  %s:%08X `%s`\n", sect_name, m_sect_offs, name);
  }

private:
  uint32_t m_sect_offs;
  uint16_t m_sect_num;
};

/* An S_UDT binds a name to a type index, and cl emits one for two very
   different reasons.  These are the shapes, with the source that produces
   each and the records that come out:

     struct Hello { };            LF_STRUCTURE `Hello`
				  S_UDT `Hello` -> that record
       The S_UDT only repeats the tag's own name.  It is the tag, not an
       alias of it, so it belongs in STRUCT_DOMAIN.

     using StructAlias = Hello;   S_UDT `StructAlias` -> LF_STRUCTURE `Hello`
       A real alias: the two names differ.  TYPE_DOMAIN, wrapped in a
       TYPE_CODE_TYPEDEF, the shape DWARF gives a DW_TAG_typedef.

     typedef int int32_t;         S_UDT `int32_t` -> T_INT4
       Also an alias, but the reader names the T_INT4 primitive int32_t as
       well, so the two names are equal while the record is not a tag.
       Only a tagged target can be a tag.

     namespace { struct X { }; }  LF_STRUCTURE ``anonymous-namespace'::X`
				  S_UDT `?A0x0e56a160::X`
       One type, two spellings: cl identifies an anonymous namespace by an
       id in a symbol name and by a quoted scope in a type name.  Comparing
       the strings byte for byte calls this an alias and invents a typedef
       that no source declares.  */

/* Pointer past the quoted scope starting at P, or P itself when none
   starts there.  cl opens a quote with a backtick and closes it with an
   apostrophe, `like this', and quotes never nest.  E.g.

     `ns::f'::`2'::<lambda_1>

   The quoted text can itself contain an apostrophe.  */

static const char *
pdb_skip_quoted_scope (const char *p)
{
  if (*p != '`')
    return p;

  for (const char *q = p + 1; *q != '\0'; q++)
    if (*q == '\'' && (q[1] == '\0' || (q[1] == ':' && q[2] == ':')))
      return q + 1;

  return p + strlen (p);
}

/* See pdb-internal.h.  */

size_t
pdb_scope_component_len (const char *name)
{
  const char *p = name;

  while (*p != '\0')
    {
      const char *after = pdb_skip_quoted_scope (p);
      if (after != p)
	p = after;
      else
	p += cp_find_first_component (p);

      if (*p == '\0' || (p[0] == ':' && p[1] == ':'))
	break;

      /* A lone ':' is not a separator; step over it so the scan makes
	 progress.  */
      p++;
    }

  return p - name;
}

/* Recognize an anonymous-namespace id: "?A0x" followed by hex digits.  */

static bool
pdb_is_anon_namespace_id (const char *name, size_t len)
{
  if (len <= 4 || strncmp (name, "?A0x", 4) != 0)
    return false;

  for (size_t i = 4; i < len; i++)
    if (!isxdigit ((unsigned char) name[i]))
      return false;

  return true;
}

/* Recognize only the two anonymous-namespace scope spellings.  */

static bool
pdb_is_anon_namespace_scope (const char *name, size_t len)
{
  static const char *const anon[] = {
    "`anonymous namespace'",
    "`anonymous-namespace'",
  };

  for (const char *a : anon)
    if (len == strlen (a) && strncmp (name, a, len) == 0)
      return true;

  return false;
}

/* Match NAME to TAG, accepting anonymous-namespace ids in NAME.  */

static bool
pdb_udt_name_is_tag (const char *name, const char *tag)
{
  while (*name != '\0' && *tag != '\0')
    {
      size_t nlen = pdb_scope_component_len (name);
      size_t tlen = pdb_scope_component_len (tag);

      bool same = (nlen == tlen && strncmp (name, tag, nlen) == 0);
      if (!same
	  && !(pdb_is_anon_namespace_id (name, nlen)
	       && pdb_is_anon_namespace_scope (tag, tlen)))
	return false;

      name += nlen;
      tag += tlen;
      if (*name == ':')
	name += 2;
      if (*tag == ':')
	tag += 2;
    }

  return *name == *tag;
}

/* Copy NAME's enclosing scope to the objfile, or return nullptr.  */

static const char *
pdb_enclosing_scope (pdb_per_objfile *pdb, const char *name)
{
  if (name == nullptr)
    return nullptr;

  size_t prefix = 0;
  for (const char *p = name; *p != '\0'; )
    {
      size_t len = pdb_scope_component_len (p);
      if (p[len] != ':')
	break;

      prefix = (p - name) + len;
      p += len + 2;
    }

  if (prefix == 0)
    return nullptr;

  return obstack_strndup (&pdb->objfile->objfile_obstack, name, prefix);
}

/* Which block an S_UDT name binding is added to.

   Two TUs can bind the same alias name to different types.  The linker then
   dedups the name: one alias ends up in the global records and is dropped
   from its own TU, while the others stay in their module streams.  The one
   from the global records goes into the CU's global list, where symbol
   lookup finds it after the lookup in the CU fails.  */
enum class pdb_udt_scope
{
  /* Global records -> global block.  */
  global,
  /* Module stream, file scope -> static block.  */
  file,
  /* Module stream, function scope -> local block.  */
  local,
};

/* S_UDT.  */
struct pdb_udt_sym final : pdb_sym
{
  pdb_udt_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
	       buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    set_type (PDB_SYMBOL_UDT_TYPE_OFFS);
    set_name (PDB_SYMBOL_UDT_NAME_OFFS);
  }

  symbol *create_gdb_sym (pdb_udt_scope scope)
  {
    if (cu == nullptr)
      return nullptr;

    type &target = pdb_tpi_resolve_type (pdb, type_index);
    type_code target_code = target.code ();
    bool target_is_tag = (target_code == TYPE_CODE_STRUCT
			  || target_code == TYPE_CODE_UNION
			  || target_code == TYPE_CODE_ENUM);

    /* cl emits an S_UDT both for an alias and for a plain mention of a tag,
       the latter repeating the tag's own name.  Only an alias is a typedef;
       give it the shape DWARF gives a DW_TAG_typedef, a TYPE_CODE_TYPEDEF
       in TYPE_DOMAIN.  A tag stays in STRUCT_DOMAIN, where the type symbols
       built from the TPI also live.  The name alone does not decide it: a
       primitive carries the stdint name an alias of it also has.  */
    if (target_is_tag
	&& target.name () != nullptr
	&& pdb_udt_name_is_tag (name, target.name ()))
      {
	allocate_symbol (STRUCT_DOMAIN);
	sym->set_loc_class_index (LOC_TYPEDEF);
	sym->set_type (&target);
      }
    else
      {
	allocate_symbol (TYPE_DOMAIN);
	sym->set_loc_class_index (LOC_TYPEDEF);

	type_allocator alloc (pdb->objfile, pdb->cur_parser->lang ());
	type *alias = alloc.new_type (TYPE_CODE_TYPEDEF, 0, name);
	alias->set_target_is_stub (true);
	alias->set_target_type (&target);
	sym->set_type (alias);
      }

    std::vector<symbol *> *list;
    if (scope == pdb_udt_scope::global)
      list = &cu->get_global_symbols ();
    else if (scope == pdb_udt_scope::file)
      list = &cu->get_file_symbols ();
    else
      list = &cu->get_local_symbols ();
    add_symbol_to_list (sym, *list);
    return sym;
  }
};

/* S_BLOCK32.  */
struct pdb_block_sym final : pdb_sym
{
  uint32_t code_sz;
  uint32_t sect_offs;
  uint16_t sect_num;

  pdb_block_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
		 buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    code_sz = read_u32 (rec_data + PDB_SYMBOL_BLOCK_CODE_SIZE_OFFS);
    sect_offs = read_u32 (rec_data + PDB_SYMBOL_BLOCK_SECTION_OFFS_OFFS);
    sect_num = read_u16 (rec_data + PDB_SYMBOL_BLOCK_SECTION_NUM_OFFS);
    set_name (PDB_SYMBOL_BLOCK_NAME_OFFS);
  }

  symbol *create_gdb_sym ()
  {
    CORE_ADDR start = pdb->map_section_offset_to_pc (sect_num, sect_offs);
    cu->push_context (start);
    return nullptr;
  }

  void dump ()
  {
    auto sect_name = pdb->get_section_name (sect_num);
    gdb_printf ("  %s:%08X sz=%u `%s`\n", sect_name, sect_offs, code_sz, name);
  }
};

/* S_THUNK32 (0x1102) — thunk record for indirect calls.  */
struct pdb_thunk_sym final : pdb_sym
{
  pdb_thunk_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb)
    : pdb_sym (rec_data, rectype, pdb)
  {
    m_sect_offs = read_u32 (rec_data + PDB_SYMBOL_THUNK_SECTION_OFFS_OFFS);
    m_sect_num = read_u16 (rec_data + PDB_SYMBOL_THUNK_SECTION_NUM_OFFS);
    m_code_sz = read_u16 (rec_data + PDB_SYMBOL_THUNK_CODE_SIZE_OFFS);
    set_name (PDB_SYMBOL_THUNK_NAME_OFFS);
  }

  void dump ()
  {
    auto sect_name = pdb->get_section_name (m_sect_num);
    gdb_printf ("  %s:%08X sz=%u `%s`\n", sect_name, m_sect_offs, m_code_sz,
		name);
  }

private:
  uint32_t m_sect_offs;
  uint16_t m_sect_num;
  uint32_t m_code_sz;
};

/* S_INLINESITE / S_INLINESITE2 — inlined functions.  */
struct pdb_inlinesite_sym final : pdb_sym
{
  pdb_inlinesite_sym (gdb_byte *rec_data, uint16_t rectype,
		      pdb_per_objfile *pdb, buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    m_inlinee_idx = read_u32 (rec_data
			      + PDB_SYMBOL_INLINESITE_INLINEE_IDX_OFFS);
    m_binannot_offs = (rectype == S_INLINESITE2)
			? PDB_SYMBOL_INLINESITE2_BINANNOT_OFFS
			: PDB_SYMBOL_INLINESITE_BINANNOT_OFFS;
  }

  uint32_t inlinee_idx () const
  {
    return m_inlinee_idx;
  }

  /* Byte offset of the binary annotations within the records - differs
     between S_INLINESITE and S_INLINESITE2.  */
  size_t binannot_offs () const
  {
    return m_binannot_offs;
  }

  /* Build the inlined function symbol: a LOC_BLOCK function symbol flagged
     inlined, named INAME with signature SIG_TI (TPI), located at ADDR.  It is
     attached to the inline block by buildsym (set_current_context_function ->
     finish_block).  */
  symbol *create_gdb_sym (const char *iname, uint32_t sig_ti, CORE_ADDR addr)
  {
    allocate_symbol_named (FUNCTION_DOMAIN, addr, iname);
    sym->set_loc_class_index (LOC_BLOCK);
    sym->set_is_inlined (true);
    if (sig_ti != 0)
      sym->set_type (&pdb_tpi_resolve_type (pdb, sig_ti));
    sym->set_section_index (SECT_OFF_TEXT (pdb->objfile));
    return sym;
  }

  void dump (size_t body_size)
  {
    std::string iname;
    pdb_ipi_lookup_inlinee (pdb, m_inlinee_idx, &iname, nullptr);
    gdb_printf ("  inlinee_idx=%04X `%s`", m_inlinee_idx,
		iname.empty () ? "?" : iname.c_str ());

    const gdb_byte *annot = rec_data + m_binannot_offs;
    size_t annot_len = body_size - m_binannot_offs;
    auto chunks = pdb_decode_inline_annotations (annot, annot_len);
    gdb_printf (" ranges=%zu", chunks.size ());
    for (const pdb_inline_chunk &c : chunks)
      {
	gdb_printf (" b%u[+%X,+%X)", c.base_index, c.start_off, c.end_off);
	if (c.file_id == PDB_INLINE_FILE_ID_BASE)
	  gdb_printf (" f:base");
	else
	  gdb_printf (" f:0x%X", c.file_id);
      }
    gdb_printf ("\n");
  }

private:
  uint32_t m_inlinee_idx = 0;
  size_t m_binannot_offs = PDB_SYMBOL_INLINESITE_BINANNOT_OFFS;
};

/* Scope marker (S_END, S_INLINESITE_END, S_PROC_ID_END) — no data.  */
struct pdb_scope_end_sym final : pdb_sym
{
  pdb_scope_end_sym (gdb_byte *rec_data, uint16_t rectype,
		     pdb_per_objfile *pdb)
    : pdb_sym (rec_data, rectype, pdb)
  {
  }

  void dump ()
  {
    gdb_printf ("  (scope end marker)\n");
  }
};

/* Common base for the ranged S_DEFRANGE_* records (REGISTER, REGISTER_REL,
   FRAMEPOINTER_REL).  Holds the shared CV_lvar_addr_range + gap-list parsing,
   the location-building helper and the range/gaps dump tail.  Each derived
   record reads its own record-specific fields and resolves its own GDB
   register number before calling add_location.  */
struct pdb_defrange_sym : pdb_sym
{
  pdb_defrange_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
		    uint16_t reclen, buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu),
      m_reclen (reclen)
  {
  }

protected:
  /* Parse range fields (offStart, isect, cbRange) from a given offset.  */
  void parse_range (uint32_t range_offs)
  {
    m_range.section_offset = read_u32 (rec_data + range_offs
				       + CV_RANGE_OFF_START_OFFS);
    m_range.section_index = read_u16 (rec_data + range_offs
				      + CV_RANGE_ISECT_OFFS);
    m_range.length = read_u16 (rec_data + range_offs + CV_RANGE_CBRANGE_OFFS);
  }

  /* Parse the variable-length gap array from a given offset.  */
  void parse_gaps (uint32_t gaps_offs)
  {
    int data_size = m_reclen - 2;
    int gaps_bytes = data_size - (int) gaps_offs;
    int num_gaps = (gaps_bytes > 0) ? gaps_bytes / 4 : 0;
    const gdb_byte *gaps_ptr = rec_data + gaps_offs;
    for (int i = 0; i < num_gaps; i++)
      {
	auto gap_offset = read_u16 (gaps_ptr);
	auto gap_len = read_u16 (gaps_ptr + 2);
	m_gaps.push_back ({ gap_offset, gap_len });
	gaps_ptr += 4;
      }
  }

  /* Attach the parsed range/gaps to LAST_LOCAL as a location valid across
     [start, start + length) using the already-resolved GDB_REGNUM.  */
  void add_location (symbol *last_local, int gdb_regnum, int32_t offset,
		     bool is_register)
  {
    if (last_local == nullptr)
      return;

    /* Keep the range where the linker put it: the objfile can be relocated
       after this runs, and nothing revisits these entries.  */
    CORE_ADDR start
      = pdb->map_section_offset_unrelocated (m_range.section_index,
					     m_range.section_offset);

    pdb_add_loc_entry (last_local, pdb, start, start + m_range.length,
		       m_range.section_index, gdb_regnum, offset, is_register,
		       false, m_gaps);
  }

  /* Print " range=[sect:start-end]" plus any gaps and a trailing newline.
     The caller prints the record-specific prefix first.  */
  void dump_range_tail ()
  {
    auto sect_name = pdb->get_section_name (m_range.section_index);
    uint32_t end_addr = m_range.section_offset + m_range.length - 1;
    gdb_printf (" range=[%s:%08X-%08X]", sect_name, m_range.section_offset,
		end_addr);
    if (!m_gaps.empty ())
      {
	gdb_printf (" gaps=[");
	for (size_t i = 0; i < m_gaps.size (); ++i)
	  {
	    if (i > 0)
	      gdb_printf (", ");
	    uint32_t gap_start = m_range.section_offset + m_gaps[i].offset;
	    uint32_t gap_end = gap_start + m_gaps[i].length - 1;
	    gdb_printf ("%08X-%08X", gap_start, gap_end);
	  }

	gdb_printf ("]");
      }

    gdb_printf ("\n");
  }

  /* PC range where this location is valid.  */
  struct
  {
    uint32_t section_offset;
    uint16_t section_index;
    uint16_t length;
  } m_range;
  uint16_t m_reclen;
  std::vector<pdb_defrange_gap> m_gaps;
};

/* S_DEFRANGE_REGISTER_REL — register-relative location for S_LOCAL.  */
struct pdb_defrange_regrel_sym final : pdb_defrange_sym
{
  pdb_defrange_regrel_sym (gdb_byte *rec_data, uint16_t rectype,
			   pdb_per_objfile *pdb, uint16_t reclen,
			   buildsym_compunit *cu = nullptr)
    : pdb_defrange_sym (rec_data, rectype, pdb, reclen, cu)
  {
    m_cv_reg = read_u16 (rec_data + PDB_SYMBOL_DEFRANGE_REGREL_REG_OFFS);
    m_flags = read_u16 (rec_data + PDB_SYMBOL_DEFRANGE_REGREL_FLAGS_OFFS);
    m_base_offset = read_i32 (rec_data
			      + PDB_SYMBOL_DEFRANGE_REGREL_OFFSET_OFFS);
    parse_range (PDB_SYMBOL_DEFRANGE_REGREL_RANGE_OFFS);
    parse_gaps (PDB_SYMBOL_DEFRANGE_REGREL_GAPS_OFFS);
  }

  void create_gdb_sym (symbol *last_local)
  {
    /* The record describes one member sitting at its own slot, not the
       object holding it.  Reading the whole type from here would take the
       neighbouring bytes for the other members.  */
    if ((m_flags & PDB_DEFRANGE_REGREL_SPILLED_MEMBER) != 0)
      return;

    add_location (last_local, cv_reg_to_gdb_regnum (pdb, m_cv_reg),
		  m_base_offset, false);
  }

  void dump ()
  {
    gdb_printf ("  reg=%u+%d", m_cv_reg, m_base_offset);
    if ((m_flags & PDB_DEFRANGE_REGREL_SPILLED_MEMBER) != 0)
      gdb_printf (" member@%u", m_flags >> 4);
    dump_range_tail ();
  }

private:
  uint16_t m_cv_reg;
  uint16_t m_flags;
  int32_t m_base_offset;
};

/* S_DEFRANGE_REGISTER — register location for S_LOCAL.  */
struct pdb_defrange_reg_sym final : pdb_defrange_sym
{
  pdb_defrange_reg_sym (gdb_byte *rec_data, uint16_t rectype,
			pdb_per_objfile *pdb, uint16_t reclen,
			buildsym_compunit *cu = nullptr)
    : pdb_defrange_sym (rec_data, rectype, pdb, reclen, cu)
  {
    m_cv_reg = read_u16 (rec_data + PDB_SYMBOL_DEFRANGE_REG_REG_OFFS);
    m_attr = read_u16 (rec_data + PDB_SYMBOL_DEFRANGE_REG_ATTR_OFFS);
    parse_range (PDB_SYMBOL_DEFRANGE_REG_RANGE_OFFS);
    parse_gaps (PDB_SYMBOL_DEFRANGE_REG_GAPS_OFFS);
  }

  void create_gdb_sym (symbol *last_local)
  {
    add_location (last_local, cv_reg_to_gdb_regnum (pdb, m_cv_reg), 0, true);
  }

  void dump ()
  {
    gdb_printf ("  reg=%u", m_cv_reg);
    dump_range_tail ();
  }

private:
  uint16_t m_cv_reg;
  uint16_t m_attr;
};

/* S_DEFRANGE_FRAMEPOINTER_REL — FP-relative location with explicit range
   and gaps.  The frame register is supplied by the caller (resolved from
   the enclosing function's S_FRAMEPROC).  */
struct pdb_defrange_fprel_sym final : pdb_defrange_sym
{
  pdb_defrange_fprel_sym (gdb_byte *rec_data, uint16_t rectype,
			  pdb_per_objfile *pdb, uint16_t reclen,
			  buildsym_compunit *cu = nullptr)
    : pdb_defrange_sym (rec_data, rectype, pdb, reclen, cu)
  {
    m_fp_offset = read_i32 (rec_data + PDB_SYMBOL_DEFRANGE_FPREL_OFFSET_OFFS);
    parse_range (PDB_SYMBOL_DEFRANGE_FPREL_OFFSTART_OFFS);
    parse_gaps (PDB_SYMBOL_DEFRANGE_FPREL_GAPS_OFFS);
  }

  void create_gdb_sym (symbol *last_local, int frame_regnum)
  {
    add_location (last_local, frame_regnum, m_fp_offset, false);
  }

  void dump ()
  {
    gdb_printf ("  fp+%d", m_fp_offset);
    dump_range_tail ();
  }

private:
  int32_t m_fp_offset;
};

/* S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE — FP-relative location for
   S_LOCAL, valid for the entire enclosing function.  */
struct pdb_defrange_fprel_fullscope_sym final : pdb_sym
{
  pdb_defrange_fprel_fullscope_sym (gdb_byte *rec_data, uint16_t rectype,
				    pdb_per_objfile *pdb,
				    buildsym_compunit *cu = nullptr)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    m_fp_offset = read_i32 (rec_data
			    + PDB_SYMBOL_DEFRANGE_FPREL_FULLSCOPE_OFFSET_OFFS);
  }

  void create_gdb_sym (symbol *last_local, int frame_regnum)
  {
    if (last_local == nullptr)
      return;
    pdb_add_loc_entry (last_local, pdb, 0, 0, 0, frame_regnum, m_fp_offset,
		       false,
		       true, {});
  }

  void dump ()
  {
    gdb_printf ("  fp+%d\n", m_fp_offset);
  }

private:
  int32_t m_fp_offset;
};

/* S_PROCREF / S_LPROCREF — reference to a procedure in a module stream.  */
struct pdb_procref_sym : pdb_sym
{
  pdb_procref_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb)
    : pdb_sym (rec_data, rectype, pdb)
  {
    m_sym_offset = read_u32 (rec_data + PDB_SYMBOL_REF_SYM_OFFSET_OFFS);
    m_mod_index = read_u16 (rec_data + PDB_SYMBOL_REF_MOD_INDEX_OFFS);
    set_name (PDB_SYMBOL_REF_NAME_OFFS);
  }

  void dump ()
  {
    gdb_printf ("  mod=%u/%u `%s`\n", m_mod_index, m_sym_offset, name);
  }

private:
  uint32_t m_sym_offset;
  uint16_t m_mod_index;
};

/* S_DATAREF — reference to a data symbol in a module stream.
  Layout matches S_PROCREF, so this reuses pdb_procref_sym parsing.  */
struct pdb_dataref_sym final : pdb_procref_sym
{
  using pdb_procref_sym::pdb_procref_sym;
};

/* Handle S_GPROC32 / S_LPROC32 / S_GPROC32_ID / S_LPROC32_ID.
   Opens a scope; the matching S_END will close it.  */

void
pdb_sym_parser::handle_func_sym (gdb_byte *rec_data, size_t body_size,
				 uint16_t rectype)
{
  if (body_size < PDB_SYMBOL_FUNC_MIN_SIZE)
    {
      pdb_warning (
	"truncated S_*PROC32 record (body=%zu, need >=%u), skipping",
	body_size, (unsigned) PDB_SYMBOL_FUNC_MIN_SIZE);
      return;
    }

  pdb_func_sym fsym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return fsym.dump ();

  if (!pdb->section_valid (fsym.sect_num))
    {
      pdb_complaint ("PDB: bad section %u for function '%s'", fsym.sect_num,
		     fsym.name);
      return;
    }

  auto *sym = fsym.create_gdb_sym ();

  /* Determine how many S_REGREL32 records are function parameters.
     Look up the function's TPI type record to get parmcount.  */
  m_remaining_params = pdb_tpi_get_func_param_count (pdb, fsym.type_index);

  CORE_ADDR start = pdb->map_section_offset_to_pc (fsym.sect_num,
						   fsym.sect_offs);
  CORE_ADDR end = start + fsym.code_sz;
  cu->push_context (start);
  if (sym != nullptr)
    cu->set_current_context_function (sym);

  m_cur_proc_sect = fsym.sect_num;
  m_cur_proc_offs = fsym.sect_offs;

  if (func_ranges != nullptr)
    func_ranges->push_back ({ start, end });

  m_scope_stack.push_back ({ end, {}, pdb_enclosing_scope (pdb, fsym.name) });
}

/* Handle S_PUB32.  Used only for dumping; real S_PUB32 consumption
   happens in pdb_build_minsyms (minimal symbols).  S_PUB32 has no type
   info and its name/address are already covered by the typed S_GPROC32 &
   S_GDATA32 records, so we do not create a GDB symbol here.  */

void
pdb_sym_parser::handle_pub_sym (gdb_byte *rec_data, uint16_t rectype)
{
  if ((flags & PDB_DUMP_SYM) == 0)
    return;

  pdb_pub_sym sym (rec_data, rectype, pdb, cu);
  sym.dump ();
}

/* Handle S_GDATA32 / S_LDATA32.  */

void
pdb_sym_parser::handle_var_sym (gdb_byte *rec_data, uint16_t rectype)
{
  pdb_var_sym sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym (!m_scope_stack.empty ());
}

/* Handle S_LOCAL / S_LOCAL32.  Updates m_last_local so subsequent
   S_DEFRANGE* records can add location entries to the symbol's baton.
   Allocates the baton with entries=nullptr; DEFRANGE handlers fill it in.  */

void
pdb_sym_parser::handle_local_sym (gdb_byte *rec_data, uint16_t rectype)
{
  pdb_local_sym sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    {
      sym.dump ();
      m_last_local = nullptr;
      return;
    }

  sym.set_deref_param (m_lang == language_c);

  auto *gdb_sym = sym.create_gdb_sym ();

  /* If the compiler marked this variable as optimized out, keep it as
     LOC_OPTIMIZED_OUT and leave m_last_local null so subsequent DEFRANGE
     records are skipped.  */
  if (sym.flags & CV_LVARFLAG_OptOut)
    {
      add_symbol_to_list (gdb_sym, cu->get_local_symbols ());
      m_last_local = nullptr;
      return;
    }

  /* Allocate the loclist baton with no entries.  S_DEFRANGE* records will add
     parsed entries via pdb_add_loc_entry.  */
  pdb_loclist_baton *baton = OBSTACK_ZALLOC (&pdb->objfile->objfile_obstack,
					     pdb_loclist_baton);
  baton->entries = nullptr;
  baton->pdb = pdb;
  baton->deref = sym.deref_param ();

  SYMBOL_LOCATION_BATON (gdb_sym) = baton;
  gdb_sym->set_loc_class_index (pdb_loclist_index);
  add_symbol_to_list (gdb_sym, cu->get_local_symbols ());
  m_last_local = gdb_sym;
}

/* Handle S_REGREL32 — register-relative local variable.
   This is the record LLVM actually emits for local variables in PDBs.
   Self-contained: register + offset + type + name in one record.  */

void
pdb_sym_parser::handle_regrel_sym (gdb_byte *rec_data, uint16_t rectype)
{
  pdb_regrel_sym sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  /* Skip the legacy S_REGREL32 if S_LOCAL already created this variable.  */
  if (local_already_defined (sym.name))
    return;

  auto *gdb_sym = sym.create_gdb_sym ();

  /* The first m_remaining_params S_REGREL32 records after a function start
     are parameters (including implicit 'this' for member functions).  */
  if (m_remaining_params > 0)
    {
      gdb_sym->set_is_argument (true);
      m_remaining_params--;
    }
}

/* Handle S_REGISTER — variable in a register.  */

void
pdb_sym_parser::handle_register_sym (gdb_byte *rec_data, uint16_t rectype)
{
  pdb_register_sym sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  /* Skip if S_LOCAL already created this variable.  */
  if (local_already_defined (sym.name))
    return;

  sym.create_gdb_sym ();
}

/* Handle S_CONSTANT — compile-time constant.  */

void
pdb_sym_parser::handle_const_sym (gdb_byte *rec_data, uint16_t rectype,
				  uint16_t reclen)
{
  pdb_constant_sym sym (rec_data, rectype, pdb, cu, reclen);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym (!m_scope_stack.empty ());
}

/* Handle S_LABEL32.  */

void
pdb_sym_parser::handle_label_sym (gdb_byte *rec_data, uint16_t rectype)
{
  pdb_label_sym sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym ();
}

/* Handle S_UDT.  FROM_GLOBALS is true when the record comes from the
   global records.  */

void
pdb_sym_parser::handle_udt_sym (gdb_byte *rec_data, uint16_t rectype,
				bool from_globals)
{
  pdb_udt_sym sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  pdb_udt_scope scope;
  if (from_globals)
    {
      /* First record in stream order wins, in every mode.  */
      if (!pdb->global_udt_names.insert (std::string_view (sym.name)).second)
	return;
      /* <pdb-types> may already hold this name, depending on which of the
	 two is built first; either one alone answers a type lookup.  */
      if (pdb->built_type_names.count (std::string_view (sym.name)) != 0)
	return;
      scope = pdb_udt_scope::global;
    }
  else if (m_scope_stack.empty ())
    scope = pdb_udt_scope::file;
  else
    scope = pdb_udt_scope::local;
  sym.create_gdb_sym (scope);
}

/* Handle S_BLOCK32.
   Opens a nested scope; the matching S_END will close it.  */

void
pdb_sym_parser::handle_block_sym (gdb_byte *rec_data, uint16_t rectype)
{
  pdb_block_sym sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  CORE_ADDR start = pdb->map_section_offset_to_pc (sym.sect_num,
						   sym.sect_offs);
  CORE_ADDR end = start + sym.code_sz;
  cu->push_context (start);
  m_scope_stack.push_back ({ end });
}

/* Handle S_THUNK32.  Opens a scope; the matching S_END will close it.  */

void
pdb_sym_parser::handle_thunk_sym (gdb_byte *rec_data, uint16_t rectype)
{
  pdb_thunk_sym sym (rec_data, rectype, pdb);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  cu->push_context (0);
  m_scope_stack.push_back ({ 0 });
}

/* Handle S_INLINESITE.  Decode the binary annotations into the inlined body's
   code ranges, resolve the inlinee name/signature from the IPI stream, and
   open a scope owned by an inlined LOC_BLOCK function symbol.  Falls back to a
   plain empty scope when ranges or the inlinee cannot be recovered, so a
   malformed record never desynchronises the scope stack.  S_INLINESITE_END
   closes the scope and applies the ranges to the block.  */

void
pdb_sym_parser::handle_inlinesite_sym (gdb_byte *rec_data, size_t body_size,
				       uint16_t rectype)
{
  pdb_inlinesite_sym sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump (body_size);

  const gdb_byte *annot = rec_data + sym.binannot_offs ();
  size_t annot_len = body_size - sym.binannot_offs ();
  std::vector<pdb_inline_chunk> chunks
    = pdb_decode_inline_annotations (annot, annot_len);
  std::vector<pdb_code_origin> bases = { { m_cur_proc_sect, m_cur_proc_offs } };
  pdb_range_pair_vec ranges = pdb_inline_chunk_ranges (pdb, chunks, bases);

  std::string iname;
  uint32_t sig_ti = 0;
  pdb_ipi_lookup_inlinee (pdb, sym.inlinee_idx (), &iname, &sig_ti);

  /* If no range build an empty frame so that S_INLINESITE_END has something
     to close.  */
  if (ranges.empty () || iname.empty ())
    {
      cu->push_context (0);
      m_scope_stack.push_back ({ 0 });
      return;
    }

  CORE_ADDR lo = ranges.front ().first;
  CORE_ADDR hi = ranges.front ().second;
  for (const auto &r : ranges)
    {
      lo = std::min (lo, r.first);
      hi = std::max (hi, r.second);
    }

  symbol *inline_sym = sym.create_gdb_sym (iname.c_str (), sig_ti, lo);

  /* find_frame_sal takes the *caller* frame's source location from the
     inlined symbol's line, so without it a backtrace shows the enclosing
     function with no file:line.

     The lookup is an exact PC match against the caller's raw C13 table, so
     it only succeeds for a site beginning at a line entry.  Several calls
     inlined into one statement leave the later sites unmatched, and those
     keep line 0.  Widening this to the covering line is wrong: it gives
     every site the statement's line, and GDB's same-line rule then steps
     straight through them.  The gap closes at the producer -- clang's
     -gcolumn-info emits an entry per site.  */
  queue_line_for (inline_sym, lo, /*with_line=*/true);

  /* Put the inlinee in the caller's scope so name lookup can reach it, as
     DWARF does for DW_TAG_inlined_subroutine.  Must precede push_context,
     which moves the caller's list away.  */
  add_symbol_to_list (inline_sym, cu->get_local_symbols ());

  cu->push_context (lo);
  cu->set_current_context_function (inline_sym);

  /* Emit line-table entries inside the inline block so find_pc_line lands in
     it (needed for inline breakpoints and frame unwinding).  */
  std::vector<const std::vector<pdb_inline_line_span> *> enclosing;
  for (auto frame = m_scope_stack.rbegin (); frame != m_scope_stack.rend ();
       ++frame)
    if (!frame->line_spans.empty ())
      enclosing.push_back (&frame->line_spans);

  pdb_scope_frame frame { hi, std::move (ranges) };
  pdb_record_inline_lines (pdb, m_mod_info, cu, sym.inlinee_idx (),
			   bases, chunks, enclosing, m_sym_lines,
			   &frame.line_spans);

  m_scope_stack.push_back (std::move (frame));
}

/* Handle S_END / S_INLINESITE_END / S_PROC_ID_END.
   Close the innermost open scope.  */

void
pdb_sym_parser::handle_scope_end_sym (gdb_byte *rec_data, uint16_t rectype)
{
  pdb_scope_end_sym sym (rec_data, rectype, pdb);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  if (!m_scope_stack.empty ())
    {
      pdb_scope_frame frame = std::move (m_scope_stack.back ());
      m_scope_stack.pop_back ();
      block *b = cu->pop_context (frame.end);
      if (b != nullptr && frame.scope != nullptr)
	b->set_scope (frame.scope, &pdb->objfile->objfile_obstack);
      if (b != nullptr && !frame.ranges.empty ())
	{
	  std::vector<blockrange> rangevec;
	  rangevec.reserve (frame.ranges.size ());
	  for (const auto &r : frame.ranges)
	    rangevec.emplace_back (r.first, r.second);
	  b->set_ranges (make_blockranges (pdb->objfile, rangevec));
	}
    }
}

/* Handle S_DEFRANGE_REGISTER_REL — attach register-relative location
   to the most recent S_LOCAL symbol.  On success, adds the symbol to the
   local symbols list.  */

void
pdb_sym_parser::handle_defrange_regrel (gdb_byte *rec_data, uint16_t reclen)
{
  pdb_defrange_regrel_sym sym (rec_data, S_DEFRANGE_REGISTER_REL, pdb, reclen,
			       cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym (m_last_local);
}

/* Handle S_DEFRANGE_REGISTER — attach register location
   to the most recent S_LOCAL symbol.  */

void
pdb_sym_parser::handle_defrange_reg (gdb_byte *rec_data, uint16_t reclen)
{
  pdb_defrange_reg_sym sym (rec_data, S_DEFRANGE_REGISTER, pdb, reclen, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym (m_last_local);
}

/* Handle S_DEFRANGE_FRAMEPOINTER_REL — FP-relative with range/gaps.  */

void
pdb_sym_parser::handle_defrange_fprel (gdb_byte *rec_data, uint16_t reclen)
{
  pdb_defrange_fprel_sym sym (rec_data, S_DEFRANGE_FRAMEPOINTER_REL, pdb,
			      reclen, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym (m_last_local, frame_regnum_for (m_last_local));
}

/* Handle S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE — attach FP-relative
   location (valid for the whole function) to the most recent S_LOCAL.  */

void
pdb_sym_parser::handle_defrange_fprel_fullscope (gdb_byte *rec_data)
{
  pdb_defrange_fprel_fullscope_sym sym (rec_data,
					S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE,
					pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym (m_last_local, frame_regnum_for (m_last_local));
}

/* Handle S_PROCREF / S_LPROCREF.  */

void
pdb_sym_parser::handle_procref_sym (gdb_byte *rec_data, uint16_t rectype)
{
  pdb_procref_sym rsym (rec_data, rectype, pdb);

  if (flags & PDB_DUMP_SYM)
    rsym.dump ();
}

/* Handle S_DATAREF.  */

void
pdb_sym_parser::handle_dataref_sym (gdb_byte *rec_data, uint16_t rectype)
{
  pdb_dataref_sym rsym (rec_data, rectype, pdb);

  if (flags & PDB_DUMP_SYM)
    rsym.dump ();
}

/* S_FRAMEPROC — extract frame pointer registers for current function.
   Updates cur_frame_gdb_regnum / cur_param_frame_gdb_regnum (GDB regnums)
   so subsequent register conversions can resolve CV_REG_VFRAME via the
   gdbarch_codeview_local_base_pointer_regnum hook, with a fallback to
   gdbarch_codeview_default_frame_regnum.  The record bases locals and
   parameters separately, and a frame that moves the stack pointer after
   the prologue gives them different registers.  */

void
pdb_sym_parser::handle_frameproc_sym (gdb_byte *rec_data)
{
  gdbarch *gdbarch = pdb->objfile->arch ();
  uint32_t fp_flags = read_u32 (rec_data + PDB_SYMBOL_FRAMEPROC_FLAGS_OFFS);

  auto decode = [&] (int shift) -> int
    {
      uint32_t enc = (fp_flags >> shift) & PDB_FRAMEPROC_BP_MASK;

      int regnum = -1;
      if (gdbarch_codeview_local_base_pointer_regnum_p (gdbarch))
	regnum = gdbarch_codeview_local_base_pointer_regnum (gdbarch, enc);
      if (regnum < 0 && gdbarch_codeview_default_frame_regnum_p (gdbarch))
	regnum = gdbarch_codeview_default_frame_regnum (gdbarch);

      return regnum;
    };

  int regnum = decode (PDB_FRAMEPROC_LOCAL_BP_SHIFT);
  int param_regnum = decode (PDB_FRAMEPROC_PARAM_BP_SHIFT);

  cur_frame_gdb_regnum = regnum;
  cur_param_frame_gdb_regnum = param_regnum;

  if (flags & PDB_DUMP_SYM)
    {
      const char *fp_name
	= (regnum >= 0 ? gdbarch_register_name (gdbarch, regnum) : "?");
      const char *pfp_name
	= (param_regnum >= 0 ? gdbarch_register_name (gdbarch, param_regnum)
			     : "?");
      gdb_printf ("  frame size:%u local_fp:%s param_fp:%s\n",
		  read_u32 (rec_data + PDB_SYMBOL_FRAMEPROC_FRAME_SIZE_OFFS),
		  fp_name, pfp_name);
    }
}

/* Handle S_UNAMESPACE — using namespace directive.  */

void
pdb_sym_parser::handle_unamespace_sym (const gdb_byte *rec_data)
{
  auto ns = CSTR (rec_data + PDB_SYMBOL_UNAMESPACE_NAME_OFFS);
  if (flags & PDB_DUMP_SYM)
    {
      gdb_printf ("  `%s`\n", ns);
      return;
    }

  const char *ns_copy = obstack_strdup (&pdb->objfile->objfile_obstack, ns);
  auto **directives = m_scope_stack.empty ()
			? cu->get_global_using_directives ()
			: cu->get_local_using_directives ();
  add_using_directive (directives, "", ns_copy, nullptr, nullptr,
		       std::vector<const char *> (), 0,
		       &pdb->objfile->objfile_obstack);
  register_namespace (ns_copy);
}

void
pdb_sym_parser::register_namespace (const char *ns_name)
{
  if (ns_name == nullptr || cu == nullptr || *ns_name == '\0')
    return;
  if (!m_namespaces_seen.insert (ns_name).second)
    return;

  type *ns_type = type_allocator (pdb->objfile, language_cplus)
		    .new_type (TYPE_CODE_NAMESPACE, 0, ns_name);

  symbol *ns_sym = pdb->objfile->new_symbol<symbol> ();
  ns_sym->set_language (language_cplus, &pdb->objfile->objfile_obstack);
  ns_sym->compute_and_set_names (ns_name, true, pdb->objfile->per_bfd);
  ns_sym->set_domain (TYPE_DOMAIN);
  ns_sym->set_loc_class_index (LOC_TYPEDEF);
  ns_sym->set_type (ns_type);
  add_symbol_to_list (ns_sym, cu->get_global_symbols ());
}

/* Whether C can appear in a scope name.  cl also builds identifiers from
   '?', '@' and '$'.  */

static bool
pdb_scope_name_char (unsigned char c)
{
  return isalnum (c) || c == '_' || c == '?' || c == '@' || c == '$';
}

void
pdb_for_each_scope_prefix (const char *qname,
			   gdb::function_view<void (std::string_view)> f)
{
  if (qname == nullptr)
    return;

  /* A namespace is named by an identifier, so the prefixes stop at the
     first scope that is not one: a template, a function signature or a
     quoted local scope, whose own scopes are classes or functions.  */
  for (const char *p = qname; *p != '\0'; )
    {
      size_t len = pdb_scope_component_len (p);
      if (p[len] != ':')
	return;

      for (size_t i = 0; i < len; i++)
	if (!pdb_scope_name_char ((unsigned char) p[i]))
	  return;

      f (std::string_view (qname, p + len - qname));
      p += len + 2;
    }
}

/* A quoted scope is split at the "::" it contains, which costs an extra
   pass but leaves the last "::" where it is.  A name whose final component
   is quoted would come out short and need pdb_skip_quoted_scope here; no
   such name has been seen.  */

size_t
pdb_last_component_offset (const char *qname)
{
  if (qname == nullptr)
    return 0;

  size_t previous_len = 0;
  for (unsigned int current_len = cp_find_first_component (qname);
       qname[current_len] != '\0';
       current_len += cp_find_first_component (qname + current_len))
    {
      current_len += 2;
      previous_len = current_len;
    }

  return previous_len;
}

/* Whether NAME names a tagged type.  A symbol name identifies an anonymous
   namespace by an id where a type name quotes it, so the plain query misses
   every class declared in an anonymous namespace.  */

static bool
pdb_prefix_is_tagged_type (pdb_tpi_context *tpi, const std::string &prefix)
{
  if (pdb_tpi_is_tagged_type_name (tpi, prefix.c_str ()))
    return true;

  if (prefix.find ("?A0x") == std::string::npos)
    return false;

  static const char *const anon[] = {
    "`anonymous-namespace'",
    "`anonymous namespace'",
  };

  for (const char *a : anon)
    {
      std::string alt;
      alt.reserve (prefix.size () + 32);

      for (size_t i = 0; i < prefix.size (); )
	{
	  const char *comp = prefix.c_str () + i;
	  size_t len = pdb_scope_component_len (comp);

	  if (pdb_is_anon_namespace_id (comp, len))
	    alt += a;
	  else
	    alt.append (prefix, i, len);

	  i += len;
	  if (i < prefix.size ())
	    {
	      alt += "::";
	      i += 2;
	    }
	}

      if (pdb_tpi_is_tagged_type_name (tpi, alt.c_str ()))
	return true;
    }

  return false;
}

void
pdb_sym_parser::ensure_namespaces_for (const char *qname, uint32_t type_index)
{
  /* The property word describes the tag, so it answers for QNAME only when
     QNAME is the tag's own name.  An S_UDT that aliases a tag carries the
     alias's scopes, which are unrelated to the target's.  A symbol names an
     anonymous namespace by an id where the tag quotes it, so the two spell
     one name two ways.  */
  const char *tag = pdb_tpi_tag_name (&pdb->tpi, type_index);
  if (tag != nullptr && qname != nullptr && pdb_udt_name_is_tag (qname, tag))
    {
      uint16_t props = pdb_tpi_tag_props (&pdb->tpi, type_index);

      if ((props & CV_PROP_SCOPED) != 0)
	return;

      if ((props & CV_PROP_ISNESTED) == 0)
	{
	  pdb_for_each_scope_prefix (qname, [&] (std::string_view scope)
	    {
	      register_namespace (std::string (scope).c_str ());
	    });
	  return;
	}

      /* A class encloses it, and the nested-type index says which one, so
	 the scopes are those of that class.  */
      auto owner = pdb->tpi.nested_owner.find (type_index);
      if (owner != pdb->tpi.nested_owner.end ())
	{
	  const char *owner_name = pdb_tpi_tag_name (&pdb->tpi,
						     owner->second);
	  if (owner_name != nullptr && owner_name[0] != '\0')
	    {
	      ensure_namespaces_for (owner_name, owner->second);
	      return;
	    }
	}
    }

  /* A method declares no scope of its own, so its namespaces are its
     class's.  LF_MFUNCTION names that class outright.  */
  uint32_t class_ti = pdb_tpi_mfunction_class (&pdb->tpi, type_index);
  if (class_ti != 0)
    {
      const char *cls = pdb_tpi_tag_name (&pdb->tpi, class_ti);
      if (cls != nullptr && cls[0] != '\0')
	{
	  ensure_namespaces_for (cls, class_ti);
	  return;
	}
    }

  /* A data symbol's type index is its own type, so it names no scope.  A
     static data member is declared in its class's field list, and that
     declaration is indexed under the same qualified name.  */
  if (qname != nullptr)
    {
      auto owner = pdb->tpi.static_member_owner.find (qname);
      if (owner != pdb->tpi.static_member_owner.end ())
	{
	  const char *cls = pdb_tpi_tag_name (&pdb->tpi, owner->second);
	  if (cls != nullptr && cls[0] != '\0')
	    {
	      ensure_namespaces_for (cls, owner->second);
	      return;
	    }
	}
    }

  pdb_for_each_scope_prefix (qname, [&] (std::string_view scope)
    {
      std::string prefix (scope);

      /* Already-classified prefixes short-circuit the TPI lookup.  */
      if (m_namespaces_seen.count (prefix) != 0
	  || m_tag_prefixes_seen.count (prefix) != 0)
	return;

      if (pdb_prefix_is_tagged_type (&pdb->tpi, prefix))
	m_tag_prefixes_seen.insert (std::move (prefix));
      else
	register_namespace (prefix.c_str ());
    });
}

/* Dump name for symbol types not yet handled by the reader.  */

static void
handle_sym_dump (pdb_per_objfile * /*pdb*/, const gdb_byte *rec_data,
		 uint32_t name_offset, pdb_sym_flags flags)
{
  if (flags & PDB_DUMP_SYM)
    gdb_printf ("  `%s` {unsupported}\n", CSTR (rec_data + name_offset));
}

std::string
pdb_sym_rec_type_name (uint16_t rectype)
{
  switch (rectype)
    {
    case S_GPROC32:
      return "S_GPROC32";
    case S_LPROC32:
      return "S_LPROC32";
    case S_GPROC32_ID:
      return "S_GPROC32_ID";
    case S_LPROC32_ID:
      return "S_LPROC32_ID";
    case S_GDATA32:
      return "S_GDATA32";
    case S_LDATA32:
      return "S_LDATA32";
    case S_LOCAL:
      return "S_LOCAL";
    case S_UDT:
      return "S_UDT";
    case S_BLOCK32:
      return "S_BLOCK32";
    case S_REGREL32:
      return "S_REGREL32";
    case S_PUB32:
      return "S_PUB32";
    case S_REGISTER:
      return "S_REGISTER";
    case S_CONSTANT:
      return "S_CONSTANT";
    case S_LABEL32:
      return "S_LABEL32";
    case S_END:
      return "S_END";
    case S_INLINESITE:
      return "S_INLINESITE";
    case S_INLINESITE_END:
      return "S_INLINESITE_END";
    case S_PROC_ID_END:
      return "S_PROC_ID_END";
    case S_THUNK32:
      return "S_THUNK32";
    case S_DEFRANGE_REGISTER:
      return "S_DEFRANGE_REGISTER";
    case S_DEFRANGE_REGISTER_REL:
      return "S_DEFRANGE_REGISTER_REL";
    case S_PROCREF:
      return "S_PROCREF";
    case S_LPROCREF:
      return "S_LPROCREF";
    case S_DATAREF:
      return "S_DATAREF";
    case S_ANNOTATIONREF:
      return "S_ANNOTATIONREF";
    case S_OBJNAME:
      return "S_OBJNAME";
    case S_COMPILE2:
      return "S_COMPILE2";
    case S_COMPILE3:
      return "S_COMPILE3";
    case S_ENVBLOCK:
      return "S_ENVBLOCK";
    case S_UNAMESPACE:
      return "S_UNAMESPACE";
    case S_BPREL32:
      return "S_BPREL32";
    case S_LTHREAD32:
      return "S_LTHREAD32";
    case S_GTHREAD32:
      return "S_GTHREAD32";
    case S_LMANDATA:
      return "S_LMANDATA";
    case S_GMANDATA:
      return "S_GMANDATA";
    case S_BUILDINFO:
      return "S_BUILDINFO";
    case S_FRAMEPROC:
      return "S_FRAMEPROC";
    case S_CALLSITEINFO:
      return "S_CALLSITEINFO";
    case S_FILESTATIC:
      return "S_FILESTATIC";
    case S_EXPORT:
      return "S_EXPORT";
    case S_SECTION:
      return "S_SECTION";
    case S_COFFGROUP:
      return "S_COFFGROUP";
    case S_TRAMPOLINE:
      return "S_TRAMPOLINE";
    case S_FRAMECOOKIE:
      return "S_FRAMECOOKIE";
    case S_HEAPALLOCSITE:
      return "S_HEAPALLOCSITE";
    case S_CALLEES:
      return "S_CALLEES";
    case S_CALLERS:
      return "S_CALLERS";
    case S_POGODATA:
      return "S_POGODATA";
    case S_INLINESITE2:
      return "S_INLINESITE2";
    case S_TOKENREF:
      return "S_TOKENREF";
    case S_GMANPROC:
      return "S_GMANPROC";
    case S_LMANPROC:
      return "S_LMANPROC";
    case S_COBOLUDT:
      return "S_COBOLUDT";
    case S_MANCONSTANT:
      return "S_MANCONSTANT";
    case S_SEPCODE:
      return "S_SEPCODE";
    case S_DISCARDED:
      return "S_DISCARDED";
    case S_ANNOTATION:
      return "S_ANNOTATION";
    case S_DEFRANGE:
      return "S_DEFRANGE";
    case S_DEFRANGE_SUBFIELD:
      return "S_DEFRANGE_SUBFIELD";
    case S_DEFRANGE_FRAMEPOINTER_REL:
      return "S_DEFRANGE_FRAMEPOINTER_REL";
    case S_DEFRANGE_SUBFIELD_REGISTER:
      return "S_DEFRANGE_SUBFIELD_REGISTER";
    case S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE:
      return "S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE";
    case S_LOCAL_2005:
      return "S_LOCAL_2005";
    case S_DEFRANGE_2005:
      return "S_DEFRANGE_2005";
    case S_DEFRANGE2_2005:
      return "S_DEFRANGE2_2005";
    case S_ARMSWITCHTABLE:
      return "S_ARMSWITCHTABLE";
    case S_MOD_TYPEREF:
      return "S_MOD_TYPEREF";
    case S_REF_MINIPDB:
      return "S_REF_MINIPDB";
    case S_PDBMAP:
      return "S_PDBMAP";
    case S_LPROC32_DPC:
      return "S_LPROC32_DPC";
    case S_LPROC32_DPC_ID:
      return "S_LPROC32_DPC_ID";
    case S_INLINEES:
      return "S_INLINEES";
    case S_FASTLINK:
      return "S_FASTLINK";
    case S_HOTPATCHFUNC:
      return "S_HOTPATCHFUNC";
    case S_FRAMEREG:
      return "S_FRAMEREG";
    case S_ATTR_FRAMEREL:
      return "S_ATTR_FRAMEREL";
    case S_ATTR_REGISTER:
      return "S_ATTR_REGISTER";
    case S_ATTR_REGREL:
      return "S_ATTR_REGREL";
    case S_ATTR_MANYREG:
      return "S_ATTR_MANYREG";
    case S_VFTABLE32:
      return "S_VFTABLE32";
    case S_WITH32:
      return "S_WITH32";
    case S_MANYREG:
      return "S_MANYREG";
    case S_MANYREG2:
      return "S_MANYREG2";
    case S_LOCALSLOT:
      return "S_LOCALSLOT";
    case S_PARAMSLOT:
      return "S_PARAMSLOT";
    default:
      return string_printf ("(UNDEFINED:%04x)", rectype);
    }
}

/* Minimum number of record-body bytes (the bytes following the 2-byte
   record type) required to safely read RECTYPE's fixed fields.  For a
   record whose name is its last field (pdb_sym_has_trailing_name), this is
   the name's offset plus one.  Returns 0 for records that carry no fixed
   fields.  */

static size_t
pdb_sym_min_body_size (uint16_t rectype)
{
  switch (rectype)
    {
    case S_GPROC32:
    case S_LPROC32:
    case S_GPROC32_ID:
    case S_LPROC32_ID:
      return PDB_SYMBOL_FUNC_NAME_OFFS + 1;
    case S_PUB32:
      return PDB_SYMBOL_PUB_NAME_OFFS + 1;
    case S_GDATA32:
    case S_LDATA32:
      return PDB_SYMBOL_VAR_NAME_OFFS + 1;
    case S_LOCAL32:
      return PDB_SYMBOL_LOCAL_NAME_OFFS + 1;
    case S_REGREL32:
      return PDB_SYMBOL_REGREL_NAME_OFFS + 1;
    case S_REGISTER:
      return PDB_SYMBOL_REG_NAME_OFFS + 1;
    case S_LABEL32:
      return PDB_SYMBOL_LABEL_NAME_OFFS + 1;
    case S_UDT:
      return PDB_SYMBOL_UDT_NAME_OFFS + 1;
    case S_BLOCK32:
      return PDB_SYMBOL_BLOCK_NAME_OFFS + 1;
    case S_THUNK32:
      return PDB_SYMBOL_THUNK_NAME_OFFS + 1;
    case S_INLINESITE:
      return PDB_SYMBOL_INLINESITE_INLINEE_IDX_OFFS + 4;
    case S_INLINESITE2:
      return PDB_SYMBOL_INLINESITE2_BINANNOT_OFFS;
    case S_UNAMESPACE:
      return PDB_SYMBOL_UNAMESPACE_NAME_OFFS + 1;
    case S_FRAMEPROC:
      return PDB_SYMBOL_FRAMEPROC_FLAGS_OFFS + 4;
    case S_CONSTANT:
      /* The type index and the smallest numeric leaf.  */
      return PDB_SYMBOL_CONST_VALUE_OFFS + 2;
    case S_DEFRANGE_REGISTER:
      return PDB_SYMBOL_DEFRANGE_REG_GAPS_OFFS;
    case S_DEFRANGE_REGISTER_REL:
      return PDB_SYMBOL_DEFRANGE_REGREL_GAPS_OFFS;
    case S_DEFRANGE_FRAMEPOINTER_REL:
      return PDB_SYMBOL_DEFRANGE_FPREL_GAPS_OFFS;
    case S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE:
      return PDB_SYMBOL_DEFRANGE_FPREL_FULLSCOPE_OFFSET_OFFS + 4;
    default:
      return 0;
    }
}

/* Whether RECTYPE's last field is a NUL-terminated name starting at
   pdb_sym_min_body_size (RECTYPE) - 1.  */

static bool
pdb_sym_has_trailing_name (uint16_t rectype)
{
  switch (rectype)
    {
    case S_GPROC32:
    case S_LPROC32:
    case S_GPROC32_ID:
    case S_LPROC32_ID:
    case S_PUB32:
    case S_GDATA32:
    case S_LDATA32:
    case S_LOCAL32:
    case S_REGREL32:
    case S_REGISTER:
    case S_LABEL32:
    case S_UDT:
    case S_BLOCK32:
    case S_THUNK32:
    case S_UNAMESPACE:
      return true;
    default:
      return false;
    }
}

/* Whether the BODY_SIZE-byte body at BODY of a RECTYPE record holds its
   fixed fields and, if it has one, its whole name.  */

static bool
pdb_sym_body_valid (uint16_t rectype, const gdb_byte *body, size_t body_size)
{
  size_t min_body = pdb_sym_min_body_size (rectype);
  if (body_size < min_body)
    return false;

  return (!pdb_sym_has_trailing_name (rectype)
	  || memchr (body + min_body - 1, 0, body_size - (min_body - 1))
	       != nullptr);
}

/* CV_CFL_LANG codes carried in the iLanguage field of S_COMPILE*.  */
inline constexpr uint8_t CV_CFL_C = 0x00;
inline constexpr uint8_t CV_CFL_CXX = 0x01;

/* iLanguage is the low byte of the flags dword, which is the first
   field of S_COMPILE*.  */
inline constexpr auto PDB_SYMBOL_COMPILE_FLAGS_OFFS = 0;

/* Offset of the compiler version string, which follows the fixed fields.
   S_COMPILE3 carries one more version word than S_COMPILE2.  */
inline constexpr auto PDB_SYMBOL_COMPILE2_VERSTR_OFFS = 18;
inline constexpr auto PDB_SYMBOL_COMPILE3_VERSTR_OFFS = 22;

/* Map a CV_CFL_LANG code to a GDB language.  */
static enum language
pdb_cv_lang_to_language (uint8_t cv_lang)
{
  switch (cv_lang)
    {
    case CV_CFL_C:
      return language_c;
    case CV_CFL_CXX:
      return language_cplus;
    default:
      return language_cplus;
    }
}

/* See pdb-internal.h.  */

enum language
pdb_module_language (pdb_per_objfile *pdb, pdb_module_info *mod_info,
		     const gdb_byte *module_stream)
{
  if (mod_info->language != language_unknown)
    return mod_info->language;

  gdb::byte_vector stream_buf;
  if (module_stream == nullptr)
    {
      stream_buf = pdb_read_module_stream (pdb, mod_info);
      if (stream_buf.empty ())
	{
	  mod_info->language = language_cplus;
	  return language_cplus;
	}
      module_stream = stream_buf.data ();
    }

  const gdb_byte *data = module_stream + PDB_MODULE_SYMBOLS_OFFS;
  const gdb_byte *syms_end = module_stream + mod_info->sym_byte_size;

  enum language lang = language_cplus;
  while (data + PDB_RECORD_HDR_SIZE <= syms_end)
    {
      auto hdr = pdb_parse_sym_record_hdr (data, syms_end);
      if (!hdr)
	break;

      const gdb_byte *rec_data = data + PDB_RECORD_DATA_OFFS;
      if ((hdr->type == S_COMPILE2 || hdr->type == S_COMPILE3)
	  && hdr->rec_size () - PDB_RECORD_DATA_OFFS
	       > PDB_SYMBOL_COMPILE_FLAGS_OFFS)
	{
	  lang = pdb_cv_lang_to_language (
	    read_u8 (rec_data + PDB_SYMBOL_COMPILE_FLAGS_OFFS));
	  break;
	}

      data += hdr->rec_size ();
    }

  mod_info->language = lang;
  return lang;
}

/* See pdb-internal.h.  */

const char *
pdb_module_producer (pdb_per_objfile *pdb, pdb_module_info *mod_info,
		     const gdb_byte *module_stream)
{
  if (module_stream == nullptr)
    return nullptr;

  const gdb_byte *data = module_stream + PDB_MODULE_SYMBOLS_OFFS;
  const gdb_byte *syms_end = module_stream + mod_info->sym_byte_size;

  while (data + PDB_RECORD_HDR_SIZE <= syms_end)
    {
      auto hdr = pdb_parse_sym_record_hdr (data, syms_end);
      if (!hdr)
	break;

      if (hdr->type == S_COMPILE2 || hdr->type == S_COMPILE3)
	{
	  size_t offs = (hdr->type == S_COMPILE3
			 ? PDB_SYMBOL_COMPILE3_VERSTR_OFFS
			 : PDB_SYMBOL_COMPILE2_VERSTR_OFFS);
	  const gdb_byte *rec_data = data + PDB_RECORD_DATA_OFFS;
	  const gdb_byte *rec_end = data + hdr->rec_size ();
	  if (rec_data + offs >= rec_end)
	    return nullptr;

	  const char *ver = pdb_extract_string (rec_data + offs, rec_end);
	  if (ver == nullptr || *ver == '\0')
	    return nullptr;

	  return obstack_strdup (&pdb->objfile->objfile_obstack, ver);
	}

      data += hdr->rec_size ();
    }

  return nullptr;
}

/* Offset of the range field inside a location-range record, or 0 for a
   record kind this does not describe.  */

static uint32_t
pdb_defrange_range_offs (uint16_t type)
{
  switch (type)
    {
    case S_DEFRANGE_FRAMEPOINTER_REL:
      return PDB_SYMBOL_DEFRANGE_FPREL_OFFSTART_OFFS;
    case S_DEFRANGE_REGISTER:
      return PDB_SYMBOL_DEFRANGE_REG_RANGE_OFFS;
    case S_DEFRANGE_REGISTER_REL:
      return PDB_SYMBOL_DEFRANGE_REGREL_RANGE_OFFS;
    default:
      return 0;
    }
}

/* Derive a procedure's prologue end from where its parameters first become
   readable.  DATA points just past the procedure record, END past the last
   record of the module, SEG is the procedure's section, and OFF is the
   procedure's own section offset.

   Parameters precede any nested scope, and each S_LOCAL is followed by the
   location ranges that apply to it.  A parameter's first range starts where
   the prologue finished storing it, so when every parameter agrees on that
   address, it is the end of the prologue.  A procedure that allocates no
   frame, uses no frame register and takes no parameters has nothing to set
   up, so its prologue ends at OFF.  Returns the section offset, or nothing
   when the parameters disagree, have no location, or the run ends before one
   is seen.  */

static std::optional<uint32_t>
pdb_infer_prologue_end (gdb_byte *data, const gdb_byte *end, uint16_t seg,
			uint32_t off)
{
  uint32_t common = 0;
  bool want_range = false;
  bool empty_frame = false;
  int located = 0;

  while (data + PDB_RECORD_HDR_SIZE <= end)
    {
      auto hdr = pdb_parse_sym_record_hdr (data, end);
      if (!hdr)
	return {};

      size_t rec_size = hdr->rec_size ();
      gdb_byte *rec = data + PDB_RECORD_DATA_OFFS;
      size_t body = rec_size - PDB_RECORD_DATA_OFFS;
      uint32_t range_offs = pdb_defrange_range_offs (hdr->type);

      if (range_offs != 0)
	{
	  /* Only the first range of each parameter marks where it starts
	     being readable; later ones continue an already-counted one.  */
	  if (want_range)
	    {
	      if (body < range_offs + CV_RANGE_CBRANGE_OFFS + 2
		  || read_u16 (rec + range_offs + CV_RANGE_ISECT_OFFS) != seg)
		return {};

	      uint32_t start = read_u32 (rec + range_offs
					 + CV_RANGE_OFF_START_OFFS);
	      if (located != 0 && start != common)
		return {};

	      common = start;
	      located++;
	      want_range = false;
	    }
	}
      else if (hdr->type == S_LOCAL)
	{
	  if (body < PDB_SYMBOL_LOCAL_NAME_OFFS)
	    return {};
	  if ((read_u16 (rec + PDB_SYMBOL_LOCAL_FLAGS_OFFS)
	       & CV_LVARFLAG_IsParam) == 0)
	    break;
	  want_range = true;
	  empty_frame = false;
	}
      else if (hdr->type == S_FRAMEPROC)
	{
	  if (body < PDB_SYMBOL_FRAMEPROC_FLAGS_OFFS + 4)
	    return {};

	  uint32_t frame_size
	    = read_u32 (rec + PDB_SYMBOL_FRAMEPROC_FRAME_SIZE_OFFS);
	  uint32_t fp_flags = read_u32 (rec + PDB_SYMBOL_FRAMEPROC_FLAGS_OFFS);

	  empty_frame
	    = (frame_size == 0
	       && ((fp_flags >> PDB_FRAMEPROC_LOCAL_BP_SHIFT)
		   & PDB_FRAMEPROC_BP_MASK) == 0
	       && ((fp_flags >> PDB_FRAMEPROC_PARAM_BP_SHIFT)
		   & PDB_FRAMEPROC_BP_MASK) == 0);
	}
      else
	break;

      data += rec_size;
    }

  if (located != 0)
    return common;
  if (empty_frame && !want_range)
    return off;

  return {};
}

/* See pdb-internal.h.  */

void
pdb_collect_prologue_ends (pdb_per_objfile *pdb, pdb_module_info *mod_info,
			   gdb_byte *module_stream,
			   std::unordered_map<CORE_ADDR,
					      pdb_prologue_end> *out)
{
  gdb_byte *data = module_stream + PDB_MODULE_SYMBOLS_OFFS;
  const gdb_byte *end = module_stream + mod_info->sym_byte_size;

  while (data + PDB_RECORD_HDR_SIZE <= end)
    {
      auto hdr = pdb_parse_sym_record_hdr (data, end);
      if (!hdr)
	break;

      size_t rec_size = hdr->rec_size ();
      gdb_byte *rec = data + PDB_RECORD_DATA_OFFS;
      size_t body = rec_size - PDB_RECORD_DATA_OFFS;

      switch (hdr->type)
	{
	case S_GPROC32:
	case S_LPROC32:
	case S_GPROC32_ID:
	case S_LPROC32_ID:
	  if (pdb_sym_body_valid (hdr->type, rec, body))
	    {
	      uint32_t dbg_start
		= read_u32 (rec + PDB_SYMBOL_FUNC_DBGSTART_OFFS);
	      uint32_t code_size
		= read_u32 (rec + PDB_SYMBOL_FUNC_CODE_SIZE_OFFS);
	      uint32_t off = read_u32 (rec + PDB_SYMBOL_FUNC_SECTION_OFFS_OFFS);
	      uint16_t seg = read_u16 (rec + PDB_SYMBOL_FUNC_SECTION_NUM_OFFS);
	      bool inferred = false;

	      /* clang-cl leaves DbgStart zero for most procedures, so fall
		 back to the parameters' own ranges.  */
	      if (dbg_start == 0)
		{
		  auto at = pdb_infer_prologue_end (data + rec_size, end, seg,
						    off);
		  if (at.has_value () && *at >= off)
		    {
		      dbg_start = *at - off;
		      inferred = true;
		    }
		}

	      if (inferred ? dbg_start < code_size
			   : (dbg_start != 0 && dbg_start < code_size))
		{
		  CORE_ADDR start = pdb->map_section_offset_to_pc (seg, off);
		  CORE_ADDR after
		    = pdb->map_section_offset_to_pc (seg, off + dbg_start);
		  if (start != 0 && after != 0)
		    out->emplace (start, pdb_prologue_end { after, inferred });
		}
	    }
	  break;

	default:
	  break;
	}

      data += rec_size;
    }
}

/* See pdb-internal.h.  */

void
pdb_parse_symbols (pdb_per_objfile *pdb, pdb_module_info *mod_info,
		   gdb_byte *module_stream, buildsym_compunit *cu,
		   pdb_sym_flags flags, pdb_range_pair_vec *func_ranges,
		   enum language lang, pdb_symbol_lines *sym_lines)
{
  gdb_byte *syms_start = module_stream + PDB_MODULE_SYMBOLS_OFFS;
  const gdb_byte *syms_end = module_stream + mod_info->sym_byte_size;

  pdb_sym_parser parser (pdb, cu, flags, func_ranges, lang, mod_info,
			 sym_lines);

  /* We iterate by grabbing two values at a time: reclen and rectype.  In each
     loop check if there are 2 values available.  */
  gdb_byte *data = syms_start;
  while (data + PDB_RECORD_HDR_SIZE <= syms_end)
    {
      QUIT;
      auto hdr = pdb_parse_sym_record_hdr (data, syms_end);
      if (!hdr)
	{
	  pdb_warning ("bad record header at offset %td, stopping",
		       (ptrdiff_t) (data - syms_start));
	  break;
	}
      uint16_t reclen = hdr->len;
      uint16_t rectype = hdr->type;
      size_t rec_size = hdr->rec_size ();

      gdb_byte *rec_data = data + PDB_RECORD_DATA_OFFS;

      /* In dump mode, print the record header before dispatching.  */
      if (flags & PDB_DUMP_SYM)
	{
	  auto rec_name = pdb_sym_rec_type_name (rectype);
	  gdb_printf ("    %-30s len:%u", rec_name.c_str (), reclen);
	}

      /* Reject records too small to hold their fixed fields, or whose name
	 runs past the record, before any handler reads them.  */
      size_t body_size = rec_size - PDB_RECORD_DATA_OFFS;
      if (!pdb_sym_body_valid (rectype, rec_data, body_size))
	{
	  if (flags & PDB_DUMP_SYM)
	    gdb_printf ("  {truncated}\n");
	  else
	    pdb_complaint ("truncated %s record (body=%zu), skipping",
			   pdb_sym_rec_type_name (rectype).c_str (),
			   body_size);
	  data += rec_size;
	  continue;
	}

      switch (rectype)
	{
	case S_GPROC32:
	case S_LPROC32:
	case S_GPROC32_ID:
	case S_LPROC32_ID:
	  parser.handle_func_sym (rec_data, rec_size - PDB_RECORD_DATA_OFFS,
				  rectype);
	  break;

	case S_PUB32:
	  parser.handle_pub_sym (rec_data, rectype);
	  break;

	case S_GDATA32:
	case S_LDATA32:
	  parser.handle_var_sym (rec_data, rectype);
	  break;

	case S_LOCAL32:
	  parser.handle_local_sym (rec_data, rectype);
	  break;

	case S_DEFRANGE_REGISTER:
	  parser.handle_defrange_reg (rec_data, reclen);
	  break;

	case S_DEFRANGE_REGISTER_REL:
	  parser.handle_defrange_regrel (rec_data, reclen);
	  break;

	case S_DEFRANGE_FRAMEPOINTER_REL:
	  parser.handle_defrange_fprel (rec_data, reclen);
	  break;

	case S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE:
	  parser.handle_defrange_fprel_fullscope (rec_data);
	  break;

	case S_REGREL32:
	  parser.handle_regrel_sym (rec_data, rectype);
	  break;

	case S_REGISTER:
	  parser.handle_register_sym (rec_data, rectype);
	  break;

	case S_CONSTANT:
	  parser.handle_const_sym (rec_data, rectype, reclen);
	  break;

	case S_LABEL32:
	  parser.handle_label_sym (rec_data, rectype);
	  break;

	case S_UDT:
	  parser.handle_udt_sym (rec_data, rectype, /*from_globals=*/false);
	  break;

	case S_BLOCK32:
	  parser.handle_block_sym (rec_data, rectype);
	  break;

	case S_THUNK32:
	  parser.handle_thunk_sym (rec_data, rectype);
	  break;

	case S_INLINESITE:
	case S_INLINESITE2:
	  parser.handle_inlinesite_sym (rec_data, body_size, rectype);
	  break;

	case S_END:
	case S_INLINESITE_END:
	case S_PROC_ID_END:
	  parser.handle_scope_end_sym (rec_data, rectype);
	  break;

	case S_FRAMEPROC:
	  parser.handle_frameproc_sym (rec_data);
	  break;
	case S_UNAMESPACE:
	  parser.handle_unamespace_sym (rec_data);
	  break;

	/* Unsupported symbols with name field - just dump the name.  */
	case S_OBJNAME:
	  handle_sym_dump (pdb, rec_data, PDB_SYMBOL_OBJNAME_NAME_OFFS, flags);
	  break;
	case S_BPREL32:
	  handle_sym_dump (pdb, rec_data, PDB_SYMBOL_BPREL_NAME_OFFS, flags);
	  break;
	case S_LTHREAD32:
	case S_GTHREAD32:
	case S_LMANDATA:
	case S_GMANDATA:
	  handle_sym_dump (pdb, rec_data, PDB_SYMBOL_VAR_NAME_OFFS, flags);
	  break;
	case S_SECTION:
	  handle_sym_dump (pdb, rec_data, PDB_SYMBOL_SECTION_NAME_OFFS, flags);
	  break;
	case S_EXPORT:
	  handle_sym_dump (pdb, rec_data, PDB_SYMBOL_EXPORT_NAME_OFFS, flags);
	  break;
	case S_COFFGROUP:
	  handle_sym_dump (pdb, rec_data, PDB_SYMBOL_COFFGROUP_NAME_OFFS,
			   flags);
	  break;
	case S_FILESTATIC:
	  handle_sym_dump (pdb, rec_data, PDB_SYMBOL_FILESTATIC_NAME_OFFS,
			   flags);
	  break;
	case S_COBOLUDT:
	  handle_sym_dump (pdb, rec_data, PDB_SYMBOL_UDT_NAME_OFFS, flags);
	  break;

	/* Unsupported symbols without the name field */
	case S_COMPILE2:
	case S_COMPILE3:
	case S_ENVBLOCK:
	case S_BUILDINFO:
	case S_CALLSITEINFO:
	case S_TRAMPOLINE:
	case S_FRAMECOOKIE:
	case S_HEAPALLOCSITE:
	case S_CALLEES:
	case S_CALLERS:
	case S_POGODATA:
	case S_TOKENREF:
	case S_GMANPROC:
	case S_LMANPROC:
	case S_MANCONSTANT:
	case S_DISCARDED:
	case S_ANNOTATION:
	case S_DEFRANGE:
	case S_DEFRANGE_SUBFIELD:
	case S_DEFRANGE_SUBFIELD_REGISTER:
	case S_ARMSWITCHTABLE:
	case S_MOD_TYPEREF:
	case S_REF_MINIPDB:
	case S_PDBMAP:
	case S_LPROC32_DPC:
	case S_LPROC32_DPC_ID:
	case S_INLINEES:
	case S_FASTLINK:
	case S_HOTPATCHFUNC:
	case S_FRAMEREG:
	case S_DATAREF:
	  if (flags & PDB_DUMP_SYM)
	    gdb_printf ("  {unsupported}\n");
	  break;

	default:
	  if (flags & PDB_DUMP_SYM)
	    gdb_printf ("  {unsupported}\n");
	  break;
	}

      /* Reset last_local when we leave the S_LOCAL/S_DEFRANGE* sequence.  */
      bool is_defrange = (rectype == S_DEFRANGE_REGISTER
			  || rectype == S_DEFRANGE_REGISTER_REL
			  || rectype == S_DEFRANGE_FRAMEPOINTER_REL
			  || rectype == S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE
			  || rectype == S_DEFRANGE_SUBFIELD_REGISTER);
      if (!is_defrange && rectype != S_LOCAL32)
	parser.clear_last_local ();

      data += rec_size;
    }

  /* Close any remaining open scopes left by S_BLOCK32 records.  */
  while (!parser.scope_stack ().empty ())
    {
      CORE_ADDR end_addr = parser.scope_stack ().back ().end;
      parser.scope_stack ().pop_back ();
      cu->pop_context (end_addr);
    }
}

/* Walk the SymRecordStream and create global-scope symbols
   (S_GDATA32 / S_LDATA32 / S_CONSTANT / S_UDT) into CU.  */

static void
pdb_load_global_syms (pdb_per_objfile *pdb, buildsym_compunit *cu)
{
  if (pdb->sym_record_data.empty ())
    return;

  gdb_byte *syms_data = pdb->sym_record_data.data ();
  size_t syms_size = pdb->sym_record_data.size ();

  const gdb_byte *syms_end = syms_data + syms_size;
  gdb_byte *data = syms_data;

  pdb_sym_parser parser (pdb, cu, 0, nullptr, language_cplus, nullptr);

  while (data + PDB_RECORD_HDR_SIZE <= syms_end)
    {
      gdb_byte *rec_data = data + PDB_RECORD_DATA_OFFS;

      auto hdr = pdb_parse_sym_record_hdr (data, syms_end);
      if (!hdr)
	{
	  pdb_warning ("bad record header at offset %td in global sym stream,"
		       " stopping",
		       (ptrdiff_t) (data - syms_data));
	  break;
	}
      uint16_t reclen = hdr->len;
      uint16_t rectype = hdr->type;
      size_t rec_size = hdr->rec_size ();

      if (rectype == S_GDATA32 || rectype == S_LDATA32)
	{
	  if (!pdb_sym_body_valid (rectype, rec_data,
				   rec_size - PDB_RECORD_DATA_OFFS))
	    pdb_complaint ("truncated %s record in global stream, skipping",
			   pdb_sym_rec_type_name (rectype).c_str ());
	  else
	    {
	      uint32_t type_ti = read_u32 (rec_data
					   + PDB_SYMBOL_VAR_TYPE_OFFS);
	      if (!pdb_tpi_type_is_fwdref (&pdb->tpi, type_ti))
		parser.handle_var_sym (rec_data, rectype);
	    }
	}
      else if (rectype == S_CONSTANT)
	{
	  if (pdb_sym_body_valid (rectype, rec_data,
				  rec_size - PDB_RECORD_DATA_OFFS))
	    parser.handle_const_sym (rec_data, rectype, reclen);
	}
      else if (rectype == S_UDT)
	{
	  if (!pdb_sym_body_valid (rectype, rec_data,
				   rec_size - PDB_RECORD_DATA_OFFS))
	    pdb_complaint ("truncated %s record in global stream, skipping",
			   pdb_sym_rec_type_name (rectype).c_str ());
	  else
	    parser.handle_udt_sym (rec_data, rectype, /*from_globals=*/true);
	}

      data += rec_size;
    }
}

/* See pdb-internal.h.  */

void
pdb_read_sym_record_stream (pdb_per_objfile *pdb)
{
  uint16_t idx = pdb->sym_record_stream;
  if (idx == 0 || idx == 0xFFFF)
    return;

  /* Move into the owned buffer — GSI lookups reference these bytes
     directly for the objfile's lifetime.  */
  pdb->sym_record_data = pdb->read_stream (idx);
}

/* See pdb-internal.h.  */

std::optional<pdb_gsi_hdr>
pdb_parse_gsi_hash_header (gdb_byte *data, uint32_t data_size)
{
  if (data_size < GSI_HASH_HDR_SIZE)
    return std::nullopt;

  pdb_gsi_hdr hdr = {};
  hdr.sig = read_u32 (data + 0);
  hdr.ver = read_u32 (data + 4);
  uint32_t hr_bytes = read_u32 (data + 8);
  uint32_t bucket_bytes = read_u32 (data + 12);

  uint64_t needed = (uint64_t) GSI_HASH_HDR_SIZE + hr_bytes + bucket_bytes;
  if (needed > data_size)
    {
      pdb_warning ("GSI hash data truncated: need %" PRIu64 " bytes,"
		   " have %u", needed, data_size);
      return std::nullopt;
    }

  hdr.hr = gdb::make_array_view (data + GSI_HASH_HDR_SIZE, hr_bytes);
  hdr.bucket = gdb::make_array_view (hdr.hr.data () + hr_bytes, bucket_bytes);
  return hdr;
}

/* Parse a single symbol record from the SymRecordStream, and dump it
   if PDB_DUMP_SYM is set in FLAGS.
   Used by both pdb_parse_sym_record_stream and pdb_dump_gsi_hash_records.  */

void
pdb_dump_parse_record (pdb_per_objfile *pdb, gdb_byte *rec_data,
		       uint16_t rectype, uint16_t reclen, pdb_sym_flags flags)
{
  pdb_sym_parser parser (pdb, nullptr, flags, nullptr, language_cplus,
			 nullptr);

  switch (rectype)
    {
    case S_PUB32:
      parser.handle_pub_sym (rec_data, rectype);
      break;
    case S_PROCREF:
    case S_LPROCREF:
      parser.handle_procref_sym (rec_data, rectype);
      break;
    case S_DATAREF:
      parser.handle_dataref_sym (rec_data, rectype);
      break;
    case S_UDT:
      parser.handle_udt_sym (rec_data, rectype, /*from_globals=*/false);
      break;
    case S_CONSTANT:
      parser.handle_const_sym (rec_data, rectype, reclen);
      break;
    case S_GDATA32:
    case S_LDATA32:
      parser.handle_var_sym (rec_data, rectype);
      break;
    default:
      if (flags & PDB_DUMP_SYM)
	gdb_printf (" UNSUPPORTED! (no data)\n");
      break;
    }
}

void
pdb_build_minsyms (pdb_per_objfile *pdb)
{
  if (pdb->sym_record_data.empty ())
    return;

  if (pdb->psgsi_stream == 0 || pdb->psgsi_stream == 0xFFFF)
    return;

  auto minsym_start = std::chrono::steady_clock::now ();

  /* PSGSI stream — used only here, auto-freed at scope exit.  */
  auto data_buf = pdb->read_stream (pdb->psgsi_stream);
  if (data_buf.size () < PSGSI_HDR_SIZE)
    return;

  gdb_byte *data = data_buf.data ();

  gdb_byte *gsi_data = data + PSGSI_HDR_SIZE;

  auto stream_size = pdb->streams[pdb->psgsi_stream].size;
  uint32_t sym_hash_size = read_u32 (data + PSGSI_HDR_SYM_HASH_OFFS);

  if (auto gsi_avail = stream_size - PSGSI_HDR_SIZE; sym_hash_size > gsi_avail)
    return;

  auto gsi = pdb_parse_gsi_hash_header (gsi_data, sym_hash_size);
  if (!gsi)
    return;

  minimal_symbol_reader reader (pdb->objfile);

  /* Natural names of data objects, keyed by unrelocated address.  A public
     carries only the decorated name and there is no MSVC demangler, so a
     data object that has both is better shown by this one.  */
  std::unordered_map<CORE_ADDR, std::pair<const char *, uint16_t>> data_names;
  {
    gdb_byte *rec = pdb->sym_record_data.data ();
    const gdb_byte *rec_end = rec + pdb->sym_record_data.size ();

    while (rec + PDB_RECORD_HDR_SIZE <= rec_end)
      {
	auto hdr = pdb_parse_sym_record_hdr (rec, rec_end);
	if (!hdr)
	  break;

	size_t rec_size = hdr->rec_size ();
	gdb_byte *body = rec + PDB_RECORD_DATA_OFFS;

	if ((hdr->type == S_GDATA32 || hdr->type == S_LDATA32)
	    && pdb_sym_body_valid (hdr->type, body,
				   rec_size - PDB_RECORD_DATA_OFFS))
	  {
	    uint32_t sect_offs = read_u32 (body
					   + PDB_SYMBOL_VAR_SECTION_OFFS_OFFS);
	    uint16_t sect_num = read_u16 (body
					  + PDB_SYMBOL_VAR_SECTION_NUM_OFFS);
	    const char *dname
	      = pdb_extract_string (body + PDB_SYMBOL_VAR_NAME_OFFS,
				    rec + rec_size);

	    if (pdb->section_valid (sect_num) && dname != nullptr
		&& *dname != '\0')
	      data_names.emplace (pdb->map_section_offset_unrelocated (sect_num,
								       sect_offs),
				  std::make_pair (dname, sect_num));
	  }

	rec += rec_size;
      }
  }

  /* Walk GSI hash records (microsoft-pdb gsi.h: HRFile).
     Each record is GSI_HASH_RECORD_SIZE (8) bytes:
       uint32_t offs — SymRecordStream byte offset, stored as offset+1
		       so that 0 can mean "empty slot"
       uint32_t cref — reference count (unused here)  */
  for (uint32_t i = 0; i < gsi->hr.size () / GSI_HASH_RECORD_SIZE; i++)
    {
      const gdb_byte *rec = gsi->hr.data () + i * GSI_HASH_RECORD_SIZE;
      uint32_t offs = read_u32 (rec + GSI_HASH_RECORD_SYMOFFS_OFFS);

      /* Skip empty slots.  */
      if (offs == 0)
	continue;

      offs -= 1;
      if (offs + PDB_RECORD_HDR_SIZE > pdb->sym_record_data.size ())
	{
	  pdb_warning ("PSGSI hash record %u: offset %u out of range "
		       "(sym_record_size=%zu)",
		       i, offs, pdb->sym_record_data.size ());
	  continue;
	}

      gdb_byte *sym_start = pdb->sym_record_data.data () + offs;
      const gdb_byte *syms_end = pdb->sym_record_data.data ()
				 + pdb->sym_record_data.size ();

      /* Validate the full record length fits within the symbol stream.  */
      auto hdr = pdb_parse_sym_record_hdr (sym_start, syms_end);
      if (!hdr)
	{
	  pdb_warning ("PSGSI hash record %u: bad record header at offset %u",
		       i, offs);
	  continue;
	}
      uint16_t reclen = hdr->len;
      uint16_t rectype = hdr->type;
      size_t rec_size = hdr->rec_size ();

      if (rectype != S_PUB32)
	continue;

      /* S_PUB32 body must contain at least flags + sect_offs + sect_num
	 + one byte for the (possibly empty) NUL-terminated name.  */
      const gdb_byte *rec_data = sym_start + PDB_RECORD_DATA_OFFS;
      const gdb_byte *rec_end = sym_start + rec_size;
      if (rec_data + PDB_SYMBOL_PUB_NAME_OFFS + 1 > rec_end)
	{
	  pdb_warning ("PSGSI hash record %u: S_PUB32 body truncated"
		       " (reclen=%u)",
		       i, reclen);
	  continue;
	}

      auto pub_flags = read_u32 (rec_data + PDB_SYMBOL_PUB_FLAGS_OFFS);
      auto sect_offs = read_u32 (rec_data + PDB_SYMBOL_PUB_SECT_OFFS_OFFS);
      auto sect_num = read_u16 (rec_data + PDB_SYMBOL_PUB_SECT_NUM_OFFS);
      const gdb_byte *name_ptr = rec_data + PDB_SYMBOL_PUB_NAME_OFFS;
      auto name = pdb_extract_string (name_ptr, rec_end);
      if (name == nullptr)
	{
	  pdb_warning ("PSGSI hash record %u: S_PUB32 name not"
		       " NUL-terminated",
		       i);
	  continue;
	}

      if (!pdb->section_valid (sect_num))
	continue;

      CORE_ADDR addr = pdb->map_section_offset_to_pc (sect_num, sect_offs);

      /* Record the mangled name keyed by address.  */
      pdb->pub_mangled_names[pdb->map_section_offset_unrelocated (sect_num,
								  sect_offs)] = name;

      /* cvpsfFunction.  */
      const bool is_function = (pub_flags & CV_PUBSYMFLAGS_FUNCTION) != 0;
      enum minimal_symbol_type type = is_function ? mst_text : mst_data;
      const char *msym_name = name;

      if (!is_function)
	{
	  auto nat = data_names.find (
	    pdb->map_section_offset_unrelocated (sect_num, sect_offs));
	  if (nat != data_names.end ())
	    msym_name = nat->second.first;
	}

      int sect_idx = sect_num - 1;
      reader.record_with_info (msym_name, unrelocated_addr (addr), type,
			       sect_idx);
    }

  /* Publics only cover what the linker exported, so a file-static object has
     none and its address would resolve to whichever public happens to
     precede it.  The data records name the rest.  */
  for (const auto &[unrel, info] : data_names)
    {
      if (pdb->pub_mangled_names.find (unrel) != pdb->pub_mangled_names.end ())
	continue;

      const char *sname = pdb->get_section_name (info.second);
      bool bss = sname != nullptr && strcmp (sname, ".bss") == 0;
      reader.record_with_info (info.first, unrelocated_addr (unrel),
			       bss ? mst_file_bss : mst_file_data,
			       info.second - 1);
    }

  double minsym_ms = std::chrono::duration<double, std::milli> (
		       std::chrono::steady_clock::now () - minsym_start).count ();
  pdb_dbg_printf ("Built %zu minimal symbols in %.2f ms",
		  reader.count (), minsym_ms);
  reader.install ();
}

/* See pdb-internal.h.  */

void
pdb_load_global_syms_cu (pdb_per_objfile *pdb)
{
  if (pdb->sym_record_data.empty ())
    return;

  auto start = std::chrono::steady_clock::now ();

  scoped_restore decrementer = increment_reading_symtab ();

  buildsym_compunit cu (pdb->objfile, "<pdb-globals>", "", language_c, 0);
  cu.record_debugformat ("CodeView");
  pdb_load_global_syms (pdb, &cu);
  cu.end_compunit_symtab (0);

  double ms = std::chrono::duration<double, std::milli> (
		std::chrono::steady_clock::now () - start).count ();
  pdb_dbg_printf ("built <pdb-globals> eagerly in %.2f ms", ms);
}

/* Read the record at OFFSET in the SymRecordStream and create its GDB
   global symbol via PARSER (which owns the destination CU).  Shared by
   the per-name and build-all lazy-globals paths.  */

static void
pdb_create_global_from_record (pdb_per_objfile *pdb, pdb_sym_parser &parser,
			       uint32_t offset, uint16_t rectype)
{
  if (offset + PDB_RECORD_HDR_SIZE > pdb->sym_record_data.size ())
    return;

  gdb_byte *rec = pdb->sym_record_data.data () + offset;
  const gdb_byte *end = pdb->sym_record_data.data ()
			+ pdb->sym_record_data.size ();
  auto hdr = pdb_parse_sym_record_hdr (rec, end);
  if (!hdr || hdr->type != rectype)
    return;

  gdb_byte *rec_data = rec + PDB_RECORD_DATA_OFFS;

  switch (rectype)
    {
    case S_GDATA32:
    case S_LDATA32:
      {
	uint32_t type_ti = read_u32 (rec_data + PDB_SYMBOL_VAR_TYPE_OFFS);
	if (!pdb_tpi_type_is_fwdref (&pdb->tpi, type_ti))
	  parser.handle_var_sym (rec_data, rectype);
      }

      break;

    case S_CONSTANT:
      parser.handle_const_sym (rec_data, rectype, hdr->len);
      break;

    case S_UDT:
      parser.handle_udt_sym (rec_data, rectype, /*from_globals=*/true);
      break;

    default:
      break;
    }
}

/* See pdb-internal.h.  */

void
pdb_build_lazy_globals_index (pdb_per_objfile *pdb)
{
  if (pdb->sym_record_data.empty ())
    return;

  auto start = std::chrono::steady_clock::now ();

  gdb_byte *syms = pdb->sym_record_data.data ();
  const gdb_byte *end = syms + pdb->sym_record_data.size ();
  gdb_byte *data = syms;

  while (data + PDB_RECORD_HDR_SIZE <= end)
    {
      auto hdr = pdb_parse_sym_record_hdr (data, end);
      if (!hdr)
	break;

      uint16_t rectype = hdr->type;
      if (rectype == S_GDATA32 || rectype == S_LDATA32
	  || rectype == S_CONSTANT || rectype == S_UDT)
	{
	  gdb_byte *rec_data = data + PDB_RECORD_DATA_OFFS;
	  size_t body = hdr->rec_size () - PDB_RECORD_DATA_OFFS;
	  const gdb_byte *body_end = rec_data + body;
	  const char *name = nullptr;

	  if (rectype == S_CONSTANT)
	    {
	      uint64_t v;
	      uint32_t consumed = pdb_cv_read_numeric (
		rec_data + PDB_SYMBOL_CONST_VALUE_OFFS,
		body > PDB_SYMBOL_CONST_VALUE_OFFS
		  ? body - PDB_SYMBOL_CONST_VALUE_OFFS : 0,
		&v);
	      if (consumed > 0)
		name = pdb_extract_string (rec_data
					   + PDB_SYMBOL_CONST_VALUE_OFFS
					   + consumed, body_end);
	    }
	  else
	    {
	      uint32_t name_offs = (rectype == S_UDT)
				     ? PDB_SYMBOL_UDT_NAME_OFFS
				     : PDB_SYMBOL_VAR_NAME_OFFS;
	      if (body > name_offs)
		name = pdb_extract_string (rec_data + name_offs, body_end);
	    }

	  if (name != nullptr && *name != '\0')
	    pdb->lazy_globals.push_back (
	      { name, (uint32_t) (data - syms), rectype });
	}

      data += hdr->rec_size ();
    }

  /* Ties break on stream offset so records sharing a name stay in the
     order the eager path reads them; std::sort is not stable, and which
     duplicate a lookup returns must not depend on the sort.  */
  std::sort (pdb->lazy_globals.begin (), pdb->lazy_globals.end (),
	     [] (const pdb_per_objfile::pdb_lazy_global &a,
		 const pdb_per_objfile::pdb_lazy_global &b)
	       {
		 std::string_view an (a.name), bn (b.name);
		 if (an != bn)
		   return an < bn;
		 return a.sym_offset < b.sym_offset;
	       });

  double ms = std::chrono::duration<double, std::milli> (
		std::chrono::steady_clock::now () - start).count ();
  pdb_dbg_printf ("built lazy-globals index: %zu entries in %.2f ms",
		  pdb->lazy_globals.size (), ms);
}

/* See pdb-internal.h.  */

void
pdb_register_global_namespaces (pdb_per_objfile *pdb)
{
  auto qualified = [] (const pdb_per_objfile::pdb_lazy_global &e)
    {
      return pdb_last_component_offset (e.name) != 0;
    };

  if (std::none_of (pdb->lazy_globals.begin (), pdb->lazy_globals.end (),
		    qualified))
    return;

  scoped_restore decrementer = increment_reading_symtab ();

  buildsym_compunit cu (pdb->objfile, "<pdb-globals>", "", language_c, 0);
  cu.record_debugformat ("CodeView");
  pdb_sym_parser parser (pdb, &cu, 0, nullptr, language_cplus, nullptr);

  /* Every indexed record kind starts with its type index.  */
  const gdb_byte *syms = pdb->sym_record_data.data ();
  for (const auto &e : pdb->lazy_globals)
    if (qualified (e))
      parser.ensure_namespaces_for (e.name,
				    read_u32 (syms + e.sym_offset
					      + PDB_RECORD_DATA_OFFS));

  cu.end_compunit_symtab (0);
}

/* See pdb-internal.h.  */

void
pdb_build_global (pdb_per_objfile *pdb, std::string_view name)
{
  if (pdb->lazy_globals.empty ())
    return;

  auto lo = std::lower_bound (
    pdb->lazy_globals.begin (), pdb->lazy_globals.end (), name,
    [] (const pdb_per_objfile::pdb_lazy_global &e, std::string_view n)
      { return std::string_view (e.name) < n; });

  std::vector<const pdb_per_objfile::pdb_lazy_global *> todo;
  for (auto it = lo;
       it != pdb->lazy_globals.end () && std::string_view (it->name) == name;
       ++it)
    if (pdb->built_global_names.count (std::string_view (it->name)) == 0)
      todo.push_back (&*it);

  if (todo.empty ())
    return;

  scoped_restore decrementer = increment_reading_symtab ();

  buildsym_compunit cu (pdb->objfile, "<pdb-globals>", "", language_c, 0);
  cu.record_debugformat ("CodeView");
  pdb_sym_parser parser (pdb, &cu, 0, nullptr, language_cplus, nullptr);
  for (const auto *e : todo)
    {
      pdb->built_global_names.insert (std::string_view (e->name));
      pdb_create_global_from_record (pdb, parser, e->sym_offset, e->rectype);
    }
  cu.end_compunit_symtab (0);
}

/* See pdb-internal.h.  */

void
pdb_build_all_globals (pdb_per_objfile *pdb)
{
  if (pdb->all_globals_built || pdb->lazy_globals.empty ())
    return;
  pdb->all_globals_built = true;

  auto start = std::chrono::steady_clock::now ();

  scoped_restore decrementer = increment_reading_symtab ();

  size_t built = 0;
  buildsym_compunit cu (pdb->objfile, "<pdb-globals>", "", language_c, 0);
  cu.record_debugformat ("CodeView");
  pdb_sym_parser parser (pdb, &cu, 0, nullptr, language_cplus, nullptr);
  for (const auto &e : pdb->lazy_globals)
    {
      if (!pdb->built_global_names.insert (
	     std::string_view (e.name)).second)
	continue;
      pdb_create_global_from_record (pdb, parser, e.sym_offset, e.rectype);
      built++;
    }
  cu.end_compunit_symtab (0);

  double ms = std::chrono::duration<double, std::milli> (
		std::chrono::steady_clock::now () - start).count ();
  pdb_dbg_printf ("built all %zu remaining globals in %.2f ms", built, ms);
}

} /* namespace pdb */

INIT_GDB_FILE (pdb_read_symbols)
{
  pdb::pdb_init_loclist ();
}
