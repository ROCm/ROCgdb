/* PDB symbol reader.

   Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 1994-2026 Advanced Micro Devices, Inc. All rights reserved.

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

/* This file implements reading of CodeView symbols from PDB module streams and
   the global symbol records stream and building of GDB symbols with resolved
   types and locations.

   Entry points:
     pdb_parse_symbols       — walk a module's symbol records, create
				GDB symbols (or dump in diagnostic mode)
     pdb_load_global_syms    — read S_GDATA32/S_LDATA32 from SymRecordStream
     pdb_parse_gsi_hash      — parse GSI hash header (used by cooked index
				and info commands)

   Location handling:
     S_LOCAL (local variable symbol) is followed by S_DEFRANGE* records that
     describe register-relative or FP-relative ranges with gaps.  S_REGREL32
     and S_REGISTER records are self-contained register or register-relative
     offset.  All these are parsed into location batons (pdb_loclist_baton) that
     contain location chains (pdb_loc_entry) and are served at read time via
     symbol_computed_ops (pdb_loclist_read_variable()).

   References:
     - microsoft-pdb cvinfo.h: https://github.com/microsoft/microsoft-pdb  */

#include "symtab.h"
#include "gdbtypes.h"
#include "objfiles.h"
#include "buildsym.h"
#include "complaints.h"
#include "source.h"
#include "block.h"
#include "pdb/pdb.h"
#include "pdb/pdb-cv-regs-amd64.h"
#include "frame.h"
#include "value.h"
#include "gdbarch.h"
#include "extract-store-integer.h"
#include "namespace.h"

#include <string.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

/*  Module symbol records start after the 4-byte CV signature.  */
#define PDB_MODULE_SYMBOLS_OFFS  4

/* Symbol record.  Layout:
     uint16_t reclen;    (length of data following reclen)
     uint16_t rectype;   (symbol type)
     [...  record-type-specific data ...]  */
#define PDB_RECORD_LEN_OFFS  0
#define PDB_RECORD_TYPE_OFFS 2
#define PDB_RECORD_DATA_OFFS 4
#define PDB_RECORD_HDR_SIZE  4

/* There is about 200 different symbol types in microsoft-pdb cvinfo.h.
   Below are the supported record types.  */

/* S_GPROC32 / S_LPROC32 - Global/local functions [cvinfo.h: PROCSYM32]
       uint32_t pParent;
       uint32_t pEnd;
       uint32_t pNext;
       uint32_t len;         procedure code size
       uint32_t DbgStart;
       uint32_t DbgEnd;
       uint32_t typind;      type index
       uint32_t off;         offset in section
       uint16_t seg;         section number
       uint8_t  flags;
       char     name[];  */
#define PDB_SYMBOL_FUNC_TYPE_OFFS         24
#define PDB_SYMBOL_FUNC_CODE_SIZE_OFFS    12
#define PDB_SYMBOL_FUNC_SECTION_OFFS_OFFS 28
#define PDB_SYMBOL_FUNC_SECTION_NUM_OFFS  32
#define PDB_SYMBOL_FUNC_FLAGS_OFFS        34
#define PDB_SYMBOL_FUNC_NAME_OFFS         35


/* S_GDATA32 / S_LDATA32 - Global/local data/variable [cvinfo.h: DATASYM3]
       uint32_t typind;     TPI type index
       uint32_t offset;     offset within section
       uint16_t segment;    section number
       char     name[];  */
#define PDB_SYMBOL_VAR_TYPE_OFFS         0
#define PDB_SYMBOL_VAR_SECTION_OFFS_OFFS 4
#define PDB_SYMBOL_VAR_SECTION_NUM_OFFS  8
#define PDB_SYMBOL_VAR_NAME_OFFS         10

/* S_PUB32 - Public symbol (name + address, no type info) [cvinfo.h: PUBSYM32]
       uint32_t flags;      public flags
       uint32_t offset;     offset within section
       uint16_t segment;    section number
       char     name[];  */
#define PDB_SYMBOL_PUB_FLAGS_OFFS       0
#define PDB_SYMBOL_PUB_SECT_OFFS_OFFS   4
#define PDB_SYMBOL_PUB_SECT_NUM_OFFS    8
#define PDB_SYMBOL_PUB_NAME_OFFS       10

/* S_LOCAL / S_LOCAL32 - Local variable [cvinfo.h: LOCALSYM]
      uint32_t typind;      TPI type index
      uint16_t flags;       CV_LVARFLAGS bitfield (param, optimized out, etc.)
      char     name[];  */
#define PDB_SYMBOL_LOCAL_TYPE_OFFS  0
#define PDB_SYMBOL_LOCAL_FLAGS_OFFS 4
#define PDB_SYMBOL_LOCAL_NAME_OFFS  6

/* CV_LVARFLAGS bitfield (S_LOCAL flags field)  [cvinfo.h: CV_LVARFLAGS]  */
#define CV_LVARFLAG_fIsParam         0x0001  /* Variable is a parameter */
#define CV_LVARFLAG_fIsOptimizedOut  0x0100  /* Variable optimized away */

/* S_UDT - User-defined type (struct/class/union) [cvinfo.h: UDTSYM]
       uint32_t typind;     TPI type index
       uint32_t name[];     name  */
#define PDB_SYMBOL_UDT_TYPE_OFFS  0
#define PDB_SYMBOL_UDT_NAME_OFFS  4

/*  S_BLOCK32 - Block scope marker  [cvinfo.h: BLOCKSYM32]
       uint32_t pParent;   offset to parent scope record
       uint32_t pEnd;      offset to S_END record for this block
       uint32_t len;       length of code in bytes
       uint32_t offs;      offset within section
       uint16_t seg;       section number
       char     name[];  */

/* [cvinfo.h: BLOCKSYM32]
   pParent    — unsigned long pParent (exact)
   pEnd       — unsigned long pEnd (exact)
   len        — unsigned long len (exact)
   offs       — CV_uoff32_t off (renamed off→offs)
   seg        — unsigned short seg (exact)
   name       — unsigned char name[1]  */
#define PDB_SYMBOL_BLOCK_CODE_SIZE_OFFS      8
#define PDB_SYMBOL_BLOCK_SECTION_OFFS_OFFS  12
#define PDB_SYMBOL_BLOCK_SECTION_NUM_OFFS   16
#define PDB_SYMBOL_BLOCK_NAME_OFFS          18

/*  S_INLINESITE - Inlined function site, opens scope [cvinfo.h: INLINESITESYM]
     Marks where a function was inlined, variable size.
       uint32_t pParent;             offset to parent scope
       uint32_t pEnd;                offset to S_INLINESITE_END
       uint32_t inlinee;             type index of the inlined function
       uint8_t  binaryAnnotations[]; binary annotation data  */
#define PDB_SYMBOL_INLINESITE_PARENT_OFFS         0
#define PDB_SYMBOL_INLINESITE_END_OFFS            4
#define PDB_SYMBOL_INLINESITE_INLINEE_IDX_OFFS    8
#define PDB_SYMBOL_INLINESITE_BINANNOTATIONS_OFFS 12


/*  S_REGISTER - Register variable   [cvinfo.h: REGSYM]
       uint32_t typind;    TPI type index
       uint16_t reg;       CV register
       char     name[];  */
#define PDB_SYMBOL_REG_TYPE_OFFS  0
#define PDB_SYMBOL_REG_REG_OFFS   4
#define PDB_SYMBOL_REG_NAME_OFFS  6

/*   S_REGREL32 record layout  [cvinfo.h: REGREL32]
       uint32_t off;       offset from register
       uint32_t typind;    TPI type index
       uint16_t reg;       CV register ID
       char     name[];  */
#define PDB_SYMBOL_REGREL_OFFS_OFFS   0
#define PDB_SYMBOL_REGREL_TYPE_OFFS   4
#define PDB_SYMBOL_REGREL_REG_OFFS    8
#define PDB_SYMBOL_REGREL_NAME_OFFS   10

/*   S_CONSTANT - Constant value.  [cvinfo.h: CONSTSYM]
       uint32_t typind;     TPI type index
       uint16_t value;      numeric leaf containing value
       char     name[];  */
#define PDB_SYMBOL_CONST_TYPE_OFFS  0
#define PDB_SYMBOL_CONST_VALUE_OFFS 4

/*  S_LABEL32 - Code label   [cvinfo.h: LABELSYM32]
       uint32_t offs;       offset within section
       uint16_t seg;        section number
       uint8_t  flags;      CV_PROCFLAGS
       char     name[];  */
#define PDB_SYMBOL_LABEL_OFFS_OFFS  0
#define PDB_SYMBOL_LABEL_SEG_OFFS   4
#define PDB_SYMBOL_LABEL_FLAGS_OFFS 6
#define PDB_SYMBOL_LABEL_NAME_OFFS  7

/*  S_FRAMEPROC - Frame procedure info    [cvinfo.h: FRAMEPROCSYM]
       uint32_t cbFrame;       count of bytes of total frame of procedure
       uint32_t cbPad;         count of bytes of padding in the frame
       uint32_t offPad;        offset (FP relative) to where padding starts
       uint32_t cbSaveRegs;    count of bytes of callee save registers
       uint32_t offExHdlr;     offset of exception handler
       uint16_t sectExHdlr;    section id of exception handler
       uint32_t flags;         bit fields:
	       bits 14-15 = encodedLocalBasePointer (0=RSP, 1=RSP, 2=RBP, 3=R13)
	       bits 16-17 = encodedParamBasePointer  */
#define PDB_SYMBOL_FRAMEPROC_FRAME_SIZE_OFFS  0
#define PDB_SYMBOL_FRAMEPROC_PAD_SIZE_OFFS    4
#define PDB_SYMBOL_FRAMEPROC_PAD_OFFSET_OFFS  8
#define PDB_SYMBOL_FRAMEPROC_SAVE_REGS_OFFS   12
#define PDB_SYMBOL_FRAMEPROC_EX_HANDLER_OFFS  16
#define PDB_SYMBOL_FRAMEPROC_EX_SECT_OFFS     20
#define PDB_SYMBOL_FRAMEPROC_FLAGS_OFFS       22
#define PDB_FRAMEPROC_LOCAL_BP_SHIFT          14
#define PDB_FRAMEPROC_BP_MASK                 0x3

/*  S_THUNK32 - Thunk record for indirect calls [cvinfo.h: THUNKSYM32]
       uint32_t pParent;   pointer to the parent
       uint32_t pEnd;      pointer to this blocks end
       uint32_t pNext;     pointer to next symbol
       uint32_t off;       offset within section
       uint16_t seg;       section number
       uint16_t len;       length of thunk
       uint8_t  ord;       THUNK_ORDINAL specifying type of thunk
       char     name[];    */
#define PDB_SYMBOL_THUNK_PARENT_OFFS       0
#define PDB_SYMBOL_THUNK_SECTION_OFFS_OFFS 12
#define PDB_SYMBOL_THUNK_SECTION_NUM_OFFS  16
#define PDB_SYMBOL_THUNK_CODE_SIZE_OFFS    18
#define PDB_SYMBOL_THUNK_NAME_OFFS         21

/*  S_PROCREF / S_LPROCREF / S_DATAREF  - Symbol reference [cvinfo.h: REFSYM2]
       uint32_t sumName;   checksum of symbol name
       uint32_t ibSym;     symbol offset in module stream
       uint16_t imod;      module index
       char     name[];  */
#define PDB_SYMBOL_REF_SUMNAME_OFFS    0
#define PDB_SYMBOL_REF_SYM_OFFSET_OFFS 4
#define PDB_SYMBOL_REF_MOD_INDEX_OFFS  8
#define PDB_SYMBOL_REF_NAME_OFFS       10

/*  S_DEFRANGE_REGISTER - S_LOCAL reg location [cvinfo.h: DEFRANGESYMREGISTER]
       uint16_t reg;       CV register ID
       uint16_t attr;      (2)  attributes
       Followed by variable-length range and gap record.  */
#define PDB_SYMBOL_DEFRANGE_REG_REG_OFFS    0
#define PDB_SYMBOL_DEFRANGE_REG_ATTR_OFFS   2
#define PDB_SYMBOL_DEFRANGE_REG_RANGE_OFFS  4
#define PDB_SYMBOL_DEFRANGE_REG_GAPS_OFFS  12

/*  S_DEFRANGE_REGISTER_REL - Register-relative location for S_LOCAL
       uint16_t reg;       (0)  CV register ID
       uint16_t flags;     (2)  flags
       int32_t  off;       (4)  offset from register
       Followed by variable-length range and gap records.  */
#define PDB_SYMBOL_DEFRANGE_REGREL_REG_OFFS     0
#define PDB_SYMBOL_DEFRANGE_REGREL_FLAGS_OFFS   2
#define PDB_SYMBOL_DEFRANGE_REGREL_OFFSET_OFFS  4
#define PDB_SYMBOL_DEFRANGE_REGREL_RANGE_OFFS   8
#define PDB_SYMBOL_DEFRANGE_REGREL_GAPS_OFFS    16

/*  S_DEFRANGE_FRAMEPOINTER_REL - FP-relative location for S_LOCAL
       int32_t  off;        offset from frame pointer
       uint32_t offStart;   section-relative start PC
       uint16_t isectStart; section index
       uint16_t cbRange;    byte length of valid range
       uint16_t gaps[];     variable-length gaps array  */
#define PDB_SYMBOL_DEFRANGE_FPREL_OFFSET_OFFS        0
#define PDB_SYMBOL_DEFRANGE_FPREL_OFFSTART_OFFS      4
#define PDB_SYMBOL_DEFRANGE_FPREL_ISECT_OFFS         8
#define PDB_SYMBOL_DEFRANGE_FPREL_CBRANGE_OFFS       10
#define PDB_SYMBOL_DEFRANGE_FPREL_GAPS_OFFS          12

/* S_DEFRANGE* address range  [cvinfo.h: CV_lvar_addr_range]
       uint32_t offStart;     section-relative start offset
       uint16_t isectStart;   section index
       uint16_t cbRange;      byte length of range  */
#define CV_RANGE_OFF_START_OFFS   0
#define CV_RANGE_ISECT_OFFS       4
#define CV_RANGE_CBRANGE_OFFS     6

/*  S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE - FP-relative, entire function
       int32_t  offs;       (0)  offset from frame pointer
       No range or gap data — valid for the whole enclosing function.  */
#define PDB_SYMBOL_DEFRANGE_FPREL_FULLSCOPE_OFFSET_OFFS  0

/*  S_END - End of scope marker
    S_PROC_ID_END (0x114f)    - End of procedure with ID
				PROCSYM32.pEnd points to S_END or S_PROC_ID_END
    S_INLINESITE_END (0x114e) - End of inlined function
				(INLINESITESYM.pEnd points to S_INLINESITE_END)
       No data payload for these 3 — just reclen (=0) and rectype.  */

/* Below are the name field offsets for unsupported symbols for which
   we want at least to able to show the name.  */

/* S_OBJNAME: sig(4) name(...)    [cvinfo.h: OBJNAMESYM]*/
#define PDB_SYMBOL_OBJNAME_NAME_OFFS    4

/* S_BPREL32: off(4) typind(4) name(...)   [cvinfo.h: BPRELSYM32]*/
#define PDB_SYMBOL_BPREL_OFFS_OFFS      0
#define PDB_SYMBOL_BPREL_TYPE_OFFS      4
#define PDB_SYMBOL_BPREL_NAME_OFFS      8

/* S_LTHREAD32 / S_GTHREAD32: same as S_GDATA32  [cvinfo.h: THREADSYM32]  */
/* (use PDB_SYMBOL_VAR_* offsets) */

/* S_UNAMESPACE: name(...) [cvinfo.h: UNAMESPACE]  */
#define PDB_SYMBOL_UNAMESPACE_NAME_OFFS 0

/* S_EXPORT: ordinal(2) flags(2) name(...)  [cvinfo.h: EXPORTSYM]  */
#define PDB_SYMBOL_EXPORT_NAME_OFFS     4

/* S_SECTION: sectnum(2) align(1) pad(1) rva(4) len(4) chars(12) name()  */
#define PDB_SYMBOL_SECTION_NAME_OFFS    16

/* S_COFFGROUP: len(4) chars(4) off(4) seg(2) name(...)  */
#define PDB_SYMBOL_COFFGROUP_NAME_OFFS  14

/* S_FILESTATIC: typind(4) modfileoffs(4) flags(2) name(...)  */
#define PDB_SYMBOL_FILESTATIC_NAME_OFFS 10

/* Virtual frame pointer (resolved via S_FRAMEPROC or FPO data).
   Shared across all CodeView architectures.  */
#define CV_REG_VFRAME 30006

/* Convert a CodeView register ID to a GDB register number.

   CodeView has its own register numbering that differs per architecture. We map
   CV register IDs to DWARF register numbers in per-architecture helpers, then
   call gdbarch_dwarf2_reg_to_regnum to get the final GDB register number.  This
   keeps all CodeView knowledge inside the PDB reader without requiring a new
   architecture hook in gdbarch.

   Adding a new architecture: create a pdb-cv-regs-<arch>.h header with
   the CV defines and a cv_<arch>_reg_to_dwarf function, then add an
   else-if branch here.

   Returns -1 for unrecognized registers.  */
static int
cv_reg_to_gdb_regnum (uint16_t cv_reg, struct gdbarch *gdbarch)
{
  const struct bfd_arch_info *arch = gdbarch_bfd_arch_info (gdbarch);
  int dwarf_regnum;

  if (arch->arch == bfd_arch_i386 && arch->mach == bfd_mach_x86_64)
    {
      /* CV_REG_VFRAME is a virtual frame pointer — mapping comes from
	 S_FRAMEPROC, using RSP as default.  */
      dwarf_regnum = cv_amd64_reg_to_dwarf (
			   cv_reg == CV_REG_VFRAME ? CV_AMD64_RSP : cv_reg);
    }
  else
    pdb_error ("PDB: unsupported architecture %s for CodeView register mapping",
	       arch->printable_name);

  if (dwarf_regnum < 0)
    return -1;
  return gdbarch_dwarf2_reg_to_regnum (gdbarch, dwarf_regnum);
}

/* Return a value of the given type marked entirely unavailable.  */
static struct value *
value_unavailable (struct type *type)
{
  struct value *val = value::allocate (type);
  val->mark_bytes_unavailable (0, type->length ());
  return val;
}

/* symbol_computed_ops: read a variable described by a pdb_loclist_baton.
   Walks the parsed location entries to find the one matching current PC.  */
static struct value *
pdb_loclist_read_variable (struct symbol *symbol, const frame_info_ptr &frame)
{
  auto *baton = (struct pdb_loclist_baton *) SYMBOL_LOCATION_BATON (symbol);

  if (baton == nullptr || baton->entries == nullptr)
    return value::allocate_optimized_out (symbol->type ());

  frame_info_ptr frame_info = frame != nullptr ? frame : get_current_frame ();

  if (frame_relative_level (frame_info) != 0)
    {
      pdb_warning ("Variable '%s' not in the current frame; unable to read",
		   symbol->natural_name ());
      return value_unavailable (symbol->type ());
    }

  CORE_ADDR pc = get_frame_pc (frame_info);
  int gdb_regnum = -1;
  int32_t offset = 0;
  bool is_register = false;
  bool found = false;
  bool gapped = false;

  /* Walk the location entries to find the one matching the current PC.
     We can have both a FULL_SCOPE entry and narrower ranges. In that case a
     narrower range applies. This is also valid for gaps - a gap in a narrower
     range means the variable is unavailable at this PC regardless of
     FULL_SCOPE.  */

  for (auto *e = baton->entries; e != nullptr; e = e->next)
    {
      /* DEFRANGE_FULL_SCOPE doesn't set the (full) range but this flag */
      if (e->is_full_scope)
	{
	  if (!found)
	    {
	      gdb_regnum = e->gdb_regnum;
	      offset = e->offset;
	      is_register = e->is_register;
	      found = true;
	    }
	  continue;
	}

      if (pc < e->start || pc >= e->end)
	continue;

      /* Check gaps.  */
      bool in_gap = false;
      for (int i = 0; i < e->num_gaps; i++)
	{
	  if (pc >= e->gaps[i].start && pc < e->gaps[i].end)
	    {
	      in_gap = true;
	      break;
	    }
	}
      if (in_gap)
	{
	  gapped = true;
	  continue;
	}

      gdb_regnum = e->gdb_regnum;
      offset = e->offset;
      is_register = e->is_register;
      found = true;
    }

  if (!found || gapped)
    return value::allocate_optimized_out (symbol->type ());

  if (gdb_regnum < 0)
    return value_unavailable (symbol->type ());

  if (is_register)
    return value_from_register (symbol->type (), gdb_regnum, frame_info, 0, 0);

  CORE_ADDR regval;
  try
    {
      regval = get_frame_register_unsigned (frame_info, gdb_regnum);
    }
  catch (const gdb_exception_error &ex)
    {
      if (ex.error == NOT_AVAILABLE_ERROR)
	return value_unavailable (symbol->type ());
      throw;
    }

  return value_at_lazy (symbol->type (), regval + offset);
}

static struct value *
pdb_loclist_read_variable_at_entry (struct symbol *symbol,
				    const frame_info_ptr &frame)
{
  return pdb_loclist_read_variable (symbol, frame);
}

static void
pdb_loclist_describe_location (struct symbol *symbol, CORE_ADDR addr,
			       struct ui_file *stream)
{
  struct pdb_loclist_baton *baton
    = (struct pdb_loclist_baton *) SYMBOL_LOCATION_BATON (symbol);

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
pdb_loclist_tracepoint_var_ref (struct symbol *symbol, struct agent_expr *ax,
				struct axs_value *value)
{
  /* Not supported for PDB yet.  */
}

static void
pdb_loclist_generate_c_location (struct symbol *symbol, string_file *stream,
				 struct gdbarch *gdbarch,
				 std::vector<bool> &registers_used,
				 CORE_ADDR pc, const char *result_name)
{
  /* Not supported for PDB yet.  */
}

/* The PDB location operations table.  */
static const struct symbol_computed_ops pdb_loclist_funcs = {
  pdb_loclist_read_variable,
  pdb_loclist_read_variable_at_entry,
  pdb_loclist_describe_location,
  0,  /* location_has_loclist */
  pdb_loclist_tracepoint_var_ref,
  pdb_loclist_generate_c_location
};

/* Index returned by register_symbol_computed_impl for our PDB ops.
   Initialized eagerly from INIT_GDB_FILE via pdb_init_loclist.  */
static int pdb_loclist_index;

/* See pdb.h.  */
void
pdb_init_loclist (void)
{
  pdb_loclist_index = register_symbol_computed_impl (LOC_COMPUTED,
						     &pdb_loclist_funcs);
}

/* See pdb.h.  */
bool
pdb_is_pdb_location (struct symbol *sym)
{
  return (sym->loc_class () == LOC_COMPUTED
	  && sym->computed_ops () == &pdb_loclist_funcs);
}

/* Parse symbol record header from REC.  */
bool
pdb_parse_sym_record_hdr (gdb_byte *rec, gdb_byte *end, uint16_t &reclen,
			  uint16_t &rectype, size_t &rec_size)
{
  reclen = UINT16_CAST (rec + PDB_RECORD_LEN_OFFS);
  rectype = UINT16_CAST (rec + PDB_RECORD_TYPE_OFFS);
  rec_size = 2 + reclen;

  if (reclen < 2 || (rec + rec_size > end))
    {
      return false;
    }
  return true;
}

/* Create a full-scope location baton (S_REGREL32, S_REGISTER).  */
static void
pdb_set_symbol_location (struct symbol *sym, struct objfile *objfile,
			 int gdb_regnum, int32_t offset, bool is_register)
{
  struct pdb_loclist_baton *baton
    = OBSTACK_ZALLOC (&objfile->objfile_obstack, struct pdb_loclist_baton);
  baton->pdb = nullptr;

  /* Create a single full-scope entry.  */
  auto *entry
    = OBSTACK_ZALLOC (&objfile->objfile_obstack, struct pdb_loc_entry);
  entry->gdb_regnum = gdb_regnum;
  entry->offset = offset;
  entry->is_register = is_register;
  entry->is_full_scope = true;

  baton->entries = entry;
  SYMBOL_LOCATION_BATON (sym) = baton;
  sym->set_loc_class_index (pdb_loclist_index);
}

/* Add a fully-resolved location entry to a symbol's loclist baton.
   The baton was already allocated by handle_local_sym.
   Gaps are resolved from relative offsets to absolute CORE_ADDR.  */
static void
pdb_add_loc_entry (struct symbol *sym, struct pdb_per_objfile *pdb,
		   CORE_ADDR start, CORE_ADDR end,
		   int gdb_regnum, int32_t offset, bool is_register,
		   bool is_full_scope,
		   const std::vector<std::pair<uint16_t, uint16_t>> &gaps)
{
  auto *baton = (struct pdb_loclist_baton *) SYMBOL_LOCATION_BATON (sym);
  int num_gaps = (int) gaps.size ();

  /* Allocate entry + inline gap array on objfile obstack.  */
  size_t alloc_size = sizeof (struct pdb_loc_entry)
		      + num_gaps * sizeof (struct pdb_loc_gap);
  auto *entry = (struct pdb_loc_entry *)
    obstack_alloc (&pdb->objfile->objfile_obstack, alloc_size);

  entry->start = start;
  entry->end = end;
  entry->gdb_regnum = gdb_regnum;
  entry->offset = offset;
  entry->is_register = is_register;
  entry->is_full_scope = is_full_scope;
  entry->num_gaps = num_gaps;

  /* Resolve gap offsets to absolute CORE_ADDR.  */
  for (uint32_t i = 0; i < num_gaps; i++)
    {
      entry->gaps[i].start = start + gaps[i].first;
      entry->gaps[i].end = entry->gaps[i].start + gaps[i].second;
    }

  /* Prepend to linked list.  */
  entry->next = baton->entries;
  baton->entries = entry;
}

/* Resolve a CodeView register, attach the location to SYM, and add it
   to the local symbols list.  If the register is not recognized, warns
   and attaches a -1 (unsupported) location instead.
   Using LOC_COMPUTED for register access, in order to simplify the code.  */
static void
pdb_set_register_location (struct symbol *sym, struct objfile *objfile,
			   uint16_t cv_reg, int32_t offset,
			   bool is_register, struct buildsym_compunit *cu)
{
  struct gdbarch *gdbarch = objfile->arch ();
  int gdb_regnum = cv_reg_to_gdb_regnum (cv_reg, gdbarch);
  if (gdb_regnum < 0)
    {
      pdb_warning ("unsupported CV register %u for '%s'",
		   cv_reg, sym->natural_name ());
      gdb_regnum = -1;
    }
  /* Multiple S_DEFRANGE* records may follow S_LOCAL — each one updates
     the location, but the symbol is inserted on the first S_DEFRANGE*  */
  bool first_location = (SYMBOL_LOCATION_BATON (sym) == nullptr);
  pdb_set_symbol_location (sym, objfile, gdb_regnum, offset, is_register);
  if (first_location)
    add_symbol_to_list (sym, cu->get_local_symbols ());
}

/* Each symbol in PDB needs to be parsed and the gdb symbol is to be built for
   it. This is the job of handle_* function below. However, we introduce PDB
   symbol structs that wrap up this work. Helps with debugging and code
   organization.  */

struct pdb_sym
{
  uint16_t rectype;
  const char *name;
  uint32_t type_index;
  struct pdb_per_objfile *pdb;
  gdb_byte *rec_data;
  struct buildsym_compunit *cu;
  /* Allocated GDB symbol (if created) */
  struct symbol *sym;

  pdb_sym (gdb_byte *rec_data_in, uint16_t rectype_in,
	   struct pdb_per_objfile *pdb_in,
	   struct buildsym_compunit *cu_in = nullptr)
    : rectype (rectype_in), name (nullptr), type_index (0), pdb (pdb_in),
      rec_data (rec_data_in), cu (cu_in), sym (nullptr) {}

  /* Set name from a byte offset into rec_data.  */
  void set_name (uint32_t offset)
  {
    name = CSTR (rec_data + offset);
  }

  /* Set type_index from a byte offset into rec_data.  */
  void set_type (uint32_t offset)
  {
    type_index = UINT32_CAST (rec_data + offset);
  }

  /* Allocate a GDB symbol with the given domain.  */
  void allocate_symbol (domain_enum domain)
  {
    sym = new (&pdb->objfile->objfile_obstack) symbol;
    sym->set_language (language_c, &pdb->objfile->objfile_obstack);
    sym->compute_and_set_names (name, true, pdb->objfile->per_bfd);
    sym->set_domain (domain);
  }

  /* Set a register location for the symbol using pdb/cu from the class.  */
  void set_register_location (uint16_t reg, int32_t offset, bool is_register)
  {
    pdb_set_register_location (sym, pdb->objfile, reg, offset, is_register, cu);
  }

  /* Resolve type_index via TPI and return a printable name.  */
  std::string type_name ()
  {
    if (type_index == 0)
      return {};
    try
      {
	struct type *t = pdb_tpi_resolve_type (pdb, type_index);
	if (t == nullptr)
	  return {};
	return type_to_string (t);
      }
    catch (const gdb_exception &)
      {
	return {};
      }
  }

  /* Set symbol type via TPI resolution.  */
  void sym_set_type ()
  {
    sym->set_type (pdb_tpi_resolve_type (pdb, type_index));
  }

  /* Helper to set type on a given symbol via TPI resolution.  */
  void set_gdb_sym_type (struct symbol *sym_arg)
  {
    sym_arg->set_type (pdb_tpi_resolve_type (pdb, type_index));
  }

  /* Base dump() — prints type_index, type name, and symbol name.  */
  virtual void dump ()
  {
    auto tn = type_name ();
    gdb_printf ("  ti=%04X", type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" `%s`\n", name);
  }
};

/* Helper to get section name from section number (1-based index).  */
static const char *
pdb_get_section_name (struct pdb_per_objfile *pdb, uint16_t sect_num)
{
  if (sect_num == 0)
    return "(none)";

  bfd *abfd = pdb->objfile->obfd.get ();
  uint16_t idx = 0;
  for (auto *sect = abfd->sections; sect != nullptr; sect = sect->next)
    {
      if (++idx == sect_num)
	return bfd_section_name (sect);
    }
  return "(unknown)";
}

/* S_GPROC32 / S_LPROC32 / S_GPROC32_ID / S_LPROC32_ID.  */
struct pdb_func_sym : pdb_sym
{
  uint32_t code_sz;
  uint32_t sect_offs;
  uint16_t sect_num;
  uint8_t  flags;

  pdb_func_sym (gdb_byte *rec_data, uint16_t rectype,
		struct pdb_per_objfile *pdb, struct buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    parse ();
  }

  void parse ()
  {
    set_type (PDB_SYMBOL_FUNC_TYPE_OFFS);
    set_name (PDB_SYMBOL_FUNC_NAME_OFFS);
    code_sz = UINT32_CAST (rec_data + PDB_SYMBOL_FUNC_CODE_SIZE_OFFS);
    sect_offs = UINT32_CAST (rec_data + PDB_SYMBOL_FUNC_SECTION_OFFS_OFFS);
    sect_num = UINT16_CAST (rec_data + PDB_SYMBOL_FUNC_SECTION_NUM_OFFS);
    flags = UINT8_CAST (rec_data + PDB_SYMBOL_FUNC_FLAGS_OFFS);
  }

  struct symbol *create_gdb_sym ()
  {
    auto *of = pdb->objfile;
    allocate_symbol (FUNCTION_DOMAIN);
    sym->set_loc_class_index (LOC_BLOCK);
    set_gdb_sym_type (sym);
    sym->set_section_index (SECT_OFF_TEXT (of));
    bool is_global = (rectype == S_GPROC32 || rectype == S_GPROC32_ID);
    add_symbol_to_list (sym, is_global ? cu->get_global_symbols ()
				       : cu->get_file_symbols ());
    return sym;
  }

  void dump () override
  {
    auto tn = type_name ();
    auto sect_name = pdb_get_section_name (pdb, sect_num);
    uint32_t end_addr = sect_offs + code_sz - 1;
    gdb_printf ("  %s:%08X-%08X [sz:%u] ti=%04X", sect_name, sect_offs,
		end_addr, code_sz, type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" `%s`\n", name);
  }
};

/* S_GDATA32 / S_LDATA32.  */
struct pdb_var_sym : pdb_sym
{
  uint32_t sect_offs;
  uint16_t sect_num;

  pdb_var_sym (gdb_byte *rec_data, uint16_t rectype,
	       struct pdb_per_objfile *pdb, struct buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    parse ();
  }

  void parse ()
  {
    set_type (PDB_SYMBOL_VAR_TYPE_OFFS);
    sect_offs = UINT32_CAST (rec_data + PDB_SYMBOL_VAR_SECTION_OFFS_OFFS);
    sect_num = UINT16_CAST (rec_data + PDB_SYMBOL_VAR_SECTION_NUM_OFFS);
    set_name (PDB_SYMBOL_VAR_NAME_OFFS);
  }

  struct symbol *create_gdb_sym ()
  {
    CORE_ADDR addr = pdb_map_section_offset_to_pc (pdb, sect_num, sect_offs);

    allocate_symbol (VAR_DOMAIN);
    sym->set_loc_class_index (LOC_STATIC);
    set_gdb_sym_type (sym);
    sym->set_value_address (addr);

    if (sect_num == 0 || sect_num > pdb->num_sections)
      {
	pdb_complaint ("PDB: bad section %u for '%s'", sect_num, name);
	return nullptr;
      }

    sym->set_section_index (sect_num - 1);
    bool is_global = (rectype == S_GDATA32);
    add_symbol_to_list (sym, is_global ? cu->get_global_symbols ()
				       : cu->get_file_symbols ());
    return sym;
  }

  void dump () override
  {
    auto tn = type_name ();
    auto sect_name = pdb_get_section_name (pdb, sect_num);
    gdb_printf ("  %s:%08X ti=%04X", sect_name, sect_offs, type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" `%s`\n", name);
  }
};

/* S_PUB32 — public symbol (name + address, no type info).  */
struct pdb_pub_sym : pdb_sym
{
  uint32_t pub_flags;
  uint32_t sect_offs;
  uint16_t sect_num;

  pdb_pub_sym (gdb_byte *rec_data, uint16_t rectype,
	       struct pdb_per_objfile *pdb, struct buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    parse ();
  }

  void parse ()
  {
    pub_flags = UINT32_CAST (rec_data + PDB_SYMBOL_PUB_FLAGS_OFFS);
    sect_offs = UINT32_CAST (rec_data + PDB_SYMBOL_PUB_SECT_OFFS_OFFS);
    sect_num = UINT16_CAST (rec_data + PDB_SYMBOL_PUB_SECT_NUM_OFFS);
    set_name (PDB_SYMBOL_PUB_NAME_OFFS);
  }

  struct symbol *create_gdb_sym ()
  {
    auto *of = pdb->objfile;
    CORE_ADDR addr = pdb_map_section_offset_to_pc (pdb, sect_num, sect_offs);

    bool is_function = (pub_flags & 0x02) != 0;  /* cvpsfFunction */
    struct symbol *sym = new (&of->objfile_obstack) symbol;
    sym->set_language (language_c, &of->objfile_obstack);
    sym->compute_and_set_names (name, true, of->per_bfd);
    sym->set_domain (is_function ? FUNCTION_DOMAIN : VAR_DOMAIN);
    sym->set_loc_class_index (LOC_STATIC);
    sym->set_value_address (addr);
    if (is_function)
      sym->set_section_index (SECT_OFF_TEXT (of));
    add_symbol_to_list (sym, cu->get_global_symbols ());
    return sym;
  }

  void dump () override
  {
    auto sect_name = pdb_get_section_name (pdb, sect_num);
    gdb_printf ("  %s:%08X flags=%08X `%s`\n",
		sect_name, sect_offs, pub_flags, name);
  }
};

/* S_LOCAL / S_LOCAL32.  */
struct pdb_local_sym : pdb_sym
{
  uint16_t flags;

  pdb_local_sym (gdb_byte *rec_data, uint16_t rectype,
		 struct pdb_per_objfile *pdb, struct buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    parse ();
  }

  void parse ()
  {
    set_type (PDB_SYMBOL_LOCAL_TYPE_OFFS);
    set_name (PDB_SYMBOL_LOCAL_NAME_OFFS);
    flags = UINT16_CAST (rec_data + PDB_SYMBOL_LOCAL_FLAGS_OFFS);
  }

  struct symbol *create_gdb_sym ()
  {
    allocate_symbol (VAR_DOMAIN);
    set_gdb_sym_type (sym);

    if (flags & CV_LVARFLAG_fIsParam)
      sym->set_is_argument (true);
    /* S_LOCAL carries type + name only; actual location comes from
       subsequent S_DEFRANGE* records.  Default to optimized-out until
       DEFRANGE records attach a real location via pdb_add_loc_entry.  */
    sym->set_loc_class_index (LOC_OPTIMIZED_OUT);
    return sym;
  }

  void dump () override
  {
    auto tn = type_name ();
    gdb_printf ("  ti=%04X", type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" flags=%04X `%s`\n", flags, name);
  }
};

/* S_REGREL32 — register-relative local variable (what LLVM emits).
   Self-contained: register + offset + type + name all in one record.  */
struct pdb_regrel_sym : pdb_sym
{
  int32_t offset;
  uint16_t cv_reg;

  pdb_regrel_sym (gdb_byte *rec_data, uint16_t rectype,
		  struct pdb_per_objfile *pdb, struct buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    parse ();
  }

  void parse ()
  {
    set_type (PDB_SYMBOL_REGREL_TYPE_OFFS);
    set_name (PDB_SYMBOL_REGREL_NAME_OFFS);
    offset = INT32_CAST (rec_data + PDB_SYMBOL_REGREL_OFFS_OFFS);
    cv_reg = UINT16_CAST (rec_data + PDB_SYMBOL_REGREL_REG_OFFS);
  }

  struct symbol *create_gdb_sym ()
  {
    allocate_symbol (VAR_DOMAIN);
    sym_set_type ();
    set_register_location (cv_reg, offset, false);
    return sym;
  }

  void dump () override
  {
    auto tn = type_name ();
    gdb_printf ("  reg=%u+%d ti=%04X", cv_reg, offset, type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" `%s`\n", name);
  }
};

/* S_REGISTER — variable lives directly in a register.  */
struct pdb_register_sym : pdb_sym
{
  uint16_t cv_reg;

  pdb_register_sym (gdb_byte *rec_data, uint16_t rectype,
		    struct pdb_per_objfile *pdb, struct buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    parse ();
  }

  void parse ()
  {
    set_type (PDB_SYMBOL_REG_TYPE_OFFS);
    set_name (PDB_SYMBOL_REG_NAME_OFFS);
    cv_reg = UINT16_CAST (rec_data + PDB_SYMBOL_REG_REG_OFFS);
  }

  struct symbol *create_gdb_sym ()
  {
    allocate_symbol (VAR_DOMAIN);
    sym_set_type ();

    set_register_location (cv_reg, 0, true);
    return sym;
  }

  void dump () override
  {
    auto tn = type_name ();
    gdb_printf ("  reg=%u ti=%04X", cv_reg, type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" `%s`\n", name);
  }
};

/* S_CONSTANT — compile-time constant value.  */
struct pdb_constant_sym : pdb_sym
{
  uint64_t value;
  uint16_t m_reclen;

  pdb_constant_sym (gdb_byte *rec_data, uint16_t rectype,
		    struct pdb_per_objfile *pdb, struct buildsym_compunit *cu,
		    uint16_t reclen)
    : pdb_sym (rec_data, rectype, pdb, cu), value (0), m_reclen (reclen)
  {
    parse ();
  }

  void parse ()
  {
    set_type (PDB_SYMBOL_CONST_TYPE_OFFS);

    uint32_t max_len = m_reclen > PDB_SYMBOL_CONST_VALUE_OFFS
      ? m_reclen - PDB_SYMBOL_CONST_VALUE_OFFS : 0;
    uint32_t consumed
      = pdb_cv_read_numeric (rec_data + PDB_SYMBOL_CONST_VALUE_OFFS,
			     max_len, &value);
    if (consumed > 0)
      set_name (PDB_SYMBOL_CONST_VALUE_OFFS + consumed);
    else
      name = "";
  }

  struct symbol *create_gdb_sym ()
  {
    if (cu == nullptr)
      return nullptr;
    allocate_symbol (VAR_DOMAIN);
    sym_set_type ();
    sym->set_loc_class_index (LOC_CONST);
    sym->set_value_longest ((LONGEST) value);
    add_symbol_to_list (sym, cu->get_global_symbols ());
    return sym;
  }

  void dump () override
  {
    auto tn = type_name ();
    gdb_printf ("  ti=%04X", type_index);
    if (!tn.empty ())
      gdb_printf (" (%s)", tn.c_str ());
    gdb_printf (" val=%llu `%s`\n", (unsigned long long) value, name);
  }
};

/* S_LABEL32 — code label.  */
struct pdb_label_sym : pdb_sym
{
  uint32_t sect_offs;
  uint16_t sect_num;

  pdb_label_sym (gdb_byte *rec_data, uint16_t rectype,
		 struct pdb_per_objfile *pdb, struct buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    parse ();
  }

  void parse ()
  {
    sect_offs = UINT32_CAST (rec_data + PDB_SYMBOL_LABEL_OFFS_OFFS);
    sect_num = UINT16_CAST (rec_data + PDB_SYMBOL_LABEL_SEG_OFFS);
    set_name (PDB_SYMBOL_LABEL_NAME_OFFS);
  }

  struct symbol *create_gdb_sym ()
  {
    allocate_symbol (LABEL_DOMAIN);
    sym->set_loc_class_index (LOC_LABEL);
    CORE_ADDR addr = pdb_map_section_offset_to_pc (pdb, sect_num, sect_offs);
    sym->set_value_address (addr);
    sym->set_section_index (SECT_OFF_TEXT (pdb->objfile));
    add_symbol_to_list (sym, cu->get_local_symbols ());
    return sym;
  }

  void dump () override
  {
    auto sect_name = pdb_get_section_name (pdb, sect_num);
    gdb_printf ("  %s:%08X `%s`\n", sect_name, sect_offs, name);
  }
};

/* S_UDT.  */
struct pdb_udt_sym : pdb_sym
{
  pdb_udt_sym (gdb_byte *rec_data, uint16_t rectype,
	       struct pdb_per_objfile *pdb, struct buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    parse ();
  }

  void parse ()
  {
    set_type (PDB_SYMBOL_UDT_TYPE_OFFS);
    set_name (PDB_SYMBOL_UDT_NAME_OFFS);
  }

  struct symbol *create_gdb_sym ()
  {
    if (cu == nullptr)
      return nullptr;
    allocate_symbol (STRUCT_DOMAIN);
    sym->set_loc_class_index (LOC_TYPEDEF);
    sym_set_type ();
    add_symbol_to_list (sym, cu->get_global_symbols ());
    return sym;
  }

  /* Uses base dump() from pdb_sym */
};

/* S_BLOCK32  */
struct pdb_block_sym : pdb_sym
{
  uint32_t code_sz;
  uint32_t sect_offs;
  uint16_t sect_num;

  pdb_block_sym (gdb_byte *rec_data, uint16_t rectype,
		 struct pdb_per_objfile *pdb, struct buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    parse ();
  }

  void parse ()
  {
    code_sz = UINT32_CAST (rec_data + PDB_SYMBOL_BLOCK_CODE_SIZE_OFFS);
    sect_offs = UINT32_CAST (rec_data + PDB_SYMBOL_BLOCK_SECTION_OFFS_OFFS);
    sect_num = UINT16_CAST (rec_data + PDB_SYMBOL_BLOCK_SECTION_NUM_OFFS);
    set_name (PDB_SYMBOL_BLOCK_NAME_OFFS);
  }

  struct symbol *create_gdb_sym ()
  {
    CORE_ADDR start = pdb_map_section_offset_to_pc (pdb, sect_num, sect_offs);
    cu->push_context (0, start);
    return nullptr;
  }

  void dump () override
  {
    auto sect_name = pdb_get_section_name (pdb, sect_num);
    gdb_printf ("  %s:%08X sz=%u `%s`\n", sect_name, sect_offs, code_sz, name);
  }
};

/* S_THUNK32 (0x1102) — thunk record for indirect calls.  */
struct pdb_thunk_sym : pdb_sym
{
  uint32_t sect_offs;
  uint16_t sect_num;
  uint32_t code_sz;

  pdb_thunk_sym (gdb_byte *rec_data, uint16_t rectype,
		 struct pdb_per_objfile *pdb)
    : pdb_sym (rec_data, rectype, pdb)
  {
    parse ();
  }

  void parse ()
  {
    sect_offs = UINT32_CAST (rec_data + PDB_SYMBOL_THUNK_SECTION_OFFS_OFFS);
    sect_num = UINT16_CAST (rec_data + PDB_SYMBOL_THUNK_SECTION_NUM_OFFS);
    code_sz = UINT32_CAST (rec_data + PDB_SYMBOL_THUNK_CODE_SIZE_OFFS);
    set_name (PDB_SYMBOL_THUNK_NAME_OFFS);
  }

  void dump () override
  {
    auto sect_name = pdb_get_section_name (pdb, sect_num);
    gdb_printf ("  %s:%08X sz=%u `%s`\n", sect_name, sect_offs, code_sz, name);
  }
};

/* S_INLINESITE (0x114d) — inlined function site.  */
struct pdb_inlinesite_sym : pdb_sym
{
  uint32_t inlinee_idx;

  pdb_inlinesite_sym (gdb_byte *rec_data, uint16_t rectype,
		      struct pdb_per_objfile *pdb)
    : pdb_sym (rec_data, rectype, pdb), inlinee_idx (0)
  {
    parse ();
  }

  void parse ()
  {
    inlinee_idx = UINT32_CAST (rec_data + PDB_SYMBOL_INLINESITE_INLINEE_IDX_OFFS);
  }

  void dump () override
  {
    gdb_printf ("  inlinee_idx=%04X\n", inlinee_idx);
  }
};

/* Scope marker (S_END, S_INLINESITE_END, S_PROC_ID_END) — no data.  */
struct pdb_scope_end_sym : pdb_sym
{
  pdb_scope_end_sym (gdb_byte *rec_data, uint16_t rectype,
		     struct pdb_per_objfile *pdb)
    : pdb_sym (rec_data, rectype, pdb)
  {
  }

  void dump () override
  {
    gdb_printf ("  (scope end marker)\n");
  }
};

/* S_DEFRANGE_REGISTER_REL — register-relative location for S_LOCAL.  */
struct pdb_defrange_regrel_sym : pdb_sym
{
  uint16_t cv_reg;
  int32_t base_offset;
  uint32_t off_start;
  uint16_t isect_start;
  uint16_t cb_range;
  uint16_t reclen;
  std::vector<std::pair<uint16_t, uint16_t>> gaps;

  pdb_defrange_regrel_sym (gdb_byte *rec_data, uint16_t rectype,
			   struct pdb_per_objfile *pdb,
			   uint16_t reclen_in,
			   struct buildsym_compunit *cu = nullptr)
    : pdb_sym (rec_data, rectype, pdb, cu), reclen (reclen_in)
  {
    parse ();
  }

  void parse ()
  {
    cv_reg = UINT16_CAST (rec_data + PDB_SYMBOL_DEFRANGE_REGREL_REG_OFFS);
    base_offset = INT32_CAST (rec_data + PDB_SYMBOL_DEFRANGE_REGREL_OFFSET_OFFS);
    off_start = UINT32_CAST (rec_data + PDB_SYMBOL_DEFRANGE_REGREL_RANGE_OFFS
			     + CV_RANGE_OFF_START_OFFS);
    isect_start = UINT16_CAST (rec_data + PDB_SYMBOL_DEFRANGE_REGREL_RANGE_OFFS
			       + CV_RANGE_ISECT_OFFS);
    cb_range = UINT16_CAST (rec_data + PDB_SYMBOL_DEFRANGE_REGREL_RANGE_OFFS
			    + CV_RANGE_CBRANGE_OFFS);

    int data_size = reclen - 2;
    int gaps_bytes = data_size - PDB_SYMBOL_DEFRANGE_REGREL_GAPS_OFFS;
    int num_gaps = (gaps_bytes > 0) ? gaps_bytes / 4 : 0;
    gdb_byte *gaps_ptr = rec_data + PDB_SYMBOL_DEFRANGE_REGREL_GAPS_OFFS;
    for (int i = 0; i < num_gaps; i++)
      {
	auto gap_offset = UINT16_CAST (gaps_ptr);
	auto gap_len = UINT16_CAST (gaps_ptr + 2);
	gaps.push_back ({gap_offset, gap_len});
	gaps_ptr += 4;
      }
  }

  void create_gdb_sym (struct symbol *last_local)
  {
    if (last_local == nullptr)
      return;
    struct gdbarch *gdbarch = pdb->objfile->arch ();
    int gdb_regnum = cv_reg_to_gdb_regnum (cv_reg, gdbarch);
    CORE_ADDR start = pdb_map_section_offset_to_pc (pdb, isect_start,
						    off_start);
    pdb_add_loc_entry (last_local, pdb, start, start + cb_range,
		       gdb_regnum, base_offset, false, false, gaps);
  }

  void dump () override
  {
    auto sect_name = pdb_get_section_name (pdb, isect_start);
    uint32_t end_addr = off_start + cb_range - 1;
    gdb_printf ("  reg=%u+%d range=[%s:%08X-%08X]", cv_reg, base_offset,
		sect_name, off_start, end_addr);
    if (!gaps.empty ())
      {
	gdb_printf (" gaps=[");
	for (size_t i = 0; i < gaps.size (); ++i)
	  {
	    if (i > 0)
	      gdb_printf (", ");
	    uint32_t gap_start = off_start + gaps[i].first;
	    uint32_t gap_end = gap_start + gaps[i].second - 1;
	    gdb_printf ("%08X-%08X", gap_start, gap_end);
	  }
	gdb_printf ("]");
      }
    gdb_printf ("\n");
  }
};

/* S_DEFRANGE_REGISTER — register location for S_LOCAL.  */
struct pdb_defrange_reg_sym : pdb_sym
{
  uint16_t cv_reg;
  uint16_t attr;
  uint32_t off_start;
  uint16_t isect_start;
  uint16_t cb_range;
  uint16_t reclen;
  std::vector<std::pair<uint16_t, uint16_t>> gaps;

  pdb_defrange_reg_sym (gdb_byte *rec_data, uint16_t rectype,
			struct pdb_per_objfile *pdb,
			uint16_t reclen_in,
			struct buildsym_compunit *cu = nullptr)
    : pdb_sym (rec_data, rectype, pdb, cu), reclen (reclen_in)
  {
    parse ();
  }

protected:
  /* Constructor for derived classes — does not call parse().  */
  pdb_defrange_reg_sym (gdb_byte *rec_data, uint16_t rectype,
			struct pdb_per_objfile *pdb,
			uint16_t reclen_in,
			struct buildsym_compunit *cu, bool)
    : pdb_sym (rec_data, rectype, pdb, cu), reclen (reclen_in)
  {
  }

public:

  /* Parse range fields (offStart, isect, cbRange) from a given offset.  */
  void parse_range (uint32_t range_offs)
  {
    off_start = UINT32_CAST (rec_data + range_offs + CV_RANGE_OFF_START_OFFS);
    isect_start = UINT16_CAST (rec_data + range_offs + CV_RANGE_ISECT_OFFS);
    cb_range = UINT16_CAST (rec_data + range_offs + CV_RANGE_CBRANGE_OFFS);
  }

  /* Parse variable-length gap array from a given offset.  */
  void parse_gaps (uint32_t gaps_offs)
  {
    int data_size = reclen - 2;
    int gaps_bytes = data_size - (int) gaps_offs;
    int num_gaps = (gaps_bytes > 0) ? gaps_bytes / 4 : 0;
    gdb_byte *gaps_ptr = rec_data + gaps_offs;
    for (int i = 0; i < num_gaps; i++)
      {
	auto gap_offset = UINT16_CAST (gaps_ptr);
	auto gap_len = UINT16_CAST (gaps_ptr + 2);
	gaps.push_back ({gap_offset, gap_len});
	gaps_ptr += 4;
      }
  }

  void parse ()
  {
    cv_reg = UINT16_CAST (rec_data + PDB_SYMBOL_DEFRANGE_REG_REG_OFFS);
    attr = UINT16_CAST (rec_data + PDB_SYMBOL_DEFRANGE_REG_ATTR_OFFS);
    parse_range (PDB_SYMBOL_DEFRANGE_REG_RANGE_OFFS);
    parse_gaps (PDB_SYMBOL_DEFRANGE_REG_GAPS_OFFS);
  }

  void create_gdb_sym (struct symbol *last_local,
		       int32_t offset = 0, bool is_register = true)
  {
    if (last_local == nullptr)
      return;
    struct gdbarch *gdbarch = pdb->objfile->arch ();
    int gdb_regnum = cv_reg_to_gdb_regnum (cv_reg, gdbarch);
    CORE_ADDR start = pdb_map_section_offset_to_pc (pdb, isect_start,
						    off_start);
    pdb_add_loc_entry (last_local, pdb, start, start + cb_range,
		       gdb_regnum, offset, is_register, false, gaps);
  }

  /* Dump gaps array (shared by subclasses).  */
  void dump_gaps ()
  {
    if (!gaps.empty ())
      {
	gdb_printf (" gaps=[");
	for (size_t i = 0; i < gaps.size (); ++i)
	  {
	    if (i > 0)
	      gdb_printf (", ");
	    uint32_t gap_start = off_start + gaps[i].first;
	    uint32_t gap_end = gap_start + gaps[i].second - 1;
	    gdb_printf ("%08X-%08X", gap_start, gap_end);
	  }
	gdb_printf ("]");
      }
  }

  void dump () override
  {
    auto sect_name = pdb_get_section_name (pdb, isect_start);
    uint32_t end_addr = off_start + cb_range - 1;
    gdb_printf ("  reg=%u range=[%s:%08X-%08X]", cv_reg,
		sect_name, off_start, end_addr);
    dump_gaps ();
    gdb_printf ("\n");
  }
};

/* S_DEFRANGE_FRAMEPOINTER_REL — FP-relative location with explicit range
   and gaps.  Inherits range/gaps parsing from pdb_defrange_reg_sym.  */
struct pdb_defrange_fprel_sym : pdb_defrange_reg_sym
{
  int32_t fp_offset;

  pdb_defrange_fprel_sym (gdb_byte *rec_data, uint16_t rectype,
			  struct pdb_per_objfile *pdb,
			  uint16_t reclen_in,
			  struct buildsym_compunit *cu = nullptr)
    : pdb_defrange_reg_sym (rec_data, rectype, pdb, reclen_in, cu, true)
  {
    parse ();
  }

  void parse ()
  {
    fp_offset = INT32_CAST (rec_data + PDB_SYMBOL_DEFRANGE_FPREL_OFFSET_OFFS);
    parse_range (PDB_SYMBOL_DEFRANGE_FPREL_OFFSTART_OFFS);
    parse_gaps (PDB_SYMBOL_DEFRANGE_FPREL_GAPS_OFFS);
  }

  void create_gdb_sym (struct symbol *last_local, uint16_t frame_reg)
  {
    cv_reg = frame_reg;
    pdb_defrange_reg_sym::create_gdb_sym (last_local, fp_offset, false);
  }

  void dump () override
  {
    auto sect_name = pdb_get_section_name (pdb, isect_start);
    uint32_t end_addr = off_start + cb_range - 1;
    gdb_printf ("  fp+%d range=[%s:%08X-%08X]", fp_offset, sect_name,
		off_start, end_addr);
    dump_gaps ();
    gdb_printf ("\n");
  }
};

/* S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE — FP-relative location for
   S_LOCAL, valid for the entire enclosing function.  */
struct pdb_defrange_fprel_fullscope_sym : pdb_sym
{
  int32_t fp_offset;

  pdb_defrange_fprel_fullscope_sym (gdb_byte *rec_data, uint16_t rectype,
				    struct pdb_per_objfile *pdb,
				    struct buildsym_compunit *cu = nullptr)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    parse ();
  }

  void parse ()
  {
    fp_offset = INT32_CAST (rec_data +
			    PDB_SYMBOL_DEFRANGE_FPREL_FULLSCOPE_OFFSET_OFFS);
  }

  void create_gdb_sym (struct symbol *last_local, uint16_t frame_reg)
  {
    if (last_local == nullptr)
      return;
    struct gdbarch *gdbarch = pdb->objfile->arch ();
    int gdb_regnum = cv_reg_to_gdb_regnum (frame_reg, gdbarch);
    std::vector<std::pair<uint16_t, uint16_t>> no_gaps;
    pdb_add_loc_entry (last_local, pdb, 0, 0,
		       gdb_regnum, fp_offset, false, true, no_gaps);
  }

  void dump () override
  {
    gdb_printf ("  fp+%d\n", fp_offset);
  }
};

/* S_PROCREF / S_LPROCREF — reference to a procedure in a module stream.  */
struct pdb_procref_sym : pdb_sym
{
  uint32_t sym_offset;
  uint16_t mod_index;

  pdb_procref_sym (gdb_byte *rec_data, uint16_t rectype,
		   struct pdb_per_objfile *pdb)
    : pdb_sym (rec_data, rectype, pdb)
  {
    parse ();
  }

  void parse ()
  {
    sym_offset = UINT32_CAST (rec_data + PDB_SYMBOL_REF_SYM_OFFSET_OFFS);
    mod_index = UINT16_CAST (rec_data + PDB_SYMBOL_REF_MOD_INDEX_OFFS);
    set_name (PDB_SYMBOL_REF_NAME_OFFS);
  }

  void dump () override
  {
    gdb_printf ("  mod=%u/%u `%s`\n", mod_index, sym_offset, name);
  }
};

/* S_DATAREF — reference to data symbol in a module stream.
   TODO: Same layout as S_PROCREF - merge.  */
struct pdb_dataref_sym : pdb_procref_sym
{
  pdb_dataref_sym (gdb_byte *rec_data, uint16_t rectype,
		   struct pdb_per_objfile *pdb)
    : pdb_procref_sym (rec_data, rectype, pdb)
  {
  }
};

/* Handle S_GPROC32 / S_LPROC32 / S_GPROC32_ID / S_LPROC32_ID.
   Opens a scope; the matching S_END will close it. */
static void
handle_func_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		 uint16_t rectype, struct buildsym_compunit *cu,
		 pdb_scope_stack &scope_stack,
		 uint32_t flags,
		 pdb_range_pair_vec *func_ranges,
		 int &remaining_params)
{
  auto fsym = pdb_func_sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return fsym.dump ();

  auto *sym = fsym.create_gdb_sym ();

  /* Determine how many S_REGREL32 records are function parameters.
     Look up the function's TPI type record to get parmcount.  */
  remaining_params = pdb_tpi_get_func_param_count (pdb, fsym.type_index);

  CORE_ADDR start = pdb_map_section_offset_to_pc (pdb, fsym.sect_num,
						  fsym.sect_offs);
  CORE_ADDR end = start + fsym.code_sz;
  cu->push_context (0, start);
  scope_stack.push_back ({end, sym});

  if (func_ranges != nullptr)
    func_ranges->push_back ({start, end});
}

/* Handle S_PUB32.  */
static void
handle_pub_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		uint16_t rectype, struct buildsym_compunit *cu,
		uint32_t flags)
{
  auto sym = pdb_pub_sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym ();
}

/* Handle S_GDATA32 / S_LDATA32.  */
static void
handle_var_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		uint16_t rectype, struct buildsym_compunit *cu,
		uint32_t flags)
{
  auto sym = pdb_var_sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym ();
}

/* Handle S_LOCAL / S_LOCAL32.  Returns the symbol so that subsequent
   S_DEFRANGE* records can add location entries to its baton.
   Allocates the baton with entries=nullptr; DEFRANGE handlers fill it in.  */
static struct symbol *
handle_local_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		  uint16_t rectype, struct buildsym_compunit *cu,
		  uint32_t flags, uint16_t frame_reg)
{
  auto sym = pdb_local_sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    {
      sym.dump ();
      return nullptr;
    }

  auto *gdb_sym = sym.create_gdb_sym ();

  /* If the compiler marked this variable as optimized out, keep it as
     LOC_OPTIMIZED_OUT and return nullptr so subsequent DEFRANGE records
     are skipped.  */
  if (sym.flags & CV_LVARFLAG_fIsOptimizedOut)
    {
      add_symbol_to_list (gdb_sym, cu->get_local_symbols ());
      return nullptr;
    }

  /* Allocate the loclist baton with no entries. S_DEFRANGE* records will add
     parsed entries via pdb_add_loc_entry.  */
  struct pdb_loclist_baton *baton
    = OBSTACK_ZALLOC (&pdb->objfile->objfile_obstack, struct pdb_loclist_baton);
  baton->entries = nullptr;
  baton->pdb = pdb;

  SYMBOL_LOCATION_BATON (gdb_sym) = baton;
  gdb_sym->set_loc_class_index (pdb_loclist_index);
  add_symbol_to_list (gdb_sym, cu->get_local_symbols ());
  return gdb_sym;
}

/* Handle S_REGREL32 — register-relative local variable.
   This is the record LLVM actually emits for local variables in PDBs.
   Self-contained: register + offset + type + name in one record.  */
static void
handle_regrel_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		   uint16_t rectype, struct buildsym_compunit *cu,
		   uint32_t flags, uint16_t vframe_reg,
		   int &remaining_params)
{
  auto sym = pdb_regrel_sym (rec_data, rectype, pdb, cu);

  if (sym.cv_reg == CV_REG_VFRAME)
    sym.cv_reg = vframe_reg;

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  /* MSVC emits both S_LOCAL+DEFRANGE and S_REGREL32 for the same variable.
     If S_LOCAL already created a symbol with this name, skip the legacy
     S_REGREL32 — the S_LOCAL+DEFRANGE location is strictly more precise.  */
  if (cu != nullptr && sym.name != nullptr)
    {
      for (symbol *s : cu->get_local_symbols ())
	if (s->linkage_name () != nullptr
	    && strcmp (s->linkage_name (), sym.name) == 0)
	  return;
    }

  auto *gdb_sym = sym.create_gdb_sym ();

  /* The first remaining_params S_REGREL32 records after a function start
     are parameters (including implicit 'this' for member functions).  */
  if (remaining_params > 0)
    {
      gdb_sym->set_is_argument (true);
      remaining_params--;
    }
}

/* Handle S_REGISTER — variable in a register.  */
static void
handle_register_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		     uint16_t rectype, struct buildsym_compunit *cu,
		     uint32_t flags)
{
  auto sym = pdb_register_sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  /* Skip if S_LOCAL already created a symbol with this name.  */
  if (cu != nullptr && sym.name != nullptr)
    {
      for (symbol *s : cu->get_local_symbols ())
	if (s->linkage_name () != nullptr
	    && strcmp (s->linkage_name (), sym.name) == 0)
	  return;
    }

  sym.create_gdb_sym ();
}

/* Handle S_CONSTANT — compile-time constant.  */
static void
handle_constant_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		     uint16_t rectype, uint16_t reclen,
		     struct buildsym_compunit *cu, uint32_t flags)
{
  auto sym = pdb_constant_sym (rec_data, rectype, pdb, cu, reclen);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym ();
}

/* Handle S_LABEL32.  */
static void
handle_label_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		  uint16_t rectype, struct buildsym_compunit *cu,
		  uint32_t flags)
{
  auto sym = pdb_label_sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym ();
}

/* Handle S_UDT.  */
static void
handle_udt_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		uint16_t rectype, struct buildsym_compunit *cu,
		uint32_t flags)
{
  auto sym = pdb_udt_sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym ();
}

/* Handle S_BLOCK32.
   Opens a nested scope; the matching S_END will close it.  */
static void
handle_block_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		  uint16_t rectype, struct buildsym_compunit *cu,
		  pdb_scope_stack &scope_stack,
		  uint32_t flags)
{
  auto sym = pdb_block_sym (rec_data, rectype, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  CORE_ADDR start = pdb_map_section_offset_to_pc (pdb, sym.sect_num,
						  sym.sect_offs);
  CORE_ADDR end = start + sym.code_sz;
  cu->push_context (0, start);
  scope_stack.push_back ({end, nullptr});
}

/* Handle S_THUNK32.  Opens a scope; the matching S_END will close it.  */
static void
handle_thunk_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		  uint16_t rectype, struct buildsym_compunit *cu,
		  pdb_scope_stack &scope_stack,
		  uint32_t flags)
{
  auto sym = pdb_thunk_sym (rec_data, rectype, pdb);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  cu->push_context (0, 0);
  scope_stack.push_back ({0, nullptr});
}

/* Handle S_INLINESITE.  Opens a scope; S_INLINESITE_END will close it.  */
static void
handle_inlinesite_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		       uint16_t rectype, struct buildsym_compunit *cu,
		       pdb_scope_stack &scope_stack,
		       uint32_t flags)
{
  auto sym = pdb_inlinesite_sym (rec_data, rectype, pdb);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  cu->push_context (0, 0);
  scope_stack.push_back ({0, nullptr});
}

/* Handle S_END / S_INLINESITE_END / S_PROC_ID_END.
   Close the innermost open scope.  */
static void
handle_scope_end_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		      uint16_t rectype, struct buildsym_compunit *cu,
		      pdb_scope_stack &scope_stack,
		      uint32_t flags)
{
  auto sym = pdb_scope_end_sym (rec_data, rectype, pdb);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  if (!scope_stack.empty ())
    {
      auto [end_addr, scope_sym] = scope_stack.back ();
      scope_stack.pop_back ();
      struct context_stack cstk = cu->pop_context ();
      cu->finish_block (scope_sym, cstk.old_blocks, NULL,
			cstk.start_addr, end_addr);

      /* Restore enclosing scope's local symbols.  */
      cu->get_local_symbols () = cstk.locals;
    }
}

/* Handle S_DEFRANGE_REGISTER_REL — attach register-relative location
   to the most recent S_LOCAL symbol.  On success, adds the symbol to the
   local symbols list.  */
static void
handle_defrange_regrel (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
			struct symbol *last_local,
			struct buildsym_compunit *cu,
			uint16_t vframe_reg, uint32_t flags,
			uint16_t reclen)
{
  auto sym = pdb_defrange_regrel_sym (rec_data, S_DEFRANGE_REGISTER_REL,
				      pdb, reclen, cu);
  if (sym.cv_reg == CV_REG_VFRAME)
    sym.cv_reg = vframe_reg;

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym (last_local);
}

/* Handle S_DEFRANGE_REGISTER — attach register location
   to the most recent S_LOCAL symbol.  */
static void
handle_defrange_reg (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		     struct symbol *last_local,
		     struct buildsym_compunit *cu,
		     uint16_t vframe_reg, uint32_t flags,
		     uint16_t reclen)
{
  auto sym = pdb_defrange_reg_sym (rec_data, S_DEFRANGE_REGISTER,
				   pdb, reclen, cu);
  if (sym.cv_reg == CV_REG_VFRAME)
    sym.cv_reg = vframe_reg;

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym (last_local);
}

/* Handle S_DEFRANGE_FRAMEPOINTER_REL — FP-relative with range/gaps.  */
static void
handle_defrange_fprel (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		       struct symbol *last_local,
		       struct buildsym_compunit *cu,
		       uint16_t frame_reg, uint32_t flags,
		       uint16_t reclen)
{
  auto sym = pdb_defrange_fprel_sym (rec_data,
				     S_DEFRANGE_FRAMEPOINTER_REL,
				     pdb, reclen, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym (last_local, frame_reg);
}

/* Handle S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE — attach FP-relative
   location (valid for the whole function) to the most recent S_LOCAL.  */
static void
handle_defrange_fprel_fullscope (struct pdb_per_objfile *pdb,
				 gdb_byte *rec_data,
				 struct symbol *last_local,
				 struct buildsym_compunit *cu,
				 uint16_t frame_reg, uint32_t flags)
{
  auto sym = pdb_defrange_fprel_fullscope_sym (rec_data,
		S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE, pdb, cu);

  if (flags & PDB_DUMP_SYM)
    return sym.dump ();

  sym.create_gdb_sym (last_local, frame_reg);
}

/* Handle S_PROCREF / S_LPROCREF.  */
static void
handle_procref_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		    uint16_t rectype, uint32_t flags)
{
  auto rsym = pdb_procref_sym (rec_data, rectype, pdb);

  if (flags & PDB_DUMP_SYM)
    rsym.dump ();
}

/* Handle S_DATAREF.  */
static void
handle_dataref_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		    uint16_t rectype, uint32_t flags)
{
  auto rsym = pdb_dataref_sym (rec_data, rectype, pdb);

  if (flags & PDB_DUMP_SYM)
    rsym.dump ();
}

/* S_FRAMEPROC — extract frame pointer register for current function.  */
static void
handle_frameproc_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		      uint16_t *frame_reg, uint32_t flags)
{
  uint32_t fp_flags = UINT32_CAST (rec_data + PDB_SYMBOL_FRAMEPROC_FLAGS_OFFS);
  uint32_t enc = (fp_flags >> PDB_FRAMEPROC_LOCAL_BP_SHIFT)
		 & PDB_FRAMEPROC_BP_MASK;
  switch (enc)
    {
    case 1: *frame_reg = CV_AMD64_RSP; break;
    case 2: *frame_reg = CV_AMD64_RBP; break;
    case 3: *frame_reg = CV_AMD64_R13; break;
    default: *frame_reg = CV_AMD64_RSP; break;
    }
  if (flags & PDB_DUMP_SYM)
    gdb_printf ("  frame size:%u local_fp:%s\n",
	       UINT32_CAST (rec_data + PDB_SYMBOL_FRAMEPROC_FRAME_SIZE_OFFS),
	       enc == 2 ? "RBP" : enc == 3 ? "R13" : "RSP");
}

/* Handle S_UNAMESPACE — using namespace directive.  */
static void
handle_unamespace_sym (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		       struct buildsym_compunit *cu, uint32_t flags)
{
  auto ns = CSTR(rec_data + PDB_SYMBOL_UNAMESPACE_NAME_OFFS);
  if (flags & PDB_DUMP_SYM)
    gdb_printf ("  `%s`\n", ns);
  else
    add_using_directive (cu->get_global_using_directives (),
			 ns, "", nullptr, nullptr,
			 std::vector<const char *> (),
			 0, &pdb->objfile->objfile_obstack);
}

/* Dump name for symbol types not yet handled by the reader.  */
static void
handle_sym_dump (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		 uint32_t name_offset, uint32_t flags)
{
  if (flags & PDB_DUMP_SYM)
    gdb_printf ("  `%s` {unsupported}\n", CSTR(rec_data + name_offset));
}

std::string
pdb_sym_rec_type_name (uint16_t rectype)
{
  switch (rectype)
    {
    case S_GPROC32:       return "S_GPROC32";
    case S_LPROC32:       return "S_LPROC32";
    case S_GPROC32_ID:    return "S_GPROC32_ID";
    case S_LPROC32_ID:    return "S_LPROC32_ID";
    case S_GDATA32:       return "S_GDATA32";
    case S_LDATA32:       return "S_LDATA32";
    case S_LOCAL:         return "S_LOCAL";
    case S_UDT:           return "S_UDT";
    case S_BLOCK32:       return "S_BLOCK32";
    case S_REGREL32:      return "S_REGREL32";
    case S_PUB32:         return "S_PUB32";
    case S_REGISTER:      return "S_REGISTER";
    case S_CONSTANT:      return "S_CONSTANT";
    case S_LABEL32:       return "S_LABEL32";
    case S_END:           return "S_END";
    case S_INLINESITE:     return "S_INLINESITE";
    case S_INLINESITE_END: return "S_INLINESITE_END";
    case S_PROC_ID_END:    return "S_PROC_ID_END";
    case S_THUNK32:        return "S_THUNK32";
    case S_DEFRANGE_REGISTER: return "S_DEFRANGE_REGISTER";
    case S_DEFRANGE_REGISTER_REL: return "S_DEFRANGE_REGISTER_REL";
    case S_PROCREF:       return "S_PROCREF";
    case S_LPROCREF:      return "S_LPROCREF";
    case S_DATAREF:       return "S_DATAREF";
    case S_ANNOTATIONREF: return "S_ANNOTATIONREF";
    case S_OBJNAME:       return "S_OBJNAME";
    case S_COMPILE2:      return "S_COMPILE2";
    case S_COMPILE3:      return "S_COMPILE3";
    case S_ENVBLOCK:      return "S_ENVBLOCK";
    case S_UNAMESPACE:    return "S_UNAMESPACE";
    case S_BPREL32:       return "S_BPREL32";
    case S_LTHREAD32:     return "S_LTHREAD32";
    case S_GTHREAD32:     return "S_GTHREAD32";
    case S_LMANDATA:      return "S_LMANDATA";
    case S_GMANDATA:      return "S_GMANDATA";
    case S_BUILDINFO:     return "S_BUILDINFO";
    case S_FRAMEPROC:     return "S_FRAMEPROC";
    case S_CALLSITEINFO:  return "S_CALLSITEINFO";
    case S_FILESTATIC:    return "S_FILESTATIC";
    case S_EXPORT:        return "S_EXPORT";
    case S_SECTION:       return "S_SECTION";
    case S_COFFGROUP:     return "S_COFFGROUP";
    case S_TRAMPOLINE:    return "S_TRAMPOLINE";
    case S_FRAMECOOKIE:   return "S_FRAMECOOKIE";
    case S_HEAPALLOCSITE: return "S_HEAPALLOCSITE";
    case S_CALLEES:      return "S_CALLEES";
    case S_CALLERS:      return "S_CALLERS";
    case S_POGODATA:     return "S_POGODATA";
    case S_INLINESITE2:  return "S_INLINESITE2";
    case S_TOKENREF:     return "S_TOKENREF";
    case S_GMANPROC:     return "S_GMANPROC";
    case S_LMANPROC:     return "S_LMANPROC";
    case S_COBOLUDT:     return "S_COBOLUDT";
    case S_MANCONSTANT:  return "S_MANCONSTANT";
    case S_SEPCODE:      return "S_SEPCODE";
    case S_DISCARDED:    return "S_DISCARDED";
    case S_ANNOTATION:   return "S_ANNOTATION";
    case S_DEFRANGE:     return "S_DEFRANGE";
    case S_DEFRANGE_SUBFIELD:  return "S_DEFRANGE_SUBFIELD";
    case S_DEFRANGE_FRAMEPOINTER_REL: return "S_DEFRANGE_FRAMEPOINTER_REL";
    case S_DEFRANGE_SUBFIELD_REGISTER:  return "S_DEFRANGE_SUBFIELD_REGISTER";
    case S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE:
	 return "S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE";
    case S_LOCAL_2005:     return "S_LOCAL_2005";
    case S_DEFRANGE_2005:  return "S_DEFRANGE_2005";
    case S_DEFRANGE2_2005: return "S_DEFRANGE2_2005";
    case S_ARMSWITCHTABLE: return "S_ARMSWITCHTABLE";
    case S_MOD_TYPEREF:    return "S_MOD_TYPEREF";
    case S_REF_MINIPDB:    return "S_REF_MINIPDB";
    case S_PDBMAP:         return "S_PDBMAP";
    case S_LPROC32_DPC:    return "S_LPROC32_DPC";
    case S_LPROC32_DPC_ID: return "S_LPROC32_DPC_ID";
    case S_INLINEES:       return "S_INLINEES";
    case S_FASTLINK:       return "S_FASTLINK";
    case S_HOTPATCHFUNC:   return "S_HOTPATCHFUNC";
    case S_FRAMEREG:       return "S_FRAMEREG";
    case S_ATTR_FRAMEREL:  return "S_ATTR_FRAMEREL";
    case S_ATTR_REGISTER:  return "S_ATTR_REGISTER";
    case S_ATTR_REGREL:    return "S_ATTR_REGREL";
    case S_ATTR_MANYREG:   return "S_ATTR_MANYREG";
    case S_VFTABLE32:      return "S_VFTABLE32";
    case S_WITH32:         return "S_WITH32";
    case S_MANYREG:        return "S_MANYREG";
    case S_MANYREG2:       return "S_MANYREG2";
    case S_LOCALSLOT:      return "S_LOCALSLOT";
    case S_PARAMSLOT:      return "S_PARAMSLOT";
    default:
      {
	std::ostringstream oss;
	oss << "(UNDEFINED:" << std::hex << std::setfill('0')
	    << std::setw(4) << rectype << ")";
	return oss.str();
      }
    }
}

/* See pdb.h.  */
void
pdb_parse_symbols (struct pdb_per_objfile *pdb,
				struct pdb_module_info *module,
				gdb_byte *module_stream,
				struct buildsym_compunit *cu,
				uint32_t flags,
				pdb_range_pair_vec *func_ranges)
{
  gdb_byte *syms_start = module_stream + PDB_MODULE_SYMBOLS_OFFS;
  gdb_byte *syms_end = module_stream + module->sym_byte_size;

  /* Parallel scope stack: S_END has no payload, so we track the end address
     and associated symbol (for functions) from the scope-opening record.  */
  pdb_scope_stack scope_stack;

  /* Track the most recent S_LOCAL symbol so that S_DEFRANGE* records
     can attach location information to it.  */
  struct symbol *last_local = nullptr;

  /* Current function's virtual frame register, resolved from S_FRAMEPROC.
     Default to RSP; updated each time S_FRAMEPROC is seen.  */
  uint16_t current_frame_reg = CV_AMD64_RSP;

  /* Number of remaining S_REGREL32 records to mark as parameters.
     Set when entering a function scope from the function's TPI type.  */
  int remaining_params = 0;

  /* We iterate by grabbing two values at a time: reclen and rectype. In each
     loop check if there is 2 values available.  */
  gdb_byte *data = syms_start;
  while (data + PDB_RECORD_HDR_SIZE <= syms_end)
    {
      size_t rec_size;
      uint16_t reclen, rectype;
      if (!pdb_parse_sym_record_hdr (data, syms_end, reclen, rectype, rec_size))
	{
	  pdb_warning ("bad reclen=%u at offset %td, stopping", reclen,
		       (ptrdiff_t)(data - syms_start));
	  break;
	}

      gdb_byte *rec_data = data + PDB_RECORD_DATA_OFFS;

      /* In dump mode, print the record header before dispatching.  */
      if (flags & PDB_DUMP_SYM)
	{
	  auto rec_name = pdb_sym_rec_type_name (rectype);
	  gdb_printf ("    %-30s len:%u", rec_name.c_str(), reclen);
	}

      switch (rectype)
	{
	case S_GPROC32:
	case S_LPROC32:
	case S_GPROC32_ID:
	case S_LPROC32_ID:
	  handle_func_sym (pdb, rec_data, rectype, cu, scope_stack, flags,
			   func_ranges, remaining_params);
	  break;

	case S_PUB32:
	  handle_pub_sym (pdb, rec_data, rectype, cu, flags);
	  break;

	case S_GDATA32:
	case S_LDATA32:
	  handle_var_sym (pdb, rec_data, rectype, cu, flags);
	  break;

	case S_LOCAL32:
	  last_local = handle_local_sym (pdb, rec_data, rectype, cu, flags,
					 current_frame_reg);
	  break;

	case S_DEFRANGE_REGISTER:
	  handle_defrange_reg (pdb, rec_data, last_local, cu,
			       current_frame_reg, flags, reclen);
	  break;

	case S_DEFRANGE_REGISTER_REL:
	  handle_defrange_regrel (pdb, rec_data, last_local, cu,
				  current_frame_reg, flags, reclen);
	  break;

	case S_DEFRANGE_FRAMEPOINTER_REL:
	  handle_defrange_fprel (pdb, rec_data, last_local, cu,
				 current_frame_reg, flags, reclen);
	  break;

	case S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE:
	  handle_defrange_fprel_fullscope (pdb, rec_data, last_local, cu,
					   current_frame_reg, flags);
	  break;

	case S_REGREL32:
	  handle_regrel_sym (pdb, rec_data, rectype, cu, flags,
			     current_frame_reg, remaining_params);
	  break;

	case S_REGISTER:
	  handle_register_sym (pdb, rec_data, rectype, cu, flags);
	  break;

	case S_CONSTANT:
	  handle_constant_sym (pdb, rec_data, rectype, reclen, cu, flags);
	  break;

	case S_LABEL32:
	  handle_label_sym (pdb, rec_data, rectype, cu, flags);
	  break;

	case S_UDT:
	  handle_udt_sym (pdb, rec_data, rectype, cu, flags);
	  break;

	case S_BLOCK32:
	  handle_block_sym (pdb, rec_data, rectype, cu, scope_stack, flags);
	  break;

	case S_THUNK32:
	  handle_thunk_sym (pdb, rec_data, rectype, cu, scope_stack, flags);
	  break;

	case S_INLINESITE:
	  handle_inlinesite_sym (pdb, rec_data, rectype, cu, scope_stack, flags);
	  break;

	case S_END:
	case S_INLINESITE_END:
	case S_PROC_ID_END:
	  handle_scope_end_sym (pdb, rec_data, rectype, cu, scope_stack, flags);
	  break;

	case S_FRAMEPROC:
	  handle_frameproc_sym (pdb, rec_data, &current_frame_reg, flags);
	  break;
	case S_UNAMESPACE:
	  handle_unamespace_sym (pdb, rec_data, cu, flags);
	  break;

	/* Unsupported symbols with  name field - just dump the name.  */
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
	  handle_sym_dump (pdb, rec_data, PDB_SYMBOL_COFFGROUP_NAME_OFFS, flags);
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
	case S_INLINESITE2:
	case S_TOKENREF:
	case S_GMANPROC:
	case S_LMANPROC:
	case S_MANCONSTANT:
	case S_SEPCODE:
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

      /* Reset last_local when we leave the S_LOCAL → S_DEFRANGE* sequence.  */
      bool is_defrange = (rectype == S_DEFRANGE_REGISTER
			  || rectype == S_DEFRANGE_REGISTER_REL
			  || rectype == S_DEFRANGE_FRAMEPOINTER_REL
			  || rectype == S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE
			  || rectype == S_DEFRANGE_SUBFIELD_REGISTER);
      if (!is_defrange && rectype != S_LOCAL32)
	last_local = nullptr;

      data += rec_size;
    }

  /* Close any remaining open scopes left by S_BLOCK32 records.
     TODO: Will there ever be S_BLOCK32 records that had no S_END ? Assuming
     here there might be bugs, bad parsing, that this code will cover.  */
  while (!scope_stack.empty ())
    {
      auto [end_addr, sym] = scope_stack.back ();
      scope_stack.pop_back ();
      struct context_stack cstk = cu->pop_context ();
      cu->finish_block (sym, NULL, NULL, cstk.start_addr, end_addr);
    }
}

/* See pdb.h.  */
void
pdb_load_global_syms (struct pdb_per_objfile *pdb, struct buildsym_compunit *cu)
{
  gdb_byte *syms_data = pdb->sym_record_data;
  uint32_t syms_size = pdb->sym_record_size;

  if (syms_data == nullptr || syms_size == 0)
    return;

  gdb_byte *syms_end = syms_data + syms_size;
  gdb_byte *data = syms_data;

  /* Collect records, keeping only the last occurrence of each name
     to avoid duplicate symbols in the GDB symbol table.  The user's
     own symbols are typically emitted after CRT/library symbols.  */
  struct sym_entry { gdb_byte *rec_data; uint16_t rectype; uint16_t reclen; };
  std::unordered_map<std::string, sym_entry> syms;

  /* Iterate by reading the headers - check if that many bytes are available */
  while (data + PDB_RECORD_HDR_SIZE <= syms_end)
    {
      gdb_byte *rec_data = data + PDB_RECORD_DATA_OFFS;

      uint16_t reclen, rectype;
      size_t rec_size;
      if (!pdb_parse_sym_record_hdr (data, syms_end, reclen, rectype, rec_size))
	{
	    pdb_warning ("bad reclen=%u at offset %td in global sym stream",
			 reclen, (ptrdiff_t)(data - syms_data));
	    data += rec_size;
	    continue;
	}

      if (rectype == S_GDATA32 || rectype == S_LDATA32)
	{
	  auto name = CSTR(rec_data + PDB_SYMBOL_VAR_NAME_OFFS);
	  syms[name] = {rec_data, rectype, reclen};
	}
      else if (rectype == S_CONSTANT)
	{
	  auto cs = pdb_constant_sym (rec_data, rectype, pdb, nullptr, reclen);
	  if (cs.name != nullptr && cs.name[0] != '\0')
	    syms[cs.name] = {rec_data, rectype, reclen};
	}

      data += rec_size;
    }

  for (auto &[name, entry] : syms)
    {
      if (entry.rectype == S_CONSTANT)
	handle_constant_sym (pdb, entry.rec_data, entry.rectype,
			     entry.reclen, cu, 0);
      else
	handle_var_sym (pdb, entry.rec_data, entry.rectype, cu, 0);
    }
}

/* See pdb.h.  */
void
pdb_read_sym_record_stream (struct pdb_per_objfile *pdb)
{
  uint16_t idx = pdb->sym_record_stream;
  if (idx == 0 || idx == 0xFFFF)
    return;

  /* Cache — GSI lookups reference sym_record_data directly.  */
  auto buf = pdb_read_stream (pdb, idx);
  pdb->sym_record_data = buf ? (pdb->stream_data[idx] = buf.release ())
			     : nullptr;
  pdb->sym_record_size = pdb->sym_record_data ? pdb->stream_sizes[idx] : 0;
}

/* See pdb.h.  */
struct pdb_gsi_hdr
pdb_parse_gsi_hash (gdb_byte *data, uint32_t data_size)
{
  struct pdb_gsi_hdr hdr = {};
  hdr.valid = false;

  if (data_size < GSI_HASH_HDR_SIZE)
    return hdr;

  memcpy (&hdr, data, GSI_HASH_HDR_SIZE);

  if (GSI_HASH_HDR_SIZE + hdr.hr_bytes + hdr.bucket_bytes > data_size)
    {
      pdb_warning ("GSI hash data truncated: need %u bytes, have %u",
		   GSI_HASH_HDR_SIZE + hdr.hr_bytes + hdr.bucket_bytes,
		   data_size);
      return hdr;
    }

  hdr.hr_data = data + GSI_HASH_HDR_SIZE;
  hdr.bucket_data = hdr.hr_data + hdr.hr_bytes;
  hdr.valid = true;
  return hdr;
}

/* Parse a single symbol record from the SymRecordStream, and dump it
   if PDB_DUMP_SYM is set in FLAGS.
   Used by both pdb_parse_sym_record_stream and pdb_dump_gsi_hash_records.  */
void
pdb_dump_parse_record (struct pdb_per_objfile *pdb, gdb_byte *rec_data,
		       uint16_t rectype, uint16_t reclen, uint32_t flags)
{
  switch (rectype)
    {
    case S_PUB32:
      handle_pub_sym (pdb, rec_data, rectype, nullptr, flags);
      break;
    case S_PROCREF:
    case S_LPROCREF:
      handle_procref_sym (pdb, rec_data, rectype, flags);
      break;
    case S_DATAREF:
      handle_dataref_sym (pdb, rec_data, rectype, flags);
      break;
    case S_UDT:
      handle_udt_sym (pdb, rec_data, rectype, nullptr, flags);
      break;
    case S_CONSTANT:
      handle_constant_sym (pdb, rec_data, rectype, reclen, nullptr, flags);
      break;
    case S_GDATA32:
    case S_LDATA32:
      handle_var_sym (pdb, rec_data, rectype, nullptr, flags);
      break;
    default:
      if (flags & PDB_DUMP_SYM)
	gdb_printf (" UNSUPPORTED! (no data)\n");
      break;
    }
}

void
pdb_build_minsyms (struct pdb_per_objfile *pdb)
{
  /* Nothing to do if no symbol records data.  */
  if (pdb->sym_record_data == nullptr || pdb->sym_record_size == 0)
    return;

  if (pdb->psgsi_stream == 0 || pdb->psgsi_stream == 0xFFFF)
    return;

  /* PSGSI stream — used only here, auto-freed at scope exit.  */
  auto data_buf = pdb_read_stream (pdb, pdb->psgsi_stream);
  if (data_buf == nullptr)
    return;

  gdb_byte *data = data_buf.get ();

  gdb_byte *gsi_data = data + PSGSI_HDR_SIZE;

  auto stream_size = pdb->stream_sizes[pdb->psgsi_stream];
  uint32_t sym_hash_size = UINT32_CAST (data + PSGSI_HDR_SYM_HASH_OFFS);
  auto gsi_avail = stream_size - PSGSI_HDR_SIZE;

  if (sym_hash_size > gsi_avail)
    return;

  auto gsi = pdb_parse_gsi_hash (gsi_data, sym_hash_size);
  if (!gsi.valid)
    return;

  minimal_symbol_reader reader (pdb->objfile);
  uint32_t count = 0;

  /* Walk GSI hash records (microsoft-pdb gsi.h: HRFile).
     Each record is GSI_HASH_RECORD_SIZE (8) bytes:
       uint32_t offs — SymRecordStream byte offset, stored as offset+1
		       so that 0 can mean "empty slot"
       uint32_t cref — reference count (unused here)  */
  for (uint32_t i = 0; i < gsi.hr_bytes / GSI_HASH_RECORD_SIZE; i++)
    {
      gdb_byte *rec = gsi.hr_data + i * GSI_HASH_RECORD_SIZE;
      uint32_t offs = UINT32_CAST (rec + GSI_HASH_RECORD_SYMOFFS_OFFS);

      /* Skip empty slots.  */
      if (offs == 0)
	continue;

      offs -= 1; /* Adjust for +1 encoding.  */
      if (offs + PDB_RECORD_HDR_SIZE > pdb->sym_record_size)
	{
	  pdb_warning ("PSGSI hash record %u: offset %u out of range "
			"(sym_record_size=%u)", i, offs,
			 pdb->sym_record_size);
	  continue;
	}

      gdb_byte *sym_start = pdb->sym_record_data + offs;
      auto rectype = UINT16_CAST (sym_start + PDB_RECORD_TYPE_OFFS);

      if (rectype != S_PUB32)
	continue;

      gdb_byte *rec_data = sym_start + PDB_RECORD_DATA_OFFS;
      auto pub_flags = UINT32_CAST (rec_data + PDB_SYMBOL_PUB_FLAGS_OFFS);
      auto sect_offs = UINT32_CAST (rec_data + PDB_SYMBOL_PUB_SECT_OFFS_OFFS);
      auto sect_num = UINT16_CAST (rec_data + PDB_SYMBOL_PUB_SECT_NUM_OFFS);
      const char *name = CSTR(rec_data + PDB_SYMBOL_PUB_NAME_OFFS);

      if (sect_num == 0)
	continue;

      CORE_ADDR addr = pdb_map_section_offset_to_pc (pdb, sect_num, sect_offs);

      /* cvpsfFunction = 0x02.  */
      bool is_function = (pub_flags & 0x02) != 0;
      enum minimal_symbol_type type = is_function ? mst_text : mst_data;

      int sect_idx = sect_num - 1;
      reader.record_with_info (name, unrelocated_addr (addr), type, sect_idx);
      count++;
    }

  reader.install ();
  pdb_dbg_printf ("Built %u minimal symbols from PSI stream", count);
}
