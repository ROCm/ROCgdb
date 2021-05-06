/* Target-dependent code for the ROCm amdgcn architecture.

   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   Copyright (C) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include "defs.h"

#include "amdgcn-rocm-tdep.h"
#include "arch-utils.h"
#include "dwarf2/frame.h"
#include "frame-base.h"
#include "frame-unwind.h"
#include "gdbarch.h"
#include "gdbsupport/gdb_unique_ptr.h"
#include "inferior.h"
#include "osabi.h"
#include "reggroups.h"
#include "rocm-tdep.h"

bool
rocm_is_amdgcn_gdbarch (struct gdbarch *arch)
{
  return (gdbarch_bfd_arch_info (arch)->arch == bfd_arch_amdgcn);
}

/* Return the name of register REGNUM.  */
static const char *
amdgcn_register_name (struct gdbarch *gdbarch, int regnum)
{
  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (inferior_ptid);
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);

  amd_dbgapi_register_exists_t register_exists;
  if (amd_dbgapi_wave_register_exists (wave_id, tdep->register_ids[regnum],
				       &register_exists)
	!= AMD_DBGAPI_STATUS_SUCCESS
      || register_exists != AMD_DBGAPI_REGISTER_PRESENT)
    return "";

  return tdep->register_names[regnum].c_str ();
}

static int
amdgcn_dwarf_reg_to_regnum (struct gdbarch *gdbarch, int reg)
{
  amd_dbgapi_architecture_id_t architecture_id;
  amd_dbgapi_register_id_t register_id;

  if (amd_dbgapi_get_architecture (gdbarch_bfd_arch_info (gdbarch)->mach,
				   &architecture_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return -1;

  if (amd_dbgapi_dwarf_register_to_register (architecture_id, reg,
					     &register_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return -1;

  return gdbarch_tdep (gdbarch)->regnum_map[register_id];
}

static enum return_value_convention
amdgcn_return_value (struct gdbarch *gdbarch, struct value *function,
		     struct type *type, struct regcache *regcache,
		     gdb_byte *readbuf, const gdb_byte *writebuf)
{
  return RETURN_VALUE_STRUCT_CONVENTION;
}

static struct type *
gdb_type_from_type_name (struct gdbarch *gdbarch, const std::string &type_name)
{
  size_t pos;

  /* vector types.  */
  if ((pos = type_name.find_last_of ('[')) != std::string::npos)
    {
      struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);

      auto it = tdep->vector_type_map.find (type_name);
      if (it != tdep->vector_type_map.end ())
	return it->second;

      struct type *vector_type
	= init_vector_type (gdb_type_from_type_name (gdbarch,
						     type_name.substr (0,
								       pos)),
			    std::stoi (type_name.substr (pos + 1)));

      vector_type->set_name (
	tdep->vector_type_map.emplace (type_name, vector_type)
	  .first->first.c_str ());

      return vector_type;
    }
  /* scalar types.  */
  else if (type_name == "int32_t")
    return builtin_type (gdbarch)->builtin_int32;
  else if (type_name == "uint32_t")
    return builtin_type (gdbarch)->builtin_uint32;
  else if (type_name == "int64_t")
    return builtin_type (gdbarch)->builtin_int64;
  else if (type_name == "uint64_t")
    return builtin_type (gdbarch)->builtin_uint64;
  else if (type_name == "float")
    return builtin_type (gdbarch)->builtin_float;
  else if (type_name == "double")
    return builtin_type (gdbarch)->builtin_double;
  else if (type_name == "void (*)()")
    return builtin_type (gdbarch)->builtin_func_ptr;

  return builtin_type (gdbarch)->builtin_void;
}

static struct type *
amdgcn_register_type (struct gdbarch *gdbarch, int regnum)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  char *bytes;

  if (amd_dbgapi_register_get_info (tdep->register_ids[regnum],
				    AMD_DBGAPI_REGISTER_INFO_TYPE,
				    sizeof (bytes), &bytes)
      == AMD_DBGAPI_STATUS_SUCCESS)
    {
      std::string type_name (bytes);
      xfree (bytes);

      return gdb_type_from_type_name (gdbarch, type_name);
    }

  return builtin_type (gdbarch)->builtin_void;
}

static int
amdgcn_register_reggroup_p (struct gdbarch *gdbarch, int regnum,
			    struct reggroup *group)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  const char *name = reggroup_name (group);

  auto it = tdep->register_class_map.find (name);
  if (it == tdep->register_class_map.end ())
    return group == all_reggroup;

  amd_dbgapi_register_class_state_t state;

  if (amd_dbgapi_register_is_in_register_class (it->second,
						tdep->register_ids[regnum],
						&state)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return group == all_reggroup;

  return state == AMD_DBGAPI_REGISTER_CLASS_STATE_MEMBER
	 || group == all_reggroup;
}

static int
amdgcn_breakpoint_kind_from_pc (struct gdbarch *gdbarch, CORE_ADDR *)
{
  return gdbarch_tdep (gdbarch)->breakpoint_instruction_size;
}

static const gdb_byte *
amdgcn_sw_breakpoint_from_kind (struct gdbarch *gdbarch, int kind, int *size)
{
  *size = kind;
  return gdbarch_tdep (gdbarch)->breakpoint_instruction_bytes.get ();
}

struct amdgcn_frame_cache
{
  CORE_ADDR base;
  CORE_ADDR pc;
};

static struct amdgcn_frame_cache *
amdgcn_frame_cache (struct frame_info *this_frame, void **this_cache)
{
  if (*this_cache)
    return (struct amdgcn_frame_cache *) *this_cache;

  struct amdgcn_frame_cache *cache
    = FRAME_OBSTACK_ZALLOC (struct amdgcn_frame_cache);
  (*this_cache) = cache;

  cache->pc = get_frame_func (this_frame);
  cache->base = 0;

  return cache;
}

static void
amdgcn_frame_this_id (struct frame_info *this_frame, void **this_cache,
		      struct frame_id *this_id)
{
  struct amdgcn_frame_cache *cache
    = amdgcn_frame_cache (this_frame, this_cache);

  if (get_frame_type (this_frame) == INLINE_FRAME)
    (*this_id) = frame_id_build (cache->base, cache->pc);
  else
    (*this_id) = outer_frame_id;

  if (frame_debug)
    {
      fprintf_unfiltered (gdb_stdlog,
			  "{ amdgcn_frame_this_id (this_frame=%d) type=%d -> ",
			  frame_relative_level (this_frame),
			  get_frame_type (this_frame));
      fprint_frame_id (gdb_stdlog, *this_id);
      fprintf_unfiltered (gdb_stdlog, "}\n");
    }

  return;
}

static struct frame_id
amdgcn_dummy_id (struct gdbarch *gdbarch, struct frame_info *this_frame)
{
  return frame_id_build (0, get_frame_pc (this_frame));
}

static struct value *
amdgcn_frame_prev_register (struct frame_info *this_frame, void **this_cache,
			    int regnum)
{
  return frame_unwind_got_register (this_frame, regnum, regnum);
}

static const struct frame_unwind amdgcn_frame_unwind = {
  NORMAL_FRAME,
  default_frame_unwind_stop_reason,
  amdgcn_frame_this_id,
  amdgcn_frame_prev_register,
  NULL,
  default_frame_sniffer,
  NULL,
  NULL,
};

static int
print_insn_amdgcn (bfd_vma memaddr, struct disassemble_info *di)
{
  gdb_disassembler *self
    = static_cast<gdb_disassembler *> (di->application_data);

  /* Try to read at most instruction_size bytes.  */

  amd_dbgapi_size_t instruction_size = gdbarch_max_insn_length (self->arch ());
  gdb::unique_xmalloc_ptr<gdb_byte> buffer (
    (gdb_byte *) xmalloc (instruction_size));

  /* read_memory_func doesn't support partial reads, so if the read
     fails, try one byte less, on and on until we manage to read
     something.  A case where this would happen is if we're trying to
     read the last instruction at the end of a file section and that
     instruction is smaller than the largest instruction.  */
  while (instruction_size > 0)
    {
      if (di->read_memory_func (memaddr, buffer.get (), instruction_size, di)
	  == 0)
	break;
      --instruction_size;
    }
  if (instruction_size == 0)
    {
      di->memory_error_func (-1, memaddr, di);
      return -1;
    }

  amd_dbgapi_architecture_id_t architecture_id;
  if (amd_dbgapi_get_architecture (gdbarch_bfd_arch_info (self->arch ())->mach,
				   &architecture_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return -1;

  char *instruction_text = nullptr;

  auto symbolizer
    = [] (amd_dbgapi_symbolizer_id_t id, amd_dbgapi_global_address_t address,
	  char **symbol_text) -> amd_dbgapi_status_t
  {
    string_file string;
    print_address (reinterpret_cast<struct gdbarch *> (id), address, &string);
    *symbol_text = xstrdup (string.c_str ());
    return AMD_DBGAPI_STATUS_SUCCESS;
  };

  if (amd_dbgapi_disassemble_instruction (architecture_id,
					  static_cast<
					    amd_dbgapi_global_address_t> (
					    memaddr),
					  &instruction_size, buffer.get (),
					  &instruction_text,
					  reinterpret_cast<
					    amd_dbgapi_symbolizer_id_t> (
					    self->arch ()),
					  symbolizer)
      != AMD_DBGAPI_STATUS_SUCCESS)
    {
      size_t alignment;
      if (amd_dbgapi_architecture_get_info (
	    architecture_id,
	    AMD_DBGAPI_ARCHITECTURE_INFO_MINIMUM_INSTRUCTION_ALIGNMENT,
	    sizeof (alignment), &alignment)
	  != AMD_DBGAPI_STATUS_SUCCESS)
	error (_ ("amd_dbgapi_architecture_get_info failed"));

      di->fprintf_func (di->stream, "<illegal instruction>");
      /* Skip to the next valid instruction address.  */
      return align_up (memaddr + 1, alignment) - memaddr;
    }

  /* Print the instruction.  */
  di->fprintf_func (di->stream, "%s", instruction_text);

  /* Free the memory allocated by the amd-dbgapi.  */
  xfree (instruction_text);

  return static_cast<int> (instruction_size);
}

static CORE_ADDR
amdgcn_skip_prologue (struct gdbarch *gdbarch, CORE_ADDR start_pc)
{
  CORE_ADDR func_addr;

  /* See if we can determine the end of the prologue via the symbol table.
     If so, then return either PC, or the PC after the prologue, whichever
     is greater.  */
  if (find_pc_partial_function (start_pc, NULL, &func_addr, NULL))
    {
      CORE_ADDR post_prologue_pc
	= skip_prologue_using_sal (gdbarch, func_addr);
      struct compunit_symtab *cust = find_pc_compunit_symtab (func_addr);

      /* Clang always emits a line note before the prologue and another
	 one after.  We trust clang to emit usable line notes.  */
      if (post_prologue_pc
	  && (cust != NULL && COMPUNIT_PRODUCER (cust) != NULL
	      && startswith (COMPUNIT_PRODUCER (cust), "clang ")))
	return std::max (start_pc, post_prologue_pc);
    }

  /* Can't determine prologue from the symbol table, need to examine
     instructions.  */

  return start_pc;
}

static struct gdbarch *
amdgcn_gdbarch_init (struct gdbarch_info info, struct gdbarch_list *arches)
{
  /* If there is already a candidate, use it.  */
  arches = gdbarch_list_lookup_by_info (arches, &info);
  if (arches != NULL)
    return arches->gdbarch;

  struct gdbarch_deleter
  {
    void
    operator() (struct gdbarch *gdbarch) const
    {
      gdbarch_free (gdbarch);
    }
  };

  /* Allocate space for the new architecture.  */
  std::unique_ptr<struct gdbarch_tdep> tdep (new struct gdbarch_tdep);
  std::unique_ptr<struct gdbarch, gdbarch_deleter> gdbarch_u (
    gdbarch_alloc (&info, tdep.get ()));

  struct gdbarch *gdbarch = gdbarch_u.get ();

  /* Data types.  */
  set_gdbarch_char_signed (gdbarch, 0);
  set_gdbarch_ptr_bit (gdbarch, 64);
  set_gdbarch_addr_bit (gdbarch, 64);
  set_gdbarch_short_bit (gdbarch, 16);
  set_gdbarch_int_bit (gdbarch, 32);
  set_gdbarch_long_bit (gdbarch, 64);
  set_gdbarch_long_long_bit (gdbarch, 64);
  set_gdbarch_float_bit (gdbarch, 32);
  set_gdbarch_double_bit (gdbarch, 64);
  set_gdbarch_long_double_bit (gdbarch, 128);
  set_gdbarch_float_format (gdbarch, floatformats_ieee_single);
  set_gdbarch_double_format (gdbarch, floatformats_ieee_double);
  set_gdbarch_long_double_format (gdbarch, floatformats_ieee_double);

  /* Frame Interpretation.  */
  set_gdbarch_skip_prologue (gdbarch, amdgcn_skip_prologue);
  set_gdbarch_inner_than (gdbarch, core_addr_greaterthan);
  dwarf2_append_unwinders (gdbarch);
  frame_unwind_append_unwinder (gdbarch, &amdgcn_frame_unwind);
  set_gdbarch_dummy_id (gdbarch, amdgcn_dummy_id);

  /* Registers and Memory.  */
  amd_dbgapi_architecture_id_t architecture_id;
  if (amd_dbgapi_get_architecture (gdbarch_bfd_arch_info (gdbarch)->mach,
				   &architecture_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return nullptr;

  size_t register_class_count;
  amd_dbgapi_register_class_id_t *register_class_ids;

  if (amd_dbgapi_architecture_register_class_list (architecture_id,
						   &register_class_count,
						   &register_class_ids)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return nullptr;

  /* Add register groups.  */
  reggroup_add (gdbarch, all_reggroup);

  for (size_t i = 0; i < register_class_count; ++i)
    {
      char *bytes;
      if (amd_dbgapi_architecture_register_class_get_info (
	    register_class_ids[i], AMD_DBGAPI_REGISTER_CLASS_INFO_NAME,
	    sizeof (bytes), &bytes)
	  != AMD_DBGAPI_STATUS_SUCCESS)
	continue;

      gdb::unique_xmalloc_ptr<char> name (bytes);

      auto inserted = tdep->register_class_map.emplace (name.get (),
							register_class_ids[i]);

      if (!inserted.second)
	continue;

      /* Allocate the reggroup in the gdbarch.  */
      auto *group = reggroup_gdbarch_new (gdbarch, name.get (), USER_REGGROUP);
      if (!group)
	{
	  tdep->register_class_map.erase (inserted.first);
	  continue;
	}

      reggroup_add (gdbarch, group);
    }
  xfree (register_class_ids);

  /* Add registers. */
  size_t register_count;
  amd_dbgapi_register_id_t *register_ids;

  if (amd_dbgapi_architecture_register_list (architecture_id, &register_count,
					     &register_ids)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return nullptr;

  tdep->register_ids.insert (tdep->register_ids.end (), &register_ids[0],
			     &register_ids[register_count]);

  set_gdbarch_num_regs (gdbarch, register_count);
  set_gdbarch_num_pseudo_regs (gdbarch, 0);
  xfree (register_ids);

  tdep->register_names.resize (register_count);
  for (size_t i = 0; i < register_count; ++i)
    {
      if (!tdep->regnum_map.emplace (tdep->register_ids[i], i).second)
	return nullptr;

      char *bytes;
      if (amd_dbgapi_register_get_info (tdep->register_ids[i],
					AMD_DBGAPI_REGISTER_INFO_NAME,
					sizeof (bytes), &bytes)
	  != AMD_DBGAPI_STATUS_SUCCESS)
	continue;

      tdep->register_names[i] = bytes;
      xfree (bytes);
    }

  amd_dbgapi_register_id_t pc_register_id;
  if (
    amd_dbgapi_architecture_get_info (architecture_id,
				      AMD_DBGAPI_ARCHITECTURE_INFO_PC_REGISTER,
				      sizeof (pc_register_id), &pc_register_id)
    != AMD_DBGAPI_STATUS_SUCCESS)
    return nullptr;

  set_gdbarch_pc_regnum (gdbarch, tdep->regnum_map[pc_register_id]);
  set_gdbarch_ps_regnum (gdbarch, -1);
  set_gdbarch_sp_regnum (gdbarch, -1);
  set_gdbarch_fp0_regnum (gdbarch, -1);

  set_gdbarch_dwarf2_reg_to_regnum (gdbarch, amdgcn_dwarf_reg_to_regnum);

  set_gdbarch_return_value (gdbarch, amdgcn_return_value);

  /* Register Representation.  */
  set_gdbarch_register_name (gdbarch, amdgcn_register_name);
  set_gdbarch_register_type (gdbarch, amdgcn_register_type);
  set_gdbarch_register_reggroup_p (gdbarch, amdgcn_register_reggroup_p);

  /* Disassembly.  */
  set_gdbarch_print_insn (gdbarch, print_insn_amdgcn);

  /* Instructions.  */
  amd_dbgapi_size_t max_insn_length = 0;
  if (amd_dbgapi_architecture_get_info (
	architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_LARGEST_INSTRUCTION_SIZE,
	sizeof (max_insn_length), &max_insn_length)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("amd_dbgapi_architecture_get_info failed"));

  set_gdbarch_max_insn_length (gdbarch, max_insn_length);

  if (amd_dbgapi_architecture_get_info (
	architecture_id,
	AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE,
	sizeof (tdep->breakpoint_instruction_size),
	&tdep->breakpoint_instruction_size)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("amd_dbgapi_architecture_get_info failed"));

  gdb_byte *breakpoint_instruction_bytes;
  if (amd_dbgapi_architecture_get_info (
	architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION,
	sizeof (breakpoint_instruction_bytes), &breakpoint_instruction_bytes)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("amd_dbgapi_architecture_get_info failed"));

  tdep->breakpoint_instruction_bytes.reset (breakpoint_instruction_bytes);

  set_gdbarch_breakpoint_kind_from_pc (gdbarch,
				       amdgcn_breakpoint_kind_from_pc);
  set_gdbarch_sw_breakpoint_from_kind (gdbarch,
				       amdgcn_sw_breakpoint_from_kind);

  amd_dbgapi_size_t pc_adjust;
  if (amd_dbgapi_architecture_get_info (
	architecture_id,
	AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_PC_ADJUST,
	sizeof (pc_adjust), &pc_adjust)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("amd_dbgapi_architecture_get_info failed"));

  set_gdbarch_decr_pc_after_break (gdbarch, pc_adjust);

  tdep.release ();
  gdbarch_u.release ();

  return gdbarch;
}

/* Provide a prototype to silence -Wmissing-prototypes.  */
extern initialize_file_ftype _initialize_amdgcn_rocm_tdep;

void
_initialize_amdgcn_rocm_tdep (void)
{
  gdbarch_register (bfd_arch_amdgcn, amdgcn_gdbarch_init, NULL);
}
