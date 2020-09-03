/* Builtin frame register, for GDB, the GNU debugger.

   Copyright (C) 2002-2024 Free Software Foundation, Inc.

   Contributed by Red Hat.

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

#include "user-regs.h"
#include "frame.h"
#include "gdbtypes.h"
#include "value.h"
#include "gdbarch.h"
#include "block.h"
#include "objfiles.h"
#include "dwarf2/loc.h"
#include "gdbthread.h"

static struct value *
value_of_builtin_frame_fp_reg (const frame_info_ptr &frame, const void *baton)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);

  if (gdbarch_deprecated_fp_regnum (gdbarch) >= 0)
    /* NOTE: cagney/2003-04-24: Since the mere presence of "fp" in the
       register name table overrides this built-in $fp register, there
       is no real reason for this gdbarch_deprecated_fp_regnum trickery here.
       An architecture wanting to implement "$fp" as alias for a raw
       register can do so by adding "fp" to register name table (mind
       you, doing this is probably a dangerous thing).  */
    return value_of_register (gdbarch_deprecated_fp_regnum (gdbarch),
			      get_next_frame_sentinel_okay (frame));
  else
    {
      struct type *data_ptr_type = builtin_type (gdbarch)->builtin_data_ptr;
      struct value *val = value::allocate (data_ptr_type);
      gdb_byte *buf = val->contents_raw ().data ();

      gdbarch_address_to_pointer (gdbarch, data_ptr_type,
				  buf, get_frame_base_address (frame));
      return val;
    }
}

static struct value *
value_of_builtin_frame_pc_reg (const frame_info_ptr &frame, const void *baton)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);

  if (gdbarch_pc_regnum (gdbarch) >= 0)
    return value_of_register (gdbarch_pc_regnum (gdbarch),
			      get_next_frame_sentinel_okay (frame));
  else
    {
      struct type *func_ptr_type = builtin_type (gdbarch)->builtin_func_ptr;
      struct value *val = value::allocate (func_ptr_type);
      gdb_byte *buf = val->contents_raw ().data ();

      gdbarch_address_to_pointer (gdbarch, func_ptr_type,
				  buf, get_frame_pc (frame));
      return val;
    }
}

static struct value *
value_of_builtin_frame_sp_reg (const frame_info_ptr &frame, const void *baton)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);

  if (gdbarch_sp_regnum (gdbarch) >= 0)
    return value_of_register (gdbarch_sp_regnum (gdbarch),
			      get_next_frame_sentinel_okay (frame));
  error (_("Standard register ``$sp'' is not available for this target"));
}

static struct value *
value_of_builtin_frame_ps_reg (const frame_info_ptr &frame, const void *baton)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);

  if (gdbarch_ps_regnum (gdbarch) >= 0)
    return value_of_register (gdbarch_ps_regnum (gdbarch),
			      get_next_frame_sentinel_okay (frame));
  error (_("Standard register ``$ps'' is not available for this target"));
}

/* Return the value of the $lane_pc virtual register.  */

static struct value *
value_of_builtin_frame_lane_pc_reg (const frame_info_ptr &frame,
				    const void *baton)
{
  value *pc = get_frame_lane_pc_val (frame);
  if (pc != nullptr && !pc->optimized_out ())
    return pc;

  /* If we don't have an DW_AT_LLVM_lanes attribute, return $pc.  */
  return value_of_builtin_frame_pc_reg (frame, baton);
}

/* Return the value of the $__lane_pc_array virtual register.  */

static struct value *
value_of_builtin_frame_lane_pc_array_reg (const frame_info_ptr &frame,
					  const void *baton)
{
  return get_frame_lane_pc_array_val (frame);
}

void _initialize_frame_reg ();
void
_initialize_frame_reg ()
{
  /* Frame based $fp, $pc, $sp and $ps.  These only come into play
     when the target does not define its own version of these
     registers.  */
  user_reg_add_builtin ("fp", value_of_builtin_frame_fp_reg, NULL);
  user_reg_add_builtin ("pc", value_of_builtin_frame_pc_reg, NULL);
  user_reg_add_builtin ("sp", value_of_builtin_frame_sp_reg, NULL);
  user_reg_add_builtin ("ps", value_of_builtin_frame_ps_reg, NULL);

  /* The lane's contextual PC.  If the thread does not have lanes, or
     if the lane is active, then this returns $pc.  */
  user_reg_add_builtin ("lane_pc", value_of_builtin_frame_lane_pc_reg,
			nullptr);

  /* Internal/maintainer version of $lane_pc that returns the whole
     array value returned by evaluating DW_AT_LLVM_lane_pc.  This is
     considered an internal variable -- lanes that were inactive on
     entry will have a __lane_pc_array entry of <optimized out>, which
     probably isn't what users would expect.  For such lanes, to find
     the lane PC, we'd need to evaluate DW_AT_LLVM_lane_pc in the
     caller frame (recursively, until we find a non-optimized-out
     value).  */
  user_reg_add_builtin ("__lane_pc_array",
			value_of_builtin_frame_lane_pc_array_reg, nullptr);
}
