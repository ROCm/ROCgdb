/* Everything about catching HIP API errors ("catch hiperr").

   Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.

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

#include "annotate.h"
#include "arch-utils.h"
#include "breakpoint.h"
#include "cli/cli-decode.h"
#include "extract-store-integer.h"
#include "inferior.h"
#include "gdbarch.h"
#include "progspace.h"
#include "language.h"
#include "stack.h"
#include "valprint.h"

struct hiperr_catchpoint : public code_breakpoint
{
  hiperr_catchpoint (struct gdbarch *gdbarch, bool temp,
		     const char *cond_string)
    : code_breakpoint (gdbarch, bp_catchpoint, temp, cond_string)
  {
    /* Make sure "locspec" is initialized, irrespective of the
       "__hipOnError ()" symbol being found.  This allows listing the
       breakpoint as a pending one when the binary is not loaded yet.
       Otherwise, "info breakpoints" is going to trigger an assert.  */
    locspec = new_explicit_location_spec_function ("__hipOnError");
    pspace = current_program_space;
    re_set (nullptr);
  }

  bp_location *allocate_location () override;
  void re_set (program_space *pspace) override;
  bool print_one (const bp_location **last_loc) const override;
  void print_mention () const override;
  void print_recreate (struct ui_file *fp) const override;
  enum print_stop_action print_it (const bpstat *bs) const override;
};

/* Implement the "allocate_location" method for hiperr catchpoint.  */

bp_location *
hiperr_catchpoint::allocate_location ()
{
  return new bp_location (this, bp_loc_software_breakpoint);
}

/* Implement the "re_set" method for hiperr catchpoint.  It is used
   when a breakpoint must be re-evaluated, like at symbols change.  */

void
hiperr_catchpoint::re_set (program_space * /*ps*/)
{
  std::vector<symtab_and_line> sals;

  try
    {
      sals = this->decode_location_spec (this->locspec.get (),
					 current_program_space);
    }
  catch (const gdb_exception_error &ex)
    {
      /* NOT_FOUND_ERROR is the result of a pending breakpoint.
	 Anything else, must be rethrown.  */
      if (ex.error != NOT_FOUND_ERROR)
	throw;
    }
  update_breakpoint_locations (this, current_program_space, sals, {});
}

/* Implement the "print_one" method for hiperr catchpoint.  It
   displays information about the breakpoint ("info breakpoints").  */

bool
hiperr_catchpoint::print_one (const bp_location **last_loc) const
{
  struct value_print_options opts;
  struct ui_out *uiout = current_uiout;

  get_user_print_options (&opts);

  if (opts.addressprint)
    uiout->field_skip ("addr");
  annotate_field (5);

  uiout->field_string ("what", "catch hiperr");
  if (uiout->is_mi_like_p ())
    uiout->field_string ("catch-type", "hiperr");

  return true;
}

/* Message printed when the breakpoint is set.  */

void
hiperr_catchpoint::print_mention () const
{
  struct ui_out *uiout = current_uiout;
  const char *bp_type = "Catchpoint";

  if (disposition == disp_del)
    bp_type = "Temporary catchpoint";

  uiout->message ("%s %d (HIP error)", _(bp_type), number);
}

/* Implement the "print_recreate" method for catch hiperr.
   It's used for saving the breakpoints.  */

void
hiperr_catchpoint::print_recreate (struct ui_file *fp) const
{
  bool bp_temp = (disposition == disp_del);
  gdb_printf (fp, bp_temp ? "tcatch hiperr" : "catch hiperr");
  /* The condition, if any, is read from the object's "cond_string".  */
  print_recreate_thread (fp);
}

/* Check if FRAME is the "__hipOnError ()" frame.  */

static bool
hiperr_frame_p (frame_info_ptr frame)
{
  enum language func_lang;

  gdb::unique_xmalloc_ptr<char> func_name
    = find_frame_funname (frame, &func_lang, nullptr);
  if (func_name == nullptr || strcmp (func_name.get (), "__hipOnError") != 0)
    return false;

  return true;
}

/* Message printed when the breakpoint is hit.  */

enum print_stop_action
hiperr_catchpoint::print_it (const bpstat *bs) const
{
  struct ui_out *uiout = current_uiout;
  const char *bp_type = "Catchpoint ";

  annotate_catchpoint (number);
  maybe_print_thread_hit_breakpoint (uiout);

  if (disposition == disp_del)
    bp_type = "Temporary catchpoint ";
  uiout->text (bp_type);

  print_num_locno (bs, uiout);
  uiout->text (" (HIP error)\n");

  std::optional<hiperr_parameters> err_params;
  frame_info_ptr frame = get_selected_frame (_("No frame selected"));
  struct gdbarch *barch = bs->bp_location_at->gdbarch;
  if (hiperr_frame_p (frame) && gdbarch_fetch_hiperr_parameters_p (barch))
    err_params = gdbarch_fetch_hiperr_parameters (barch, frame);

  if (err_params.has_value ())
    {
      const std::string hiperr_info
	= string_printf (_("HIP API call failed with error %s (%u): %s\n"),
			 err_params->err_name.get (),
			 err_params->err_no,
			 err_params->err_str.get ());
      uiout->text (hiperr_info.c_str ());
      uiout->text
	("\nThe $_hiperr convenience variable holds the error number.\n");
    }
  else
    uiout->text ("\nwarning: failed to extract HIP error information.\n");

  if (uiout->is_mi_like_p ())
    {
      uiout->field_string ("reason",
			   async_reason_lookup (EXEC_ASYNC_BREAKPOINT_HIT));
      uiout->field_string ("disp", bpdisp_text (disposition));

      if (err_params.has_value ())
	{
	  uiout->field_unsigned ("hiperr-code", err_params->err_no);
	  uiout->field_string ("hiperr-name", err_params->err_name.get ());
	  uiout->field_string ("hiperr-text", err_params->err_str.get ());
	}
    }

  return PRINT_NOTHING;
}

/* Instantiate a hiperr_catchpoint.  */

void
add_catch_hiperr (const char *condition, bool tempflag)
{
  auto hcp = std::make_unique<hiperr_catchpoint> (get_current_arch (),
						  tempflag, condition);
  install_breakpoint (0, std::move (hcp), 1);
}

/* Implementation of "catch hiperr" command.  */

static void
catch_hiperr_command (const char *arg,
		      int /*from_tty*/,
		      struct cmd_list_element *command)
{
  bool tempflag = command->context () == CATCH_TEMPORARY;
  const char *cond_string = nullptr;

  if (arg == nullptr)
    arg = "";
  arg = skip_spaces (arg);
  cond_string = ep_parse_optional_if_clause (&arg);

  if ((*arg != '\0') && !c_isspace (*arg))
    error (_("Junk at the end of arguments."));

  add_catch_hiperr (cond_string, tempflag);
}



/* Implement the 'make_value' method for the $_hiperr internalvar.  */

static struct value *
compute_hiperr (struct gdbarch *gdbarch,
		struct internalvar * /*var*/,
		void * /*ignore*/)
{
  frame_info_ptr frame = get_selected_frame (_("No frame selected"));

  if (!hiperr_frame_p (frame) || !gdbarch_fetch_hiperr_parameters_p (gdbarch))
    return value::allocate (builtin_type (gdbarch)->builtin_void);

  std::optional<hiperr_parameters> err_params
    = gdbarch_fetch_hiperr_parameters (gdbarch, frame);

  if (!err_params.has_value ())
    return value::allocate (builtin_type (gdbarch)->builtin_void);

  /* If there's a "hipError_t" type in the debug symbols that is
     a 4-byte enum, use it.  */
  struct block_symbol bs
    = lookup_symbol ("hipError_t", nullptr, SEARCH_TYPE_DOMAIN, nullptr);
  if (bs.symbol != nullptr
      && bs.symbol->type ()->length () == 4
      && bs.symbol->type ()->code () == TYPE_CODE_ENUM)
    {
      gdb_byte err_no[4];
      const bfd_endian byte_order = gdbarch_byte_order (gdbarch);
      store_unsigned_integer (err_no, 4, byte_order, err_params->err_no);
      return value_from_contents (bs.symbol->type (), err_no);
    }

  /* Return hiperr as an unsigned int.  */
  return value_from_ulongest (builtin_type (gdbarch)->builtin_uint32,
			      err_params->err_no);
}

/* Implementation of the '$_hiperr' variable.  */

static const struct internalvar_funcs hiperr_funcs =
{
  compute_hiperr,
  nullptr,
};



INIT_GDB_FILE (break_catch_hiperr)
{
  /* Add "(t)catch hiperr" sub-command.  */
  add_catch_command ("hiperr",
		     _("Catch a HIP error."),
		     catch_hiperr_command,
		     nullptr,
		     CATCH_PERMANENT,
		     CATCH_TEMPORARY);

  create_internalvar_type_lazy ("_hiperr", &hiperr_funcs, nullptr);
}
