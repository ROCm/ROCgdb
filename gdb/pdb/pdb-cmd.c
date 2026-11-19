/* PDB debugging format support for GDB.

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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

   This file implements the PDB-related maintenance commands.  */

#include "cli/cli-utils.h"
#include "cli/cli-cmds.h"
#include "objfiles.h"
#include "pdb/pdb-internal.h"
#include "gdbsupport/rsp-low.h"
#include "progspace.h"
#include "event-top.h"
#include "top.h"
#include "cli/cli-setshow.h"
#include "gdbsupport/buildargv.h"
#include "gdbsupport/selftest.h"
#include "safe-ctype.h"

#include <string.h>
#include <stdint.h>
#include <array>

namespace pdb
{

/* Split ARGS into its whitespace-separated words, the way GDB splits a
   command's arguments.  An empty or null ARGS has no words.  */

static std::vector<std::string>
get_arg_tokens (const char *args)
{
  std::vector<std::string> tokens;
  if (args == nullptr)
    return tokens;
  for (const char *arg : gdb_argv (args))
    tokens.emplace_back (arg);
  return tokens;
}

/* Find the argument KEY=VALUE in ARGS and return true with *OUT_VALUE
   set to VALUE.  KEY matches in any case.  Return false when ARGS has no
   such argument.  */

static bool
get_info_arg (const char *args, const char *key, std::string *out_value)
{
  size_t key_len = strlen (key);
  for (const std::string &token : get_arg_tokens (args))
    if (token.size () > key_len && token[key_len] == '='
	&& strncasecmp (token.c_str (), key, key_len) == 0)
      {
	*out_value = token.substr (key_len + 1);
	return true;
      }

  return false;
}

/* Check that every argument of ARGS is one the command takes, and throw
   an error naming the first that is not.  An argument must have the form
   KEY=VALUE, with KEY one of VALID_KEYS in any case.  */

static void
validate_args (const char *args, const std::vector<std::string> &valid_keys)
{
  for (const std::string &token : get_arg_tokens (args))
    {
      size_t eq_pos = token.find ('=');
      bool found = false;
      if (eq_pos != std::string::npos)
	for (const std::string &valid_key : valid_keys)
	  if (strcasecmp (token.substr (0, eq_pos).c_str (),
			  valid_key.c_str ()) == 0)
	    {
	      found = true;
	      break;
	    }

      if (!found)
	{
	  std::string keys;
	  for (const std::string &valid_key : valid_keys)
	    keys += (keys.empty () ? "" : ", ") + valid_key + "=";
	  error (_("Unknown argument \"%s\"; the arguments are %s."),
		 token.c_str (), keys.c_str ());
	}
    }
}

/* Convert the module index STR to a number and return true with
   *OUT_VALUE set.  STR must be a decimal number of at most 9 digits with
   nothing else around it, so that a sign, a space or an overflow is
   rejected; return false otherwise.  */

static bool
string_to_int (const std::string &str, unsigned long *out_value)
{
  if (str.empty () || str.size () > 9
      || str.find_first_not_of ("0123456789") != std::string::npos)
    return false;

  *out_value = strtoul (str.c_str (), nullptr, 10);
  return true;
}

/* Return the PDB registered for PATH, or nullptr.  */

static pdb_per_objfile *
pdb_find_loaded_pdb (const char *path)
{
  if (path == nullptr)
    return nullptr;

  for (objfile &objf : current_program_space->objfiles ())
    {
      pdb_per_objfile *pdb = get_pdb_per_objfile (&objf);
      if (pdb != nullptr && !pdb->pdb_file_path.empty ()
	  && pdb->pdb_file_path == path)
	return pdb;
    }

  return nullptr;
}

/* Return the default PDB (the one whose objfile has OBJF_MAINLINE),
   or nullptr if none.  */

static pdb_per_objfile *
pdb_get_default ()
{
  for (objfile &objf : current_program_space->objfiles ())
    {
      if ((objf.flags & OBJF_MAINLINE) == 0)
	continue;
      pdb_per_objfile *pdb = get_pdb_per_objfile (&objf);
      if (pdb != nullptr)
	return pdb;
    }

  return nullptr;
}

/* Return the loaded PDB whose path is PATH.  The commands work only on
   the PDBs of the loaded objfiles, so throw an error for any other
   path.  */

static pdb_per_objfile *
pdb_require_loaded_path (const std::string &path)
{
  pdb_per_objfile *pdb = pdb_find_loaded_pdb (path.c_str ());
  if (pdb == nullptr)
    error (_("PDB not loaded: %s"), path.c_str ());
  return pdb;
}

/* Return the PDB a command works on: the one the path= argument of ARGS
   names, or else the main program's.  Throw an error when that PDB is
   not loaded.  */

static pdb_per_objfile *
pdb_select_pdb (const char *args)
{
  if (std::string path; get_info_arg (args, "path", &path))
    return pdb_require_loaded_path (path);

  pdb_per_objfile *pdb = pdb_get_default ();
  if (pdb == nullptr)
    error (_("No PDB loaded."));
  return pdb;
}

/* Return the name of the checksum kind TYPE, a CHKSUM_TYPE_* value, as
   "maint info pdb-files-c13" prints it.  */

static const char *
pdb_checksum_type_name (uint8_t type)
{
  switch (type)
    {
    case CHKSUM_TYPE_NONE:
      return "None";
    case CHKSUM_TYPE_MD5:
      return "MD5";
    case CHKSUM_TYPE_SHA1:
      return "SHA-1";
    case CHKSUM_TYPE_SHA_256:
      return "SHA-256";
    default:
      return "Unknown";
    }
}

/* Select the PDB and the modules a command that prints module data works
   on.  The command takes path= and modi=, and the arguments EXTRA_KEYS
   as well.  Set *OUT_PDB to the PDB that path= names, or to the main
   program's.  Set [*OUT_MOD_START, *OUT_MOD_END) to the module that
   modi= names, or to all modules.  Throw an error for an argument the
   command does not take, a PDB that is not loaded or a module index past
   the PDB's modules.  */

static void
prepare_info_command (const char *args, pdb_per_objfile **out_pdb,
		      uint32_t *out_mod_start, uint32_t *out_mod_end)
{
  std::vector<std::string> valid = { "path", "modi" };
  validate_args (args, valid);

  pdb_per_objfile *pdb = pdb_select_pdb (args);
  *out_pdb = pdb;
  *out_mod_start = 0;
  *out_mod_end = pdb->modules.size ();

  std::string modi_str;
  if (get_info_arg (args, "modi", &modi_str))
    {
      unsigned long modi;
      if (!string_to_int (modi_str, &modi))
	error (_("Invalid module index: %s"), modi_str.c_str ());
      if (modi >= pdb->modules.size ())
	error (_("Module index %lu out of range: the PDB has %zu modules."),
	       modi, pdb->modules.size ());

      *out_mod_start = modi;
      *out_mod_end = modi + 1;
    }
}

/* Print one line block for "maint info pdb-lines": the file, the
   section range the block covers, and its line/offset pairs.  This is
   the callback pdb_walk_c13_line_blocks calls for each block.  */

static void
pdb_dump_line_block (const char *filename, CV_LineSection ls,
		     const gdb_byte *lines, uint32_t num_lines)
{
  QUIT;
  /* Compute the range the block and its lines cover in 64 bits, as the
     offsets the file gives can add up past 32 bits.  */
  uint64_t range_start = ls.offs;
  uint64_t range_end = (uint64_t) ls.offs + ls.code_sz;
  if (num_lines > 0)
    {
      CV_Line first (lines);
      CV_Line last (lines + (num_lines - 1) * CV_Line::SIZE);
      range_start = (uint64_t) ls.offs + first.offs;
      range_end = std::max (range_end, (uint64_t) ls.offs + last.offs + 1);
    }

  gdb_printf ("    %s\n", filename);
  gdb_printf ("      %04X:%08llX-%08llX, line/addr entries = %u\n",
	      (unsigned) ls.seci, (unsigned long long) range_start,
	      (unsigned long long) range_end, num_lines);

  /* Print line/offset pairs, ~8 per row.  */
  gdb_printf ("        ");
  for (uint32_t j = 0; j < num_lines; j++)
    {
      CV_Line line (lines + j * CV_Line::SIZE);
      uint64_t offs = (uint64_t) ls.offs + line.offs;
      gdb_printf ("%u %08llX", line.line_start, (unsigned long long) offs);
      if (j + 1 < num_lines)
	{
	  if ((j + 1) % 8 == 0)
	    gdb_printf ("\n        ");
	  else
	    gdb_printf ("    ");
	}
    }

  gdb_printf ("\n\n");
}

/* 'maint info pdb-modules' command.  */

static void
pdb_info_modules_command (const char *args, int from_tty)
{
  pdb_per_objfile *pdb;
  uint32_t mod_start;
  uint32_t mod_end;
  prepare_info_command (args, &pdb, &mod_start, &mod_end);

  gdb_printf ("PDB '%s'\n", !pdb->pdb_file_path.empty ()
			    ? pdb->pdb_file_path.c_str ()
			    : "<unknown>");

  for (uint32_t i = mod_start; i < mod_end; i++)
    {
      QUIT;
      pdb_module_info &mod = pdb->modules[i];
      pdb_read_module_files (pdb, &mod);

      gdb_printf ("  Mod %04u | `%s`:\n", i,
		  mod.module_name != nullptr ? mod.module_name : "<unnamed>");
      gdb_printf ("           Obj: `%s`:\n",
		  mod.obj_file_name != nullptr ? mod.obj_file_name : "<none>");
      gdb_printf ("           debug stream: %u, # files: %u\n",
		  (unsigned) mod.stream_number, mod.num_files);
      gdb_printf ("           symbol bytes: %u, C11 bytes: %u,"
		  " C13 bytes: %u\n",
		  mod.sym_byte_size, mod.c11_byte_size, mod.c13_byte_size);
    }
}

/* 'maintenance info pdb-loaded-files' command.  */

static void
pdb_info_loaded_files_command (const char *args, int from_tty)
{
  if (args != nullptr && *args != '\0')
    error (_("Usage: maintenance info pdb-loaded-files"));

  bool found = false;
  for (objfile &objf : current_program_space->objfiles ())
    {
      const pdb_per_objfile *pdb = get_pdb_per_objfile (&objf);
      if (pdb == nullptr || pdb->pdb_file_path.empty ())
	continue;
      gdb_printf ("%s\n", pdb->pdb_file_path.c_str ());
      found = true;
    }

  if (!found)
    gdb_printf (_("No PDB loaded.\n"));
}

/* 'maint info pdb-lines' command.  */

static void
pdb_info_lines_command (const char *args, int from_tty)
{
  pdb_per_objfile *pdb;
  uint32_t mod_start;
  uint32_t mod_end;

  prepare_info_command (args, &pdb, &mod_start, &mod_end);

  for (uint32_t i = mod_start; i < mod_end; i++)
    {
      QUIT;
      pdb_module_info *mod = &pdb->modules[i];
      /* The file names come from the C13 checksums, which reading the
	 stream keeps.  */
      auto stream_buf = pdb_read_module_stream (pdb, mod);

      gdb_printf ("  Mod %04u | `%s`:\n", i,
		  mod->module_name != nullptr ? mod->module_name : "<unnamed>");

      if (!stream_buf.empty () && mod->c13_byte_size > 0)
	pdb_walk_c13_line_blocks (pdb, mod, stream_buf.data (),
				  pdb_dump_line_block);
    }
}

/* 'maint info pdb-files' command.  */

static void
pdb_info_files_command (const char *args, int from_tty)
{
  pdb_per_objfile *pdb;
  uint32_t mod_start;
  uint32_t mod_end;

  prepare_info_command (args, &pdb, &mod_start, &mod_end);

  gdb_printf ("PDB '%s'\n", !pdb->pdb_file_path.empty ()
			      ? pdb->pdb_file_path.c_str ()
			      : "<unknown>");

  for (auto i = mod_start; i < mod_end; i++)
    {
      QUIT;
      pdb_module_info *mod = &pdb->modules[i];
      pdb_read_module_files (pdb, mod);

      gdb_printf ("  Mod %04u | `%s`:\n", i,
		  mod->module_name != nullptr ? mod->module_name : "<unnamed>");

      for (uint32_t j = 0; j < mod->num_files; j++)
	{
	  QUIT;
	  const char *fname = mod->files != nullptr ? mod->files[j] : nullptr;
	  gdb_printf ("    - %s\n", fname != nullptr ? fname : "<unknown>");
	}
    }
}

/* 'maint info pdb-files-c13' command.  */

static void
pdb_info_files_c13_command (const char *args, int from_tty)
{
  pdb_per_objfile *pdb;
  uint32_t mod_start;
  uint32_t mod_end;

  prepare_info_command (args, &pdb, &mod_start, &mod_end);

  gdb_printf ("PDB '%s'\n", !pdb->pdb_file_path.empty ()
			      ? pdb->pdb_file_path.c_str ()
			      : "<unknown>");

  for (uint32_t i = mod_start; i < mod_end; i++)
    {
      QUIT;
      pdb_module_info *mod = &pdb->modules[i];
      pdb_read_module_stream (pdb, mod);
      pdb_read_module_files_c13 (pdb, mod);

      gdb_printf ("  Mod %04u | `%s`:\n", i,
		  mod->module_name != nullptr ? mod->module_name : "<unnamed>");

      for (uint32_t j = 0; j < mod->num_files_c13; j++)
	{
	  const pdb_file_info *fi = &mod->files_c13[j];

	  if (fi->checksum_size > 0 && fi->checksum != nullptr)
	    {
	      /* Print checksum type and hash value.  */
	      std::string hash_str = bin2hex (fi->checksum, fi->checksum_size);

	      gdb_printf (" [%s: %s] ",
			  pdb_checksum_type_name (fi->checksum_type),
			  hash_str.c_str ());
	    }

	  gdb_printf ("%s\n",
		      fi->filename != nullptr ? fi->filename : "<unknown>");
	}
    }
}

/* The CodeView leaf kinds "maint info pdb-types" knows by name.  The
   command prints these names and accepts them for kind=.  */

struct pdb_leaf_kind
{
  const char *name;
  uint16_t leaf;
};

static const pdb_leaf_kind pdb_leaf_kinds[] = {
  { "LF_VTSHAPE", LF_VTSHAPE },
  { "LF_LABEL", LF_LABEL },
  { "LF_MODIFIER", LF_MODIFIER },
  { "LF_POINTER", LF_POINTER },
  { "LF_PROCEDURE", LF_PROCEDURE },
  { "LF_MFUNCTION", LF_MFUNCTION },
  { "LF_ARGLIST", LF_ARGLIST },
  { "LF_FIELDLIST", LF_FIELDLIST },
  { "LF_BITFIELD", LF_BITFIELD },
  { "LF_METHODLIST", LF_METHODLIST },
  { "LF_ARRAY", LF_ARRAY },
  { "LF_CLASS", LF_CLASS },
  { "LF_STRUCTURE", LF_STRUCTURE },
  { "LF_UNION", LF_UNION },
  { "LF_ENUM", LF_ENUM },
  { "LF_INTERFACE", LF_INTERFACE },
};

/* Return the name of the CodeView leaf LEAF, or LF_0xNNNN for a leaf
   that has no name in pdb_leaf_kinds.  */

static std::string
pdb_leaf_type_name (uint16_t leaf)
{
  for (const pdb_leaf_kind &k : pdb_leaf_kinds)
    if (k.leaf == leaf)
      return k.name;
  return string_printf ("LF_0x%04x", leaf);
}

/* Return the name TPI type record REC gives, or nullptr for a record
   that has no name or whose name does not fit in it.  */

static const char *
pdb_tpi_record_name (const pdb_tpi_type *rec)
{
  uint64_t size;
  switch (rec->leaf)
    {
    case LF_CLASS:
    case LF_STRUCTURE:
      return pdb_cv_numeric_and_name (rec, LF_STRUCT_DATA_OFFS, &size);
    case LF_UNION:
      return pdb_cv_numeric_and_name (rec, LF_UNION_DATA_OFFS, &size);
    case LF_ARRAY:
      return pdb_cv_numeric_and_name (rec, LF_ARRAY_DATA_OFFS, &size);
    case LF_ENUM:
      if (rec->data_len > LF_ENUM_NAME_OFFS)
	return pdb_extract_string (rec->data + LF_ENUM_NAME_OFFS,
				   rec->data + rec->data_len);
      return nullptr;
    default:
      return nullptr;
    }
}

/* Return a short name for type index TI, for "maint info pdb-types".  A
   simple type (below 0x1000) gets its CodeView name, with "*" appended
   for a pointer to it.  A TPI type gets the name its record gives.
   Return "" for a type that has no name, such as an LF_POINTER or an
   LF_MODIFIER.  */

static std::string
pdb_type_index_name (const pdb_tpi_context *tpi, uint32_t ti)
{
  if (cv_ti_is_simple (ti))
    {
      uint32_t kind = cv_simple_kind (ti);
      uint32_t mode = cv_simple_mode (ti);
      const char *base = nullptr;

      switch (kind)
	{
	case CV_NONE:
	  base = "<none>";
	  break;
	case CV_VOID:
	  base = "void";
	  break;
	case CV_HRESULT:
	  base = "HRESULT";
	  break;
	case CV_SIGNED_CHAR:
	  base = "signed char";
	  break;
	case CV_UNSIGNED_CHAR:
	  base = "unsigned char";
	  break;
	case CV_NARROW_CHAR:
	  base = "char";
	  break;
	case CV_WIDE_CHAR:
	  base = "wchar_t";
	  break;
	case CV_CHAR16:
	  base = "char16_t";
	  break;
	case CV_CHAR32:
	  base = "char32_t";
	  break;
	case CV_CHAR8:
	  base = "char8_t";
	  break;
	case CV_INT8:
	  base = "int8_t";
	  break;
	case CV_UINT8:
	  base = "uint8_t";
	  break;
	case CV_SHORT:
	  base = "short";
	  break;
	case CV_USHORT:
	  base = "unsigned short";
	  break;
	case CV_INT16:
	  base = "int16_t";
	  break;
	case CV_UINT16:
	  base = "uint16_t";
	  break;
	case CV_INT32:
	  base = "int";
	  break;
	case CV_UINT32:
	  base = "unsigned int";
	  break;
	case CV_LONG:
	  base = "long";
	  break;
	case CV_ULONG:
	  base = "unsigned long";
	  break;
	case CV_QUAD:
	  base = "__int64";
	  break;
	case CV_UQUAD:
	  base = "unsigned __int64";
	  break;
	case CV_INT64:
	  base = "int64_t";
	  break;
	case CV_UINT64:
	  base = "uint64_t";
	  break;
	case CV_FLOAT32:
	  base = "float";
	  break;
	case CV_FLOAT64:
	  base = "double";
	  break;
	case CV_FLOAT80:
	  base = "long double";
	  break;
	case CV_BOOL8:
	  base = "bool";
	  break;
	default:
	  return "";
	}

      if (mode == CV_TM_DIRECT)
	return base;
      return string_printf ("%s*", base);
    }

  if (ti >= tpi->type_idx_begin && ti < tpi->type_idx_end)
    {
      const char *name
	= pdb_tpi_record_name (&tpi->types[ti - tpi->type_idx_begin]);
      if (name != nullptr)
	return name;
    }

  return "";
}

/* Print a type index followed by its resolved name in parentheses.  */

static void
pdb_print_ti (const pdb_tpi_context *tpi, const char *label,
	      uint32_t ti)
{
  std::string name = pdb_type_index_name (tpi, ti);
  if (!name.empty ())
    gdb_printf ("  %s=0x%04x(%s)", label, ti, name.c_str ());
  else
    gdb_printf ("  %s=0x%04x", label, ti);
}

/* Return the leaf that STR names, such as "LF_STRUCTURE" in any case, or
   0 when STR names none.  */

static uint16_t
pdb_parse_leaf_kind (const char *str)
{
  for (const pdb_leaf_kind &k : pdb_leaf_kinds)
    if (strcasecmp (str, k.name) == 0)
      return k.leaf;
  return 0;
}

/* Print referenced type indices for a TPI record.  Each record type stores
   different cross-references; show the most useful ones.  Resolves type
   indices to names.  */

static void
pdb_print_type_refs (const pdb_tpi_context *tpi,
		     const pdb_tpi_type *rec)
{
  switch (rec->leaf)
    {
    case LF_MODIFIER:
      if (rec->data_len >= LF_MOD_SIZE)
	pdb_print_ti (tpi, "type", read_u32 (rec->data + LF_MOD_TYPE_OFFS));
      break;

    case LF_POINTER:
      if (rec->data_len >= LF_POINTER_SIZE)
	{
	  pdb_print_ti (tpi, "utype",
			read_u32 (rec->data + LF_POINTER_UTYPE_OFFS));

	  /* Decode the pointer mode.  */
	  uint32_t attr = read_u32 (rec->data + LF_POINTER_ATTR_OFFS);
	  uint32_t mode = cv_ptr_mode (attr);
	  gdb_printf ("  mode=%s", cv_ptr_mode_name (mode));

	  if (cv_ptr_is_member (mode)
	      && rec->data_len >= LF_POINTER_PMCLASS_OFFS + 4)
	    pdb_print_ti (tpi, "pmclass",
			  read_u32 (rec->data + LF_POINTER_PMCLASS_OFFS));
	}
      break;

    case LF_ARRAY:
      if (rec->data_len >= LF_ARRAY_MIN_SIZE)
	{
	  pdb_print_ti (tpi, "elem",
			read_u32 (rec->data + LF_ARRAY_ELEMTYPE_OFFS));
	  pdb_print_ti (tpi, "idx",
			read_u32 (rec->data + LF_ARRAY_IDXTYPE_OFFS));

	  /* Decode the trailing numeric leaf: total array size in bytes.  */
	  if (rec->data_len > LF_ARRAY_DATA_OFFS)
	    {
	      uint64_t bytes = 0;
	      uint32_t nr
		= pdb_cv_read_numeric (rec->data + LF_ARRAY_DATA_OFFS,
				       rec->data_len - LF_ARRAY_DATA_OFFS,
				       &bytes);
	      if (nr > 0)
		gdb_printf ("  size=%" PRIu64, bytes);
	    }
	}

      break;

    case LF_PROCEDURE:
      if (rec->data_len >= LF_PROC_SIZE)
	{
	  pdb_print_ti (tpi, "ret",
			read_u32 (rec->data + LF_PROC_RVTYPE_OFFS));
	  pdb_print_ti (tpi, "args",
			read_u32 (rec->data + LF_PROC_ARGLIST_OFFS));
	}

      break;

    case LF_MFUNCTION:
      if (rec->data_len >= LF_MFUNC_SIZE)
	{
	  pdb_print_ti (tpi, "ret",
			read_u32 (rec->data + LF_MFUNC_RVTYPE_OFFS));
	  pdb_print_ti (tpi, "class",
			read_u32 (rec->data + LF_MFUNC_CLASSTYPE_OFFS));
	  pdb_print_ti (tpi, "this",
			read_u32 (rec->data + LF_MFUNC_THISTYPE_OFFS));
	  pdb_print_ti (tpi, "args",
			read_u32 (rec->data + LF_MFUNC_ARGLIST_OFFS));
	}

      break;

    case LF_CLASS:
    case LF_STRUCTURE:
      if (rec->data_len >= LF_STRUCT_MIN_SIZE)
	{
	  uint16_t prop = read_u16 (rec->data + LF_STRUCT_PROPERTY_OFFS);
	  gdb_printf ("  fields=0x%04x",
		      read_u32 (rec->data + LF_STRUCT_FIELDLIST_OFFS));
	  if ((prop & CV_PROP_FWDREF) != 0)
	    gdb_printf (" [fwdref]");
	}

      break;

    case LF_UNION:
      if (rec->data_len >= LF_UNION_MIN_SIZE)
	{
	  uint16_t prop = read_u16 (rec->data + LF_UNION_PROPERTY_OFFS);
	  gdb_printf ("  fields=0x%04x",
		      read_u32 (rec->data + LF_UNION_FIELDLIST_OFFS));
	  if ((prop & CV_PROP_FWDREF) != 0)
	    gdb_printf (" [fwdref]");
	}

      break;

    case LF_ENUM:
      if (rec->data_len >= LF_ENUM_MIN_SIZE)
	{
	  uint16_t prop = read_u16 (rec->data + LF_ENUM_PROPERTY_OFFS);
	  pdb_print_ti (tpi, "utype",
			read_u32 (rec->data + LF_ENUM_UTYPE_OFFS));
	  gdb_printf ("  fields=0x%04x",
		      read_u32 (rec->data + LF_ENUM_FIELDLIST_OFFS));
	  if ((prop & CV_PROP_FWDREF) != 0)
	    gdb_printf (" [fwdref]");
	}

      break;

    case LF_ARGLIST:
      if (rec->data_len >= LF_ARGLIST_MIN_SIZE)
	{
	  uint32_t count = read_u32 (rec->data + LF_ARGLIST_COUNT_OFFS);
	  gdb_printf ("  count=%u", count);
	  /* Show individual argument types.  */
	  for (uint32_t i = 0;
	       i < count
	       && LF_ARGLIST_ARGS_OFFS + (i + 1) * 4 <= rec->data_len;
	       i++)
	    pdb_print_ti (tpi, "arg",
			  read_u32 (rec->data + LF_ARGLIST_ARGS_OFFS + i * 4));
	}

      break;

    case LF_BITFIELD:
      if (rec->data_len >= LF_BITFIELD_SIZE)
	{
	  pdb_print_ti (tpi, "type",
			read_u32 (rec->data + LF_BITFIELD_TYPE_OFFS));
	  gdb_printf ("  bits=%u pos=%u",
		      read_u8 (rec->data + LF_BITFIELD_LENGTH_OFFS),
		      read_u8 (rec->data + LF_BITFIELD_POSITION_OFFS));
	}

      break;

    default:
      break;
    }
}

/* 'maint info pdb-types' command: dump TPI type records with optional
   kind= filter.  */

static void
pdb_info_types_command (const char *args, int from_tty)
{
  std::string kind_str;

  validate_args (args, { "path", "kind" });

  const pdb_per_objfile *pdb = pdb_select_pdb (args);

  uint16_t kind_filter = 0;
  if (get_info_arg (args, "kind", &kind_str))
    {
      kind_filter = pdb_parse_leaf_kind (kind_str.c_str ());
      if (kind_filter == 0)
	error (_("Unknown leaf kind: %s"), kind_str.c_str ());
    }

  const pdb_tpi_context *tpi = &pdb->tpi;

  gdb_printf ("PDB '%s'\n", !pdb->pdb_file_path.empty ()
			      ? pdb->pdb_file_path.c_str ()
			      : "<unknown>");
  gdb_printf ("  TPI range: 0x%04x .. 0x%04x (%u types)\n",
	      tpi->type_idx_begin, tpi->type_idx_end,
	      tpi->type_idx_end - tpi->type_idx_begin);

  uint32_t count = 0;

  for (uint32_t ti = tpi->type_idx_begin; ti < tpi->type_idx_end; ti++)
    {
      QUIT;
      uint32_t idx = ti - tpi->type_idx_begin;
      const pdb_tpi_type *rec = &tpi->types[idx];

      if (kind_filter != 0 && rec->leaf != kind_filter)
	continue;

      std::string kind_name = pdb_leaf_type_name (rec->leaf);
      const char *rec_name = pdb_tpi_record_name (rec);

      gdb_printf ("  0x%04x: %-14s  len=%-4u", ti, kind_name.c_str (),
		  rec->data_len + 2);

      pdb_print_type_refs (tpi, rec);

      if (rec_name != nullptr && rec_name[0] != '\0')
	gdb_printf ("  `%s`", rec_name);

      gdb_printf ("\n");
      count++;
    }

  gdb_printf ("  %u type records%s.\n", count,
	      kind_filter != 0 ? " (filtered)" : "");
}

/* 'maint info pdb-streams' command.  */

static void
pdb_info_streams_command (const char *args, int from_tty)
{
  validate_args (args, { "path", "blocks" });

  const pdb_per_objfile *pdb = pdb_select_pdb (args);

  bool blocks = false;
  if (std::string value; get_info_arg (args, "blocks", &value))
    {
      std::string lower = value;
      for (char &c : lower)
	c = TOLOWER (c);
      int on = parse_cli_boolean_value (lower.c_str ());
      if (on < 0)
	error (_("\"blocks\" takes \"on\" or \"off\", not \"%s\"."),
	       value.c_str ());
      blocks = on != 0;
    }

  pdb_print_streams (pdb, blocks);
}

#if GDB_SELF_TEST

namespace selftests
{

/* Test the argument helpers.  Check that get_info_arg finds KEY=VALUE
   with KEY in any case and reads every word, also past the eleventh.
   Check that validate_args rejects an unknown key and a word without
   '=', and that string_to_int accepts only plain decimal digits.  */

static void
test_cmd_args ()
{
  std::string value;
  SELF_CHECK (get_info_arg ("PATH=a.pdb modi=3", "path", &value)
	      && value == "a.pdb");
  SELF_CHECK (get_info_arg ("path=a.pdb Modi=3", "modi", &value)
	      && value == "3");
  SELF_CHECK (!get_info_arg ("path=a.pdb", "modi", &value));
  SELF_CHECK (!get_info_arg ("modifier=1", "modi", &value));
  SELF_CHECK (!get_info_arg (nullptr, "modi", &value));

  /* Words past the eleventh are read too.  */
  std::string many;
  for (int k = 0; k < 12; k++)
    many += "path=x ";
  many += "modi=7";
  SELF_CHECK (get_info_arg (many.c_str (), "modi", &value) && value == "7");

  auto validate = [] (const char *args)
    {
      return error_of ([&] { validate_args (args, { "path", "modi" }); });
    };
  SELF_CHECK (validate (nullptr).empty ());
  SELF_CHECK (validate ("PATH=a modi=1").empty ());
  SELF_CHECK (validate ("blocks=on")
	      == "Unknown argument \"blocks=on\"; the arguments are path=,"
		 " modi=.");
  SELF_CHECK (!validate ("modi").empty ());
  SELF_CHECK (!validate ("path=a junk").empty ());

  unsigned long n;
  SELF_CHECK (string_to_int ("42", &n) && n == 42);
  SELF_CHECK (!string_to_int ("", &n));
  SELF_CHECK (!string_to_int ("-1", &n));
  SELF_CHECK (!string_to_int ("+1", &n));
  SELF_CHECK (!string_to_int (" 1", &n));
  SELF_CHECK (!string_to_int ("1x", &n));
  SELF_CHECK (!string_to_int ("4294967296", &n));
}

/* Test pdb_read_module_files_c13 on a C13 checksum table built in
   memory.  The table has an MD5 entry, an entry without a checksum, an
   entry whose name is not in /names, and a last entry cut short by the
   end of the table.  Check that all four are read, that the cut-short
   entry keeps no checksum, and that a second read adds nothing.  */

static void
test_files_c13 ()
{
  static const char names[] = "\0a.cpp\0b.cpp";

  gdb::byte_vector table;
  auto entry = [&] (uint32_t name, uint8_t kind, uint8_t size, size_t have)
    {
      size_t off = table.size ();
      put_u32 (table, off, name);
      table.push_back (size);
      table.push_back (kind);
      for (size_t k = 0; k < have; k++)
	table.push_back (k);
      if (have == size)
	table.resize (align_up (table.size (), (size_t) 4), 0);
    };
  entry (1, CHKSUM_TYPE_MD5, 16, 16);
  entry (7, CHKSUM_TYPE_NONE, 0, 0);
  entry (100, CHKSUM_TYPE_NONE, 0, 0);
  entry (1, CHKSUM_TYPE_SHA_256, 32, 10);

  test_pdb file;
  file.streams.resize (5);
  file.build ();
  test_module_reader r (file, 1);
  if (r.pdb == nullptr)
    return;
  r.pdb->string_table = (const gdb_byte *) names;
  r.pdb->string_table_size = sizeof names;

  pdb_module_info mod;
  mod.file_checksums = table.data ();
  mod.file_checksums_size = table.size ();
  pdb_read_module_files_c13 (r.pdb, &mod);
  SELF_CHECK (mod.num_files_c13 == 4);
  if (mod.num_files_c13 != 4)
    return;

  const pdb_file_info *f = mod.files_c13;
  SELF_CHECK (strcmp (f[0].filename, "a.cpp") == 0
	      && f[0].checksum_type == CHKSUM_TYPE_MD5
	      && f[0].checksum_size == 16 && f[0].checksum[15] == 15);
  SELF_CHECK (strcmp (f[1].filename, "b.cpp") == 0
	      && f[1].checksum_size == 0 && f[1].checksum == nullptr);
  SELF_CHECK (f[2].filename == nullptr);
  SELF_CHECK (strcmp (f[3].filename, "a.cpp") == 0
	      && f[3].checksum_type == CHKSUM_TYPE_SHA_256
	      && f[3].checksum_size == 0 && f[3].checksum == nullptr);

  pdb_read_module_files_c13 (r.pdb, &mod);
  SELF_CHECK (mod.num_files_c13 == 4);
}

/* Test the maint info commands on a PDB built in memory and selected by
   path=.  The PDB has one module with a.cpp's lines for the 0x20 bytes
   at 1:0x10.  Check what pdb-lines, pdb-files-c13, pdb-modules,
   pdb-files and pdb-streams print, and check the errors for a bad
   argument, a PDB that is not loaded and a module index out of
   range.  */

static void
test_maint_commands ()
{
  test_module m;
  uint32_t a = m.file (1);
  m.lines (1, 0x10, 0x20, a, { { 0, 10 }, { 8, 11 } });

  pdb_module_info info;
  info.module_name = "a.obj";
  info.obj_file_name = "a.obj";
  gdb::byte_vector stream = m.stream (&info);
  test_read_modules ({ info }, { stream }, test_tpi (), nullptr,
		     [&] (pdb_per_objfile *pdb)
    {
      pdb->pdb_file_path = "pdb-selftest.pdb";

      auto run = [] (const std::string &command)
	{
	  std::string out;
	  execute_command_to_string (out, command.c_str (), 0, false);
	  return out;
	};
      auto has = [] (const std::string &out, const std::string &text)
	{
	  return out.find (text) != std::string::npos;
	};
      const std::string path = " path=pdb-selftest.pdb";

      std::string out = run ("maint info pdb-lines" + path);
      SELF_CHECK (has (out, "a.cpp"));
      SELF_CHECK (has (out, "0001:00000010-00000030, line/addr entries = 2"));
      SELF_CHECK (has (out, "10 00000010    11 00000018"));

      out = run ("maint info pdb-files-c13" + path + " modi=0");
      SELF_CHECK (has (out, "PDB 'pdb-selftest.pdb'")
		  && has (out, "Mod 0000") && has (out, "a.cpp"));

      out = run ("maint info pdb-modules" + path + " MODI=0");
      std::string c13 = "C13 bytes: " + std::to_string (info.c13_byte_size);
      SELF_CHECK (has (out, "Mod 0000") && has (out, c13));

      out = run ("maint info pdb-files" + path);
      SELF_CHECK (has (out, "Mod 0000"));

      out = run ("maint info pdb-streams" + path);
      SELF_CHECK (has (out, "Streams: ") && !has (out, "blocks: "));
      SELF_CHECK (has (run ("maint info pdb-streams" + path + " blocks=ON"),
		       "blocks: "));
      SELF_CHECK (!has (run ("maint info pdb-streams" + path + " blocks=off"),
			"blocks: "));

      auto fails = [&] (const std::string &command)
	{
	  return error_of ([&] { run (command); });
	};
      SELF_CHECK (fails ("maint info pdb-lines" + path + " modi=1")
		  == "Module index 1 out of range: the PDB has 1 modules.");
      SELF_CHECK (fails ("maint info pdb-lines" + path + " modi=-1")
		  == "Invalid module index: -1");
      SELF_CHECK (fails ("maint info pdb-modules path=other.pdb")
		  == "PDB not loaded: other.pdb");
      SELF_CHECK (has (fails ("maint info pdb-files" + path + " junk=1"),
		       "Unknown argument \"junk=1\""));
      SELF_CHECK (has (fails ("maint info pdb-streams" + path
			      + " blocks=maybe"),
		       "\"blocks\" takes \"on\" or \"off\""));
    });
}

/* Test "maint info pdb-types".  Build a TPI with one record of each kind
   the command describes, plus forward references for a structure, a
   union and a class.  Check that each record's line shows the type
   indices it refers to and its name, that an argument list's simple
   types are printed by name, and that the TPI range is printed.  Check
   that kind= shows only records of that kind, and that an unknown kind
   is an error.  */

static void
test_types_command ()
{
  test_tpi tpi;
  uint32_t args
    = tpi.arglist ({ 0x74, 0x75, 0x12, 0x22, 0x13, 0x23, 0x40, 0x41, 0x30 });
  tpi.procedure (3, args, 9);

  gdb::byte_vector body;
  put_u32 (body, 0, 0x74);
  body.push_back (CV_MODIFIER_CONST);
  body.push_back (0);
  tpi.add (LF_MODIFIER, body);
  tpi.pointer (0x70);

  body.clear ();
  put_u32 (body, 0, 0x74);
  put_u32 (body, 4, 0x23);
  for (gdb_byte b : { 12, 0, 0 })
    body.push_back (b);
  tpi.add (LF_ARRAY, body);

  body.clear ();
  put_u32 (body, 0, 0x75);
  body.push_back (3);
  body.push_back (2);
  tpi.add (LF_BITFIELD, body);

  tpi.add (LF_VTSHAPE, gdb::byte_vector (2, 0));
  uint32_t fields = tpi.add (LF_FIELDLIST, {});
  tpi.add (LF_METHODLIST, {});

  /* A tag: COUNT, PROPERTY, the field list, then the derivation list and
     the vtable shape but for a union, a size of 4 and NAME.  */
  auto tag = [&] (uint16_t leaf, uint16_t property, uint32_t field_list,
		  const char *name)
    {
      gdb::byte_vector b (4, 0);
      b[2] = property & 0xff;
      b[3] = property >> 8;
      put_u32 (b, 4, field_list);
      if (leaf != LF_UNION)
	{
	  put_u32 (b, 8, 0);
	  put_u32 (b, 12, 0);
	}
      b.push_back (4);
      b.push_back (0);
      b.insert (b.end (), name, name + strlen (name) + 1);
      return tpi.add (leaf, b);
    };
  uint32_t agg = tag (LF_STRUCTURE, 0, fields, "Agg");
  tag (LF_UNION, 0, fields, "UAgg");
  tag (LF_CLASS, 0, fields, "Cls");
  for (uint16_t leaf : { LF_STRUCTURE, LF_UNION, LF_CLASS })
    tag (leaf, CV_PROP_FWDREF, 0,
	 leaf == LF_STRUCTURE ? "Agg" : leaf == LF_UNION ? "UAgg" : "Cls");

  body.clear ();
  put_u32 (body, 0, 0x74);
  put_u32 (body, 4, 0x74);
  put_u32 (body, 8, fields);
  body.insert (body.end (), { 'E', 'A', 'g', 'g', 0 });
  tpi.add (LF_ENUM, body);

  body.clear ();
  put_u32 (body, LF_MFUNC_RVTYPE_OFFS, 0x74);
  put_u32 (body, LF_MFUNC_CLASSTYPE_OFFS, agg);
  put_u32 (body, LF_MFUNC_THISTYPE_OFFS, 0);
  put_u32 (body, 12, 0);
  put_u32 (body, LF_MFUNC_ARGLIST_OFFS, args);
  put_u32 (body, 20, 0);
  tpi.add (LF_MFUNCTION, body);

  test_read_modules ({}, {}, tpi, nullptr, [&] (pdb_per_objfile *pdb)
    {
      pdb->pdb_file_path = "pdb-selftest.pdb";
      auto run = [] (const std::string &command)
	{
	  std::string out;
	  execute_command_to_string (out, command.c_str (), 0, false);
	  return out;
	};
      const std::string cmd
	= "maint info pdb-types path=pdb-selftest.pdb";

      /* The line of the first record of KIND in OUT.  */
      auto line_of = [] (const std::string &out, const std::string &kind)
	{
	  size_t at = out.find (" " + kind + " ");
	  if (at == std::string::npos)
	    return std::string ();
	  size_t start = out.rfind ('\n', at) + 1;
	  return out.substr (start, out.find ('\n', at) - start);
	};
      auto has = [] (const std::string &line, const char *text)
	{
	  return line.find (text) != std::string::npos;
	};

      std::string out = run (cmd);
      SELF_CHECK (has (line_of (out, "LF_ARGLIST"), "count=9")
		  && has (line_of (out, "LF_ARGLIST"), "arg=0x0074(int)"));
      SELF_CHECK (has (line_of (out, "LF_PROCEDURE"), "ret=0x0003(void)")
		  && has (line_of (out, "LF_PROCEDURE"), "args=0x1000"));
      SELF_CHECK (has (line_of (out, "LF_MODIFIER"), "type=0x0074(int)"));
      SELF_CHECK (has (line_of (out, "LF_POINTER"), "utype=0x0070(char)")
		  && has (line_of (out, "LF_POINTER"), "mode=ptr"));
      SELF_CHECK (has (line_of (out, "LF_ARRAY"), "elem=0x0074(int)")
		  && has (line_of (out, "LF_ARRAY"), "size=12"));
      SELF_CHECK (has (line_of (out, "LF_BITFIELD"),
		       "type=0x0075(unsigned int)")
		  && has (line_of (out, "LF_BITFIELD"), "bits=3 pos=2"));
      SELF_CHECK (has (line_of (out, "LF_VTSHAPE"), "len=")
		  && has (line_of (out, "LF_FIELDLIST"), "len=")
		  && has (line_of (out, "LF_METHODLIST"), "len="));
      SELF_CHECK (has (line_of (out, "LF_MFUNCTION"), "ret=0x0074(int)")
		  && has (line_of (out, "LF_MFUNCTION"), "args=0x1000"));
      std::string fields_text = string_printf ("fields=0x%04x", fields);
      SELF_CHECK (has (line_of (out, "LF_STRUCTURE"), fields_text.c_str ())
		  && has (line_of (out, "LF_STRUCTURE"), "`Agg`"));
      SELF_CHECK (has (line_of (out, "LF_UNION"), "`UAgg`")
		  && has (line_of (out, "LF_CLASS"), "`Cls`"));
      SELF_CHECK (has (line_of (out, "LF_ENUM"), "utype=0x0074(int)")
		  && has (line_of (out, "LF_ENUM"), "`EAgg`"));
      SELF_CHECK (has (out, "fields=0x0000 [fwdref]  `Agg`")
		  && has (out, "fields=0x0000 [fwdref]  `UAgg`")
		  && has (out, "fields=0x0000 [fwdref]  `Cls`"));
      for (const char *name : { "(int)", "(unsigned int)", "(long)",
				"(unsigned long)", "(__int64)",
				"(unsigned __int64)", "(float)", "(double)",
				"(bool)" })
	SELF_CHECK (has (line_of (out, "LF_ARGLIST"), name));
      SELF_CHECK (has (out, "TPI range: 0x1000 .. 0x"));

      /* Each kind= filter keeps its records and no other kind's.  */
      for (const pdb_leaf_kind &k : pdb_leaf_kinds)
	{
	  std::string filtered
	    = run (cmd + " kind=" + std::string (k.name));
	  for (const pdb_leaf_kind &other : pdb_leaf_kinds)
	    if (other.leaf != k.leaf)
	      SELF_CHECK (line_of (filtered, other.name).empty ());
	}
      SELF_CHECK (!line_of (run (cmd + " kind=lf_array"), "LF_ARRAY").empty ());

      SELF_CHECK (error_of ([&] { run (cmd + " kind=LF_BOGUS"); })
		  == "Unknown leaf kind: LF_BOGUS");
    });
}

} /* namespace selftests */

#endif /* GDB_SELF_TEST */

} /* namespace pdb */

/* Register PDB maintenance info commands.  */
INIT_GDB_FILE (pdb_info)
{
  using namespace pdb;

#if GDB_SELF_TEST
  ::selftests::register_test ("pdb-cmd-args",
			      pdb::selftests::test_cmd_args);
  ::selftests::register_test ("pdb-files-c13",
			      pdb::selftests::test_files_c13);
  ::selftests::register_test ("pdb-maint-commands",
			      pdb::selftests::test_maint_commands);
  ::selftests::register_test ("pdb-types-command",
			      pdb::selftests::test_types_command);
#endif

  add_cmd ("pdb-loaded-files", class_maintenance,
	   pdb_info_loaded_files_command,
	   _("List currently loaded PDB file paths."), &maintenanceinfolist);

  add_cmd ("pdb-modules", class_maintenance, pdb_info_modules_command,
	   _("List PDB modules.\n\
Usage: maintenance info pdb-modules [path=<pdb-path>] [modi=N]\n\
If path is omitted, uses the default (main program) PDB.\n\
If modi is omitted, dumps all modules."),
	   &maintenanceinfolist);

  add_cmd ("pdb-files", class_maintenance, pdb_info_files_command,
	   _("List source files in PDB modules (from DBI File Info).\n\
Usage: maintenance info pdb-files [path=<pdb-path>] [modi=N]\n\
If path is omitted, uses the default (main program) PDB.\n\
If modi is omitted, dumps all modules."),
	   &maintenanceinfolist);

  add_cmd ("pdb-files-c13", class_maintenance, pdb_info_files_c13_command,
	   _("List source files in PDB modules with checksums\n\
(from module streams).\n\
Usage: maintenance info pdb-files-c13 [path=<pdb-path>] [modi=N]\n\
If path is omitted, uses the default (main program) PDB.\n\
If modi is omitted, dumps all modules."),
	   &maintenanceinfolist);

  add_cmd ("pdb-lines", class_maintenance, pdb_info_lines_command,
	   _("List line/address entries for files in PDB modules.\n\
Usage: maintenance info pdb-lines [path=<pdb-path>] [modi=N]\n\
If path is omitted, uses the default (main program) PDB.\n\
If modi is omitted, dumps all modules."),
	   &maintenanceinfolist);

  add_cmd ("pdb-types", class_maintenance, pdb_info_types_command,
	   _("Dump TPI type records from the PDB.\n\
Usage: maintenance info pdb-types [path=<pdb-path>] [kind=LF_xxx]\n\
If path is omitted, uses the default (main program) PDB.\n\
kind filters by leaf type (e.g. LF_STRUCTURE, LF_ENUM)."),
	   &maintenanceinfolist);

  add_cmd ("pdb-streams", class_maintenance, pdb_info_streams_command,
	   _("Report the MSF layout of the PDB.\n\
Prints the block geometry, the PDB Info stream header, the named\n\
streams, the DBI symbol stream numbers and the size of every stream.\n\
Usage: maintenance info pdb-streams [path=<pdb-path>] [blocks=on]\n\
If path is omitted, uses the default (main program) PDB.\n\
blocks=on also lists the MSF blocks of each stream."),
	   &maintenanceinfolist);
}
