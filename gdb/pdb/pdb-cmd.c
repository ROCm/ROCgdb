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

  add_cmd ("pdb-streams", class_maintenance, pdb_info_streams_command,
	   _("Report the MSF layout of the PDB.\n\
Prints the block geometry, the PDB Info stream header, the named\n\
streams, the DBI symbol stream numbers and the size of every stream.\n\
Usage: maintenance info pdb-streams [path=<pdb-path>] [blocks=on]\n\
If path is omitted, uses the default (main program) PDB.\n\
blocks=on also lists the MSF blocks of each stream."),
	   &maintenanceinfolist);
}
