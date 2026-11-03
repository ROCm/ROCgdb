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

/* The whitespace-separated words of ARGS.  */

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

/* Set *OUT_VALUE to the value of the KEY=VALUE argument of ARGS, KEY in
   any case, and return true; return false when there is none.  */

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

/* Throw an error unless each argument of ARGS is KEY=VALUE for a KEY,
   in any case, of VALID_KEYS.  */

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

/* Set *OUT_VALUE to the decimal number STR, of at most 9 digits and
   nothing else, and return true; return false for anything else.  */

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

/* The loaded PDB whose path is PATH.  The commands accept only the PDBs
   of the loaded objfiles; throw an error for another path.  */

static pdb_per_objfile *
pdb_require_loaded_path (const std::string &path)
{
  pdb_per_objfile *pdb = pdb_find_loaded_pdb (path.c_str ());
  if (pdb == nullptr)
    error (_("PDB not loaded: %s"), path.c_str ());
  return pdb;
}

/* The PDB that the path= argument of ARGS names, else the main
   program's.  Throw an error when there is none.  */

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

/* The name of the checksum kind TYPE, a CHKSUM_TYPE_* value.  */

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

/* Parse the path= and modi= arguments of ARGS, for the commands that print
   module data, and also accept EXTRA_KEYS.  Set *OUT_PDB to the PDB
   path= names, else the main program's, and [*OUT_MOD_START,
   *OUT_MOD_END) to module modi= names, else all modules.  Throw an error
   for an argument that is not one of those, a PDB that is not loaded or
   a module that is not there.  */

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

/* Callback for pdb_walk_c13_line_blocks: print the line entries of a
   block, for "maint info pdb-lines".  */

static void
pdb_dump_line_block (const char *filename, CV_LineSection ls,
		     const gdb_byte *lines, uint32_t num_lines)
{
  QUIT;
  /* The range the block and its lines cover.  The file's offsets are
     added in 64 bits, as they can add up past 32.  */
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

/* The argument helpers: KEY=VALUE words with keys in any case, every
   word read, and an unknown key or a word without '=' an error; a module
   index of digits only.  */

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

/* The source files of a module's C13 checksum table: an MD5 entry, an
   entry without a checksum, one whose name is not in /names, and a last
   one cut short by the end of the table, kept without its checksum.  A
   second read adds nothing.  */

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

/* The commands on a PDB of one module, a.cpp's lines for the 0x20 bytes
   at 1:0x10, selected by path=: what they print, and the errors for a
   bad argument, a PDB that is not loaded and a module that is not
   there.  */

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
