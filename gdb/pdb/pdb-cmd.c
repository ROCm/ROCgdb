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

This file implements various PDB related GDB info commands.  */

#include "cli/cli-utils.h"
#include "cli/cli-cmds.h"
#include "objfiles.h"
#include "pdb/pdb-internal.h"
#include "gdbsupport/rsp-low.h"
#include "block.h"
#include "gdbarch.h"
#include "value.h"
#include "progspace.h"

#include <string.h>
#include <stdint.h>
#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>

namespace pdb
{

/* Split args string into tokens.  */
static std::vector<std::string>
get_arg_tokens (const char *args)
{
  size_t sanity_cnt = 0;
  std::string arg;
  std::vector<std::string> tokens;
  do
    {
      arg = extract_arg (&args);
      if (arg.empty () || sanity_cnt > 10)
	return tokens;
      sanity_cnt++;
      tokens.push_back (arg);
    }

  while (true);
  return tokens;
}

/* Extract the value of a key=value argument from args and return
true if key is found, otherwise false.  */
static bool
get_info_arg (const char *args, const char *key, std::string *out_value)
{
  std::vector<std::string> tokens = get_arg_tokens (args);

  size_t key_len = strlen (key);

  for (const auto &token : tokens)
    {
      /* Check if this token starts with key=.  */
      if (token.size () > key_len && token[key_len] == '='
	  && strncasecmp (token.c_str (), key, key_len) == 0)
	{
	  *out_value = token.substr (key_len + 1);
	  return true;
	}
    }

  return false;
}

/* Validate that all arguments in args are recognized key=value pairs.  */
static bool
validate_args (const char *args, const std::vector<std::string> &valid_keys)
{
  std::vector<std::string> tokens = get_arg_tokens (args);

  for (const auto &token : tokens)
    {
      /* Find the = sign.  */
      size_t eq_pos = token.find ('=');
      if (eq_pos == std::string::npos)
	return false;

      std::string key = token.substr (0, eq_pos);

      /* Check if key is in valid_keys list.  */
      bool found = false;
      for (const auto &valid_key : valid_keys)
	{
	  if (strcasecmp (key.c_str (), valid_key.c_str ()) == 0)
	    {
	      found = true;
	      break;
	    }
	}

      if (!found)
	return false;
    }

  return true;
}

/* Convert string to unsigned long.  */
static bool
string_to_int (const std::string &str, unsigned long *out_value)
{
  if (str.empty ())
    return false;

  char *endptr;
  *out_value = strtoul (str.c_str (), &endptr, 10);

  /* Check if errors in conversion.  */
  if (*endptr != '\0' || endptr == str.c_str ())
    return false;

  return true;
}

/* Return loaded PDB instance for path, or NULL if not loaded.  */
static pdb_per_objfile *
pdb_find_loaded_pdb (const char *path)
{
  if (path == nullptr)
    return nullptr;

  for (objfile &objf : current_program_space->objfiles ())
    {
      pdb_per_objfile *pdb = pdb_get_per_objfile (&objf);
      if (pdb != nullptr && !pdb->pdb_file_path.empty ()
	  && pdb->pdb_file_path == path)
	return pdb;
    }

  return nullptr;
}

/* Return the default PDB (the one whose objfile has OBJF_MAINLINE),
   or NULL if none.  */
static pdb_per_objfile *
pdb_get_default (void)
{
  for (objfile &objf : current_program_space->objfiles ())
    {
      if ((objf.flags & OBJF_MAINLINE) == 0)
	continue;
      pdb_per_objfile *pdb = pdb_get_per_objfile (&objf);
      if (pdb != nullptr)
	return pdb;
    }

  return nullptr;
}

/* Check if pdb path is in the list of loaded PDBs.  Info commands will accept
PDB paths only for PDBs associated with the loaded object files.  */
static pdb_per_objfile *
pdb_require_loaded_path (const std::string &path)
{
  pdb_per_objfile *pdb = pdb_find_loaded_pdb (path.c_str ());
  if (pdb == nullptr)
    gdb_printf ("PDB not loaded: %s\n", path.c_str ());
  return pdb;
}

/* For info commands that print checksum type.  */
static const char *
pdb_checksum_type_name (uint8_t type)
{
  switch (type)
    {
    case 0:
      return "None";
    case 1:
      return "MD5";
    case 2:
      return "SHA-1";
    case 3:
      return "SHA-256";
    default:
      return "Unknown";
    }
}

/* Shared among info pdb-files commands.  Parse path= and modi= arguments and
prepare PDB context for commands.  Returns true on success with pdb and
mod_start/mod_end set which are used to mark the modules to be processed
(either all or just one).  */
static bool
prepare_info_command (const char *args, pdb_per_objfile **out_pdb,
		      uint32_t *out_mod_start, uint32_t *out_mod_end,
		      const std::vector<std::string> &extra_keys = {})
{
  std::string path;
  std::string modi_str;

  std::vector<std::string> valid = { "path", "modi" };
  valid.insert (valid.end (), extra_keys.begin (), extra_keys.end ());

  if (!validate_args (args, valid))
    {
      gdb_printf ("Usage: [path=<pdb-path>] [modi=N]\n");
      return false;
    }

  /* Get PDB: use path= if provided, else use default.  */
  pdb_per_objfile *pdb = pdb_get_default ();
  if (get_info_arg (args, "path", &path))
    {
      pdb = pdb_require_loaded_path (path);
      if (pdb == nullptr)
	return false;
    }

  if (pdb == nullptr)
    {
      gdb_printf ("No PDB loaded and no default PDB available.\n");
      return false;
    }

  *out_pdb = pdb;
  *out_mod_start = 0;
  *out_mod_end = pdb->num_modules;

  /* Parse modi= if provided.  */
  if (get_info_arg (args, "modi", &modi_str))
    {
      unsigned long modi;
      if (!string_to_int (modi_str, &modi))
	{
	  gdb_printf ("Invalid module index: %s\n", modi_str.c_str ());
	  return false;
	}

      /* Check if module index is in valid range.  */
      if (modi >= pdb->num_modules)
	{
	  gdb_printf ("Module index %lu out of range (0..%u).\n", modi,
		      pdb->num_modules - 1);
	  return false;
	}

      *out_mod_start = (uint32_t) modi;
      *out_mod_end = *out_mod_start + 1;
    }

  return true;
}

/* Callback for pdb_walk_c13_line_blocks: dump line entries to stdout
for 'info pdb-lines'.  CB_DATA is a pdb_line_block_info.  */
static void
pdb_dump_line_block (void *cb_data)
{
  pdb_line_block_info const *info = (pdb_line_block_info *) cb_data;
  CV_LineSection const *ls = info->line_sect;
  CV_Line const *lines = info->lines;
  uint32_t num_lines = info->num_lines;

  /* Compute address range.  */
  uint32_t range_start = ls->offs;
  uint32_t range_end = ls->offs + ls->code_sz;
  if (num_lines > 0)
    {
      range_start = ls->offs + lines[0].offs;
      range_end = ls->offs + lines[num_lines - 1].offs + 1;
      if (ls->offs + ls->code_sz > range_end)
	range_end = ls->offs + ls->code_sz;
    }

  /* Print file header.  */
  gdb_printf ("    %s\n", info->filename);

  gdb_printf ("      %04X:%08X-%08X, line/addr entries = %u\n",
	      (unsigned) ls->seci, range_start, range_end, num_lines);

  /* Print line/offset pairs, ~8 per row.  */
  gdb_printf ("        ");
  for (uint32_t j = 0; j < num_lines; j++)
    {
      uint32_t offs = ls->offs + lines[j].offs;
      uint32_t line = lines[j].line_start;
      gdb_printf ("%u %08X", line, offs);
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

/* 'info pdb-modules' command.  */
static void
pdb_info_modules_command (const char *args, int /*from_tty*/)
{
  std::vector<pdb_per_objfile *> pdbs_to_dump;
  std::string path = "";

  if (!validate_args (args, { "path" }))
    {
      gdb_printf ("Usage: info pdb-modules[path=<pdb-path>]\n");
      return;
    }

  /* If pdb path is provided it must point to one of the loaded PDBs.
  Otherwise, dump all loaded PDBs.  */
  if (get_info_arg (args, "path", &path))
    {
      pdb_per_objfile *pdb = pdb_require_loaded_path (path);
      if (pdb == nullptr)
	{
	  gdb_printf ("PDB path not found in loaded PDBs: %s\n",
		      path.c_str ());
	  return;
	}

      pdbs_to_dump.push_back (pdb);
    }

  else
    {
      for (objfile &objf : current_program_space->objfiles ())
	{
	  pdb_per_objfile *pdb = pdb_get_per_objfile (&objf);
	  if (pdb != nullptr)
	    pdbs_to_dump.push_back (pdb);
	}
    }

  if (pdbs_to_dump.empty ())
    {
      gdb_printf ("No PDB loaded.\n");
      return;
    }

  /* Dump module info for each selected PDB.  */
  for (pdb_per_objfile *pdb : pdbs_to_dump)
    {
      if (pdb == nullptr)
	continue;

      gdb_printf ("  PDB '%s'\n", !pdb->pdb_file_path.empty ()
				    ? pdb->pdb_file_path.c_str ()
				    : "<unknown>");

      for (uint32_t i = 0; i < pdb->num_modules; i++)
	{
	  pdb_module_info *mod = &pdb->modules[i];
	  pdb_read_module_files (pdb, mod);

	  gdb_printf ("  Mod %04u | `%s`:\n", i,
		      mod->module_name ? mod->module_name : "<unnamed>");
	  gdb_printf ("           Obj: `%s`:\n",
		      mod->obj_file_name ? mod->obj_file_name : "<none>");
	  gdb_printf ("           debug stream: %u, # files: %u\n",
		      (unsigned) mod->stream_number, mod->num_files);
	}
    }
}

/* 'info pdb-loaded-files' command.  */
static void
pdb_info_loaded_files_command (const char *args, int /*from_tty*/)
{
  if (args != nullptr && *args != '\0')
    {
      gdb_printf ("Usage: info pdb-loaded-files\n");
      return;
    }

  bool found = false;
  for (objfile &objf : current_program_space->objfiles ())
    {
      const pdb_per_objfile *pdb = pdb_get_per_objfile (&objf);
      if (pdb == nullptr || pdb->pdb_file_path.empty ())
	continue;
      gdb_printf ("%s\n", pdb->pdb_file_path.c_str ());
      found = true;
    }

  if (!found)
    gdb_printf ("No PDB loaded.\n");
}

/* 'info pdb-lines' command.  */
static void
pdb_info_lines_command (const char *args, int /*from_tty*/)
{
  pdb_per_objfile *pdb;
  uint32_t mod_start;
  uint32_t mod_end;

  if (!prepare_info_command (args, &pdb, &mod_start, &mod_end))
    return;

  for (uint32_t i = mod_start; i < mod_end; i++)
    {
      /* pdb_read_dbi_stream already allocated pdb->modules array.  */
      pdb_module_info *mod = &pdb->modules[i];
      auto stream_buf = pdb_read_stream (pdb, mod->stream_number);

      gdb_printf ("  Mod %04u | `%s`:\n", i,
		  mod->module_name ? mod->module_name : "<unnamed>");

      if (stream_buf != nullptr && mod->c13_byte_size > 0)
	{
	  pdb_line_block_info dcb;
	  pdb_walk_c13_line_blocks (pdb, mod, stream_buf.get (),
				    pdb_dump_line_block, &dcb);
	}
    }
}

/* 'info pdb-files' command.  */
static void
pdb_info_files_command (const char *args, int /*from_tty*/)
{
  pdb_per_objfile *pdb;
  uint32_t mod_start;
  uint32_t mod_end;

  if (!prepare_info_command (args, &pdb, &mod_start, &mod_end))
    return;

  gdb_printf ("  PDB '%s'\n", !pdb->pdb_file_path.empty ()
				? pdb->pdb_file_path.c_str ()
				: "<unknown>");

  for (auto i = mod_start; i < mod_end; i++)
    {
      pdb_module_info *mod = &pdb->modules[i];
      pdb_read_module_files (pdb, mod);

      gdb_printf ("  Mod %04u | `%s`:\n", i,
		  mod->module_name ? mod->module_name : "<unnamed>");

      for (auto j = 0; j < mod->num_files; j++)
	{
	  const char *fname = mod->files ? mod->files[j] : nullptr;
	  gdb_printf ("    - %s\n", fname ? fname : "<unknown>");
	}
    }
}

/* 'info pdb-files-c13' command.  */
static void
pdb_info_files_c13_command (const char *args, int /*from_tty*/)
{
  pdb_per_objfile *pdb;
  uint32_t mod_start;
  uint32_t mod_end;

  if (!prepare_info_command (args, &pdb, &mod_start, &mod_end))
    return;

  gdb_printf ("  PDB '%s'\n", !pdb->pdb_file_path.empty ()
				? pdb->pdb_file_path.c_str ()
				: "<unknown>");

  for (uint32_t i = mod_start; i < mod_end; i++)
    {
      pdb_module_info *mod = &pdb->modules[i];
      pdb_read_module_stream (pdb, mod);
      pdb_read_module_files_c13 (pdb, mod);

      gdb_printf ("  Mod %04u | `%s`:\n", i,
		  mod->module_name ? mod->module_name : "<unnamed>");

      for (uint32_t j = 0; j < mod->num_files_c13; j++)
	{
	  const pdb_file_info *fi = &mod->files_c13[j];

	  if (fi->checksum_size > 0 && fi->checksum != nullptr)
	    {
	      /* Print hash type & hash - also record the printed length
	      so that we can correctly format entries w/o hash.  */
	      std::string hash_str = bin2hex (fi->checksum, fi->checksum_size);

	      gdb_printf (" [%s: %s] ",
			  pdb_checksum_type_name (fi->checksum_type),
			  hash_str.c_str ());
	    }

	  gdb_printf ("%s", fi->filename ? fi->filename : "<unknown>");
	  gdb_printf ("\n");
	}
    }
}

/* Dump a single symbol record with an optional prefix string.
Prefix serves different callers as they dump the record along
with other information.  */
static void
pdb_dump_sym_record (pdb_per_objfile *pdb, gdb_byte *rec, uint16_t reclen,
		     uint16_t rectype, const std::string &prefix)
{
  gdb_byte *rec_data = rec + PDB_RECORD_DATA_OFFS;

  if (!prefix.empty ())
    gdb_printf ("%s", prefix.c_str ());

  pdb_dump_parse_record (pdb, rec_data, rectype, reclen, PDB_DUMP_SYM);
}

/* Parse and dump symbol records stream.
This function exist in dump command only as the real use for this stream
is to build the cooked index and fast symbol lookup. That's done differently
(not just the linear access like here) thus the function is in this file.  */
void
pdb_parse_sym_record_stream (pdb_per_objfile *pdb, uint32_t /*flags*/)
{
  gdb_byte *syms_data = pdb->sym_record_data;
  auto stream_size = pdb->sym_record_size;

  if (syms_data == nullptr || stream_size == 0)
    {
      pdb_warning ("No symbol record stream data available");
      return;
    }

  const gdb_byte *syms_end = syms_data + stream_size;
  gdb_byte *p = syms_data;
  uint32_t record_num = 0;

  /* We iterate by grabbing two values at a time: reclen and rectype. In each
  loop check if there is 2 values available.  */
  while (p + PDB_RECORD_HDR_SIZE <= syms_end)
    {
      uint16_t reclen;
      uint16_t rectype;
      size_t rec_size;
      if (!pdb_parse_sym_record_hdr (p, syms_end, reclen, rectype, rec_size))
	{
	  pdb_dbg_printf ("bad reclen=%u at offset %td in record stream",
			  reclen, (ptrdiff_t) (p - syms_data));
	  break;
	}

      std::string prefix;
      auto rec_name = pdb_sym_rec_type_name (rectype);

      std::ostringstream oss;
      oss << "    [" << record_num << "] " << std::hex << std::setfill ('0')
	  << std::setw (4) << rectype << std::dec << " " << std::setfill (' ')
	  << std::setw (18) << std::left << rec_name << " len=" << reclen;
      prefix = oss.str ();

      pdb_dump_sym_record (pdb, p, reclen, rectype, prefix);

      p += rec_size;
      record_num++;
    }
}

/* Dump hash records from Global Symbol stream.  HR_DATA points to the
start of the HashRecord array; HR_SIZE is its byte count.  Each
record's offset is resolved into the SymRecordStream for dumping.  */
static void
pdb_dump_gsi_hash_records (pdb_per_objfile *pdb, const gdb_byte *hr_data,
			   uint32_t hr_size)
{
  gdb_byte *syms_data = pdb->sym_record_data;
  auto syms_size = pdb->sym_record_size;
  if (syms_data == nullptr || syms_size == 0)
    {
      pdb_warning ("No symbol record stream data for resolving GSI records");
      return;
    }

  uint32_t num_records = hr_size / GSI_HASH_RECORD_SIZE;

  gdb_printf ("    %u hash records:\n", num_records);

  for (uint32_t i = 0; i < num_records; i++)
    {
      const gdb_byte *rec = hr_data + i * GSI_HASH_RECORD_SIZE;
      uint32_t offs = read_u32 (rec + GSI_HASH_RECORD_SYMOFFS_OFFS) - 1;

      /* Check if offset is within SymRecordStream bounds.  */
      if (offs >= syms_size || syms_size - offs < 4)
	{
	  gdb_printf ("    [%u] offset=%u (out of bounds)\n", i, offs);
	  continue;
	}

      /* Now we got the pointer into the symbol record stream - first, analyze
      the header.  */
      gdb_byte *data = syms_data + offs;
      const gdb_byte *syms_end = syms_data + syms_size;

      uint16_t reclen;
      uint16_t rectype;
      if (size_t rec_size; !pdb_parse_sym_record_hdr (data, syms_end, reclen,
						      rectype, rec_size))
	{
	  gdb_printf ("    [%u] failed to parse symbol record header\n", i);
	  continue;
	}

      auto rec_name = pdb_sym_rec_type_name (rectype);
      std::ostringstream oss;
      oss << "    [" << i << "] offset:" << std::setw (10) << std::left << offs
	  << " " << rec_name;
      std::string prefix = oss.str ();

      pdb_dump_sym_record (pdb, data, reclen, rectype, prefix);
    }
}

/* Dump GSI hash record header info.  */
static void
pdb_dump_gsi_hdr (const struct pdb_gsi_hdr &gsi)
{
  gdb_printf (
    "    GSI header: sig:0x%08X ver:0x%08X hr_bytes:%u bucket_bytes:%u\n",
    gsi.sig, gsi.ver, gsi.hr_bytes, gsi.bucket_bytes);
}

/* See pdb.h.  */
void
pdb_dump_gsi_stream (struct pdb_per_objfile *pdb)
{
  uint16_t stream_idx = pdb->gsi_stream;

  if (stream_idx == 0 || stream_idx == 0xFFFF)
    {
      pdb_warning ("No GSI stream available");
      return;
    }

  auto data_buf = pdb_read_stream (pdb, stream_idx);
  if (data_buf == nullptr)
    {
      pdb_warning ("Failed to read GSI stream %u", stream_idx);
      return;
    }

  gdb_byte *data = data_buf.get ();

  uint32_t stream_size = pdb->stream_sizes[stream_idx];
  struct pdb_gsi_hdr gsi = pdb_parse_gsi_hash (data, stream_size);
  if (!gsi.valid)
    {
      pdb_warning ("GSI stream too small or corrupt (%u bytes)", stream_size);
      return;
    }

  gdb_printf ("  GSI stream %u:\n", stream_idx);
  pdb_dump_gsi_hdr (gsi);

  pdb_dump_gsi_hash_records (pdb, gsi.hr_data, gsi.hr_bytes);
}

/* See pdb.h.  */
void
pdb_dump_psgsi_stream (struct pdb_per_objfile *pdb)
{
  uint16_t stream_idx = pdb->psgsi_stream;

  if (stream_idx == 0 || stream_idx == 0xFFFF)
    {
      pdb_warning ("No PSGSI stream available");
      return;
    }

  auto psi_buf = pdb_read_stream (pdb, stream_idx);
  if (psi_buf == nullptr)
    {
      pdb_warning ("Failed to read PSGSI stream %u", stream_idx);
      return;
    }

  gdb_byte *psi_data = psi_buf.get ();

  uint32_t stream_size = pdb->stream_sizes[stream_idx];
  if (stream_size < PSGSI_HDR_SIZE)
    {
      pdb_warning ("PSGSI stream too small (%u bytes)", stream_size);
      return;
    }

  auto sym_hash_size = read_u32 (psi_data + PSGSI_HDR_SYM_HASH_OFFS);
  auto addr_map_size = read_u32 (psi_data + PSGSI_HDR_ADDR_MAP_OFFS);

  gdb_printf ("  PSGSI stream %u: sym_hash_size=%u addr_map_size=%u\n",
	      stream_idx, sym_hash_size, addr_map_size);

  gdb_byte *gsi_data = psi_data + PSGSI_HDR_SIZE;

  if (sym_hash_size > stream_size - PSGSI_HDR_SIZE)
    {
      gdb_printf ("PSGSI hash data extends past stream end");
      return;
    }

  auto gsi = pdb_parse_gsi_hash (gsi_data, sym_hash_size);
  if (!gsi.valid)
    {
      pdb_warning ("PSGSI hash too small or corrupt");
      return;
    }

  pdb_dump_gsi_hdr (gsi);

  pdb_dump_gsi_hash_records (pdb, gsi.hr_data, gsi.hr_bytes);

  /* Address map: array of raw symbol-stream offsets, sorted by address.
  Unlike hash records (which store offset+1), address map entries are
  direct byte offsets into the SymRecordStream — no adjustment needed.  */
  gdb_byte *addr_map_data = gsi_data + sym_hash_size;
  auto addr_map_entries = addr_map_size / sizeof (uint32_t);
  gdb_printf ("    Address map: %zu entries (sorted by symbol address)\n",
	      addr_map_entries);
  for (uint32_t i = 0; i < addr_map_entries; i++)
    {
      auto sym_offs = read_u32 (addr_map_data + i * sizeof (uint32_t));

      if (sym_offs >= pdb->sym_record_size
	  || pdb->sym_record_size - sym_offs < 4)
	{
	  gdb_printf ("    [%u] offset=%u (out of bounds)\n", i, sym_offs);
	  continue;
	}

      gdb_byte *data = pdb->sym_record_data + sym_offs;
      const gdb_byte *syms_end = pdb->sym_record_data + pdb->sym_record_size;

      uint16_t reclen;
      uint16_t rectype;

      if (size_t rec_size; !pdb_parse_sym_record_hdr (data, syms_end, reclen,
						      rectype, rec_size))
	{
	  gdb_printf (
	    "    [%u] offset=%u (record extends past stream, reclen=%u)\n", i,
	    sym_offs, reclen);
	  continue;
	}

      auto rec_name = pdb_sym_rec_type_name (rectype);

      std::ostringstream oss;
      oss << "    [" << i << "] offset:" << std::setw (10) << std::left
	  << sym_offs << " " << rec_name;
      std::string prefix = oss.str ();

      pdb_dump_sym_record (pdb, data, reclen, rectype, prefix);
    }
}

/* 'info pdb-symbols' command.  */
static void
pdb_info_symbols_command (const char *args, int /*from_tty*/)
{
  struct pdb_per_objfile *pdb;
  uint32_t mod_start;
  uint32_t mod_end;

  if (!prepare_info_command (args, &pdb, &mod_start, &mod_end, { "stream" }))
    return;

  /* If stream= is given, dispatch to the appropriate stream dumper.  */
  if (std::string stream_str; get_info_arg (args, "stream", &stream_str))
    {
      gdb_printf ("  PDB '%s'\n", !pdb->pdb_file_path.empty ()
				    ? pdb->pdb_file_path.c_str ()
				    : "<unknown>");

      if (stream_str == "sym")
	pdb_parse_sym_record_stream (pdb, PDB_DUMP_SYM);
      else if (stream_str == "gsi")
	pdb_dump_gsi_stream (pdb);
      else if (stream_str == "psi")
	pdb_dump_psgsi_stream (pdb);
      else
	gdb_printf ("Unknown stream '%s' — use sym, gsi, or psi.\n",
		    stream_str.c_str ());
      return;
    }

  gdb_printf ("  PDB '%s'\n", !pdb->pdb_file_path.empty ()
				? pdb->pdb_file_path.c_str ()
				: "<unknown>");

  for (auto i = mod_start; i < mod_end; i++)
    {
      const struct pdb_module_info *mod = &pdb->modules[i];
      auto stream_buf = pdb_read_stream (pdb, mod->stream_number);

      gdb_printf ("  Mod %04u | `%s`:\n", i,
		  mod->module_name ? mod->module_name : "<unnamed>");

      if (stream_buf == nullptr || mod->sym_byte_size < 4)
	{
	  gdb_printf ("    (no symbols)\n");
	  continue;
	}

      pdb_parse_symbols (pdb, mod, stream_buf.get (), nullptr, PDB_DUMP_SYM);
    }
}

/* 'info pdb-sym-records' command — dump the SymRecordStream.  */
static void
pdb_info_sym_records_command (const char *args, int /*from_tty*/)
{
  struct pdb_per_objfile *pdb;
  uint32_t mod_start;
  if (uint32_t mod_end = 0;
      !prepare_info_command (args, &pdb, &mod_start, &mod_end))
    return;

  gdb_printf ("  PDB '%s'\n", !pdb->pdb_file_path.empty ()
				? pdb->pdb_file_path.c_str ()
				: "<unknown>");

  pdb_parse_sym_record_stream (pdb, PDB_DUMP_SYM);
}

/* 'info pdb-gsi' command — dump GSI (global symbol index) stream.  */
static void
pdb_info_gsi_command (const char *args, int /*from_tty*/)
{
  struct pdb_per_objfile *pdb;
  uint32_t mod_start;
  if (uint32_t mod_end = 0;
      !prepare_info_command (args, &pdb, &mod_start, &mod_end))
    return;

  gdb_printf ("  PDB '%s'\n", !pdb->pdb_file_path.empty ()
				? pdb->pdb_file_path.c_str ()
				: "<unknown>");

  pdb_dump_gsi_stream (pdb);
}

/* 'info pdb-psi' command — dump PSGSI (public symbol index) stream.  */
static void
pdb_info_psi_command (const char *args, int /*from_tty*/)
{
  struct pdb_per_objfile *pdb;
  uint32_t mod_start;
  uint32_t mod_end;

  if (!prepare_info_command (args, &pdb, &mod_start, &mod_end))
    return;

  gdb_printf ("  PDB '%s'\n", !pdb->pdb_file_path.empty ()
				? pdb->pdb_file_path.c_str ()
				: "<unknown>");

  pdb_dump_psgsi_stream (pdb);
}

/* See pdb.h.  */
void
pdb_dump_locations (struct compunit_symtab *cust, struct gdbarch *gdbarch,
		    const char *symbol_filter)
{
  const blockvector *bv = cust->blockvector ();
  if (bv == nullptr)
    return;

  for (const struct block *b : bv->blocks ())
    {
      if (b == nullptr)
	continue;

      for (struct symbol *sym : b->multidict_symbols ())
	{
	  if (!pdb_is_pdb_location (sym))
	    continue;

	  const char *name = sym->natural_name ();
	  if (name == nullptr)
	    continue;

	  if (symbol_filter != nullptr && strcmp (name, symbol_filter) != 0)
	    continue;

	  auto *baton
	    = (struct pdb_loclist_baton *) SYMBOL_LOCATION_BATON (sym);
	  if (baton == nullptr)
	    {
	      gdb_printf ("  `%s`: <no baton>\n", name);
	      continue;
	    }

	  struct type *t = sym->type ();
	  gdb_printf ("  `%s`", name);
	  if (t != nullptr)
	    {
	      std::string ts = type_to_string (t);
	      if (!ts.empty ())
		gdb_printf (" (%s)", ts.c_str ());
	    }

	  gdb_printf (":\n");

	  if (baton->entries == nullptr)
	    {
	      gdb_printf ("    <no entries>\n");
	      continue;
	    }

	  for (auto *e = baton->entries; e != nullptr; e = e->next)
	    {
	      const char *regname = (e->gdb_regnum >= 0)
				      ? gdbarch_register_name (gdbarch,
							       e->gdb_regnum)
				      : "???";

	      if (e->is_full_scope)
		{
		  if (e->is_register)
		    gdb_printf ("    [full]  %s\n", regname);
		  else
		    gdb_printf ("    [full]  [%s+%d]\n", regname,
				(int) e->offset);
		}
	      else
		{
		  if (e->is_register)
		    gdb_printf ("      [0x%s-0x%s) %s", phex_nz (e->start, 8),
				phex_nz (e->end, 8), regname);
		  else
		    gdb_printf ("      [0x%s-0x%s) [%s+%d]",
				phex_nz (e->start, 8), phex_nz (e->end, 8),
				regname, (int) e->offset);
		}

	      if (!e->is_full_scope && e->num_gaps > 0)
		{
		  gdb_printf (" gaps={");
		  for (int i = 0; i < e->num_gaps; i++)
		    {
		      if (i > 0)
			gdb_printf (", ");
		      gdb_printf ("0x%s-0x%s", phex_nz (e->gaps[i].start, 8),
				  phex_nz (e->gaps[i].end, 8));
		    }

		  gdb_printf ("}");
		}

	      if (!e->is_full_scope)
		gdb_printf ("\n");
	    }
	}
    }
}

/* 'info pdb-locations' command — dump resolved variable location batons.  */
static void
pdb_info_locations_command (const char *args, int /*from_tty*/)
{
  struct pdb_per_objfile *pdb;
  uint32_t mod_start;
  uint32_t mod_end;

  if (!prepare_info_command (args, &pdb, &mod_start, &mod_end, { "symbol" }))
    {
      gdb_printf (
	"Usage: maintenance info pdb-locations modi=N[symbol=NAME]\n");
      return;
    }

  std::string symbol_str;
  const char *symbol_filter = nullptr;
  if (get_info_arg (args, "symbol", &symbol_str))
    symbol_filter = symbol_str.c_str ();

  gdb_printf ("  PDB '%s'\n", !pdb->pdb_file_path.empty ()
				? pdb->pdb_file_path.c_str ()
				: "<unknown>");

  for (auto i = mod_start; i < mod_end; i++)
    {
      struct pdb_module_info *mod = &pdb->modules[i];

      gdb_printf ("  Mod %04u | `%s`:\n", i,
		  mod->module_name ? mod->module_name : "<unnamed>");

      /* Build the module if not already expanded — this creates GDB
      symbols with batons from the raw PDB records.  */
      struct compunit_symtab *cust = pdb_build_module (pdb, mod);
      if (cust == nullptr)
	{
	  gdb_printf ("    (no symbols)\n");
	  continue;
	}

      pdb_dump_locations (cust, pdb->objfile->arch (), symbol_filter);
    }
}

/* Return a human-readable name for a CodeView leaf type record.  */
static const char *
pdb_leaf_type_name (uint16_t leaf)
{
  switch (leaf)
    {
    case LF_VTSHAPE:
      return "LF_VTSHAPE";
    case LF_LABEL:
      return "LF_LABEL";
    case LF_MODIFIER:
      return "LF_MODIFIER";
    case LF_POINTER:
      return "LF_POINTER";
    case LF_PROCEDURE:
      return "LF_PROCEDURE";
    case LF_MFUNCTION:
      return "LF_MFUNCTION";
    case LF_ARGLIST:
      return "LF_ARGLIST";
    case LF_FIELDLIST:
      return "LF_FIELDLIST";
    case LF_BITFIELD:
      return "LF_BITFIELD";
    case LF_METHODLIST:
      return "LF_METHODLIST";
    case LF_ARRAY:
      return "LF_ARRAY";
    case LF_CLASS:
      return "LF_CLASS";
    case LF_STRUCTURE:
      return "LF_STRUCTURE";
    case LF_UNION:
      return "LF_UNION";
    case LF_ENUM:
      return "LF_ENUM";
    case LF_INTERFACE:
      return "LF_INTERFACE";
    default:
      {
	static std::string buf;
	buf = string_printf ("LF_0x%04x", leaf);
	return buf.c_str ();
      }
    }
}

/* Try to extract the name from a TPI type record.
   Returns the name string, or nullptr if the record has no name.  */
static const char *
pdb_tpi_record_name (const struct pdb_tpi_type *rec)
{
  switch (rec->leaf)
    {
    case LF_CLASS:
    case LF_STRUCTURE:
      /* Name follows numeric leaf at LF_STRUCT_DATA_OFFS.  */
      if (rec->data_len > LF_STRUCT_DATA_OFFS)
	{
	  uint64_t size_val;
	  uint32_t nr = pdb_cv_read_numeric (rec->data + LF_STRUCT_DATA_OFFS,
					     rec->data_len
					       - LF_STRUCT_DATA_OFFS,
					     &size_val);
	  if (nr > 0 && LF_STRUCT_DATA_OFFS + nr < rec->data_len)
	    return CSTR (rec->data + LF_STRUCT_DATA_OFFS + nr);
	}

      break;

    case LF_UNION:
      /* Name follows numeric leaf at LF_UNION_DATA_OFFS.  */
      if (rec->data_len > LF_UNION_DATA_OFFS)
	{
	  uint64_t size_val;
	  uint32_t nr = pdb_cv_read_numeric (rec->data + LF_UNION_DATA_OFFS,
					     rec->data_len
					       - LF_UNION_DATA_OFFS,
					     &size_val);
	  if (nr > 0 && LF_UNION_DATA_OFFS + nr < rec->data_len)
	    return CSTR (rec->data + LF_UNION_DATA_OFFS + nr);
	}

      break;

    case LF_ENUM:
      /* Name at fixed offset LF_ENUM_NAME_OFFS.  */
      if (rec->data_len > LF_ENUM_NAME_OFFS)
	return CSTR (rec->data + LF_ENUM_NAME_OFFS);
      break;

    case LF_ARRAY:
      /* Name follows numeric leaf at LF_ARRAY_DATA_OFFS.  */
      if (rec->data_len > LF_ARRAY_DATA_OFFS)
	{
	  uint64_t size_val;
	  uint32_t nr = pdb_cv_read_numeric (rec->data + LF_ARRAY_DATA_OFFS,
					     rec->data_len
					       - LF_ARRAY_DATA_OFFS,
					     &size_val);
	  if (nr > 0 && LF_ARRAY_DATA_OFFS + nr < rec->data_len)
	    return CSTR (rec->data + LF_ARRAY_DATA_OFFS + nr);
	}

      break;

    default:
      break;
    }

  return nullptr;
}

/* Return a short display name for a type index.  For simple types
   (< 0x1000) returns the C type name; for TPI types returns the
   record's embedded name.  Returns nullptr for types without names
   (e.g. LF_POINTER, LF_MODIFIER).  */
static const char *
pdb_type_index_name (const struct pdb_tpi_context *tpi, uint32_t ti)
{
  if (CV_TI_IS_SIMPLE (ti))
    {
      uint32_t kind = CV_SIMPLE_KIND (ti);
      uint32_t mode = CV_SIMPLE_MODE (ti);
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
	case CV_SBYTE:
	  base = "int8_t";
	  break;
	case CV_BYTE:
	  base = "uint8_t";
	  break;
	case CV_INT16:
	  base = "short";
	  break;
	case CV_UINT16:
	  base = "unsigned short";
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
	  return nullptr;
	}

      if (mode == CV_TM_DIRECT)
	return base;

      /* Pointer to simple type — build "type*" string.  */
      static std::string pbuf;
      pbuf = string_printf ("%s*", base);
      return pbuf.c_str ();
    }

  /* TPI type — look up record and extract name.  */
  if (ti >= tpi->type_idx_begin && ti < tpi->type_idx_end)
    {
      uint32_t idx = ti - tpi->type_idx_begin;
      return pdb_tpi_record_name (&tpi->types[idx]);
    }

  return nullptr;
}

/* Print a type index followed by its resolved name in parentheses.  */
static void
pdb_print_ti (const struct pdb_tpi_context *tpi, const char *label,
	      uint32_t ti)
{
  const char *name = pdb_type_index_name (tpi, ti);
  if (name != nullptr && name[0] != '\0')
    gdb_printf ("  %s=0x%04x(%s)", label, ti, name);
  else
    gdb_printf ("  %s=0x%04x", label, ti);
}

/* Parse a leaf kind filter string.  Accepts "LF_STRUCTURE", "LF_CLASS", etc.
   Returns the numeric leaf value, or 0 if unrecognized.  */
static uint16_t
pdb_parse_leaf_kind (const char *str)
{
  struct leaf_kind
  {
    const char *name;
    uint16_t val;
  };
  static const std::array<leaf_kind, 16> kinds = { {
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
  } };

  for (const auto &k : kinds)
    if (strcasecmp (str, k.name) == 0)
      return k.val;

  return 0;
}

/* Print referenced type indices for a TPI record.  Each record type stores
   different cross-references; show the most useful ones.  Resolves type
   indices to names.  */
static void
pdb_print_type_refs (const struct pdb_tpi_context *tpi,
		     const struct pdb_tpi_type *rec)
{
  switch (rec->leaf)
    {
    case LF_MODIFIER:
      if (rec->data_len >= LF_MOD_SIZE)
	pdb_print_ti (tpi, "type", read_u32 (rec->data + LF_MOD_TYPE_OFFS));
      break;

    case LF_POINTER:
      if (rec->data_len >= LF_PTR_MIN_SIZE)
	pdb_print_ti (tpi, "utype", read_u32 (rec->data + LF_PTR_UTYPE_OFFS));
      break;

    case LF_ARRAY:
      if (rec->data_len >= LF_ARRAY_MIN_SIZE)
	{
	  pdb_print_ti (tpi, "elem",
			read_u32 (rec->data + LF_ARRAY_ELEMTYPE_OFFS));
	  pdb_print_ti (tpi, "idx",
			read_u32 (rec->data + LF_ARRAY_IDXTYPE_OFFS));
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
	  if (prop & CV_PROP_FWDREF)
	    gdb_printf (" [fwdref]");
	}

      break;

    case LF_UNION:
      if (rec->data_len >= LF_UNION_MIN_SIZE)
	{
	  uint16_t prop = read_u16 (rec->data + LF_UNION_PROPERTY_OFFS);
	  gdb_printf ("  fields=0x%04x",
		      read_u32 (rec->data + LF_UNION_FIELDLIST_OFFS));
	  if (prop & CV_PROP_FWDREF)
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
	  if (prop & CV_PROP_FWDREF)
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
	  gdb_printf ("  bits=%u pos=%u", rec->data[LF_BITFIELD_LENGTH_OFFS],
		      rec->data[LF_BITFIELD_POSITION_OFFS]);
	}

      break;

    default:
      break;
    }
}

/* 'info pdb-types' command — dump TPI type records with optional
   kind= filter.  */
static void
pdb_info_types_command (const char *args, int /*from_tty*/)
{
  std::string path;
  std::string kind_str;

  if (!validate_args (args, { "path", "kind" }))
    {
      gdb_printf ("Usage: info pdb-types[path=<pdb-path>][kind=LF_xxx]\n");
      return;
    }

  const pdb_per_objfile *pdb = pdb_get_default ();
  if (get_info_arg (args, "path", &path))
    {
      pdb = pdb_require_loaded_path (path);
      if (pdb == nullptr)
	return;
    }

  if (pdb == nullptr)
    {
      gdb_printf ("No PDB loaded and no default PDB available.\n");
      return;
    }

  uint16_t kind_filter = 0;
  if (get_info_arg (args, "kind", &kind_str))
    {
      kind_filter = pdb_parse_leaf_kind (kind_str.c_str ());
      if (kind_filter == 0)
	{
	  gdb_printf ("Unknown leaf kind: %s\n", kind_str.c_str ());
	  gdb_printf ("Valid kinds: LF_MODIFIER, LF_POINTER, LF_PROCEDURE, "
		      "LF_MFUNCTION, LF_ARGLIST, LF_FIELDLIST, LF_BITFIELD, "
		      "LF_METHODLIST, LF_ARRAY, LF_CLASS, LF_STRUCTURE, "
		      "LF_UNION, LF_ENUM, LF_VTSHAPE, LF_LABEL, "
		      "LF_INTERFA       CE\n");
	  return;
	}
    }

  const pdb_tpi_context *tpi = &pdb->tpi;

  gdb_printf ("  PDB '%s'\n", !pdb->pdb_file_path.empty ()
				? pdb->pdb_file_path.c_str ()
				: "<unknown>");
  gdb_printf ("  TPI range: 0x%04x .. 0x%04x (%u types)\n",
	      tpi->type_idx_begin, tpi->type_idx_end,
	      tpi->type_idx_end - tpi->type_idx_begin);

  uint32_t count = 0;

  for (uint32_t ti = tpi->type_idx_begin; ti < tpi->type_idx_end; ti++)
    {
      uint32_t idx = ti - tpi->type_idx_begin;
      const pdb_tpi_type *rec = &tpi->types[idx];

      if (kind_filter != 0 && rec->leaf != kind_filter)
	continue;

      const char *kind_name = pdb_leaf_type_name (rec->leaf);
      const char *rec_name = pdb_tpi_record_name (rec);

      gdb_printf ("  0x%04x: %-14s  len=%-4u", ti, kind_name, rec->length);

      pdb_print_type_refs (tpi, rec);

      if (rec_name != nullptr && rec_name[0] != '\0')
	gdb_printf ("  `%s`", rec_name);

      gdb_printf ("\n");
      count++;
    }

  gdb_printf ("  %u type records%s.\n", count,
	      kind_filter ? " (filtered)" : "");
}

} // namespace pdb

/* Register PDB maintenance info commands.  */
INIT_GDB_FILE (pdb_info)
{
  using namespace pdb;
  add_cmd ("pdb-loaded-files", class_maintenance,
	   pdb_info_loaded_files_command,
	   _ ("List currently loaded PDB file paths."), &maintenanceinfolist);

  add_cmd ("pdb-modules", class_maintenance, pdb_info_modules_command,
	   _ ("List PDB modules.\n\
  Usage: maintenance info pdb-modules[path=<pdb-path>][modi=N]\n\
  If path is omitted, dumps all modules from all loaded PDBs.\n\
  If modi is omitted, dumps all modules."),
	   &maintenanceinfolist);

  add_cmd ("pdb-files", class_maintenance, pdb_info_files_command,
	   _ ("List source files in PDB modules (from DBI File Info).\n\
  Usage: maintenance info pdb-files[path=<pdb-path>][modi=N]\n\
  If path is omitted, uses the default (main program) PDB.\n\
  If modi is omitted, dumps all modules."),
	   &maintenanceinfolist);

  add_cmd ("pdb-files-c13", class_maintenance, pdb_info_files_c13_command,
	   _ ("List source files in PDB modules with checksums\n\
  (from module streams).\n\
  Usage: maintenance info pdb-files-c13[path=<pdb-path>][modi=N]\n\
  If path is omitted, uses the default (main program) PDB.\n\
  If modi is omitted, dumps all modules."),
	   &maintenanceinfolist);

  add_cmd ("pdb-lines", class_maintenance, pdb_info_lines_command,
	   _ ("List line/address entries for files in PDB modules.\n\
  Usage: maintenance info pdb-lines[path=<pdb-path>][modi=N]\n\
  If path is omitted, uses the default (main program) PDB.\n\
  If modi is omitted, dumps all modules."),
	   &maintenanceinfolist);

  add_cmd ("pdb-symbols", class_maintenance, pdb_info_symbols_command,
	   _ ("Dump raw CodeView symbol records from PDB module streams.\n\
  Usage: maintenance info pdb-symbols[path=<pdb-path>][modi=N]\n\
  If path is omitted, uses the default (main program) PDB.\n\
  If modi is omitted, dumps all modules."),
	   &maintenanceinfolist);

  add_cmd (
    "pdb-sym-records", class_maintenance, pdb_info_sym_records_command,
    _ ("Dump raw CodeView records from the global symbol record stream.\n\
  Usage: maintenance info pdb-sym-records[path=<pdb-path>]\n\
  If path is omitted, uses the default (main program) PDB."),
    &maintenanceinfolist);

  add_cmd ("pdb-gsi", class_maintenance, pdb_info_gsi_command,
	   _ ("Dump GSI (global symbol index) hash records.\n\
  Usage: maintenance info pdb-gsi[path=<pdb-path>]\n\
  If path is omitted, uses the default (main program) PDB."),
	   &maintenanceinfolist);

  add_cmd ("pdb-psi", class_maintenance, pdb_info_psi_command,
	   _ ("Dump PSGSI (public symbol index) hash records.\n\
  Usage: maintenance info pdb-psi[path=<pdb-path>]\n\
  If path is omitted, uses the default (main program) PDB."),
	   &maintenanceinfolist);

  add_cmd ("pdb-locations", class_maintenance, pdb_info_locations_command,
	   _ ("Dump resolved PDB variable location batons.\n\
  Usage: maintenance info pdb-locations modi=N[symbol=NAME]\n\
  If symbol is given, only that variable is shown."),
	   &maintenanceinfolist);

  add_cmd ("pdb-types", class_maintenance, pdb_info_types_command,
	   _ ("Dump TPI type records from the PDB.\n\
Usage: maintenance info pdb-types[path=<pdb-path>][kind=LF_xxx]\n\
If path is omitted, uses the default (main program) PDB.\n\
kind filters by leaf type (e.g. LF_STRUCTURE, LF_ENUM)."),
	   &maintenanceinfolist);
}
