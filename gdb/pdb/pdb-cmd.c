/* PDB debugging format support for GDB.

Copyright (C) 1994-2026 Free Software Foundation, Inc.
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
along with this program.  If not, see <http://www.gnu.org/licenses/>.

This file implements various PDB related GDB info commands.  */

#include "cli/cli-utils.h"
#include "cli/cli-cmds.h"
#include "pdb/pdb.h"
#include "gdbsupport/rsp-low.h"
#include "block.h"
#include "gdbarch.h"
#include "value.h"

#include <string.h>
#include <stdint.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

/* Loaded PDB instances for all the objectfiles.  */
static std::vector<struct pdb_per_objfile *> loaded_pdbs;

/* Default PDB is the one whose object file is the main program (OBJF_MAINLINE).
Used by info commands that do not specify a PDB.  */
static struct pdb_per_objfile *default_pdb = nullptr;

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
if (arg.empty () || sanity_cnt++ > 10)
return tokens;
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
static struct pdb_per_objfile *
pdb_find_loaded_pdb (const char *path)
{
if (path == nullptr)
return nullptr;

for (struct pdb_per_objfile *pdb : loaded_pdbs)
if (pdb != nullptr && pdb->pdb_file_path != nullptr
&& strcmp (pdb->pdb_file_path, path) == 0)
return pdb;

return nullptr;
}

/* See pdb.h.  */
void
pdb_register_loaded_pdb (struct pdb_per_objfile *pdb)
{
loaded_pdbs.push_back (pdb);

if (pdb->objfile->flags & OBJF_MAINLINE)
default_pdb = pdb;
}

/* Check if pdb path is in the list of loaded PDBs.  Info commands will accept
PDB paths only for PDBs associated with the loaded object files.  */
static struct pdb_per_objfile *
pdb_require_loaded_path (const std::string &path)
{
struct pdb_per_objfile *pdb = pdb_find_loaded_pdb (path.c_str ());
if (pdb == NULL)
gdb_printf ("PDB not loaded: %s\n", path.c_str ());
return pdb;
}

/* For info commands that print checksum type.  */
static const char *
pdb_checksum_type_name (uint8_t type)
{
switch (type)
{
case 0: return "None";
case 1: return "MD5";
case 2: return "SHA-1";
case 3: return "SHA-256";
default: return "Unknown";
}
}

/* Shared among info pdb-files commands.  Parse path= and modi= arguments and
prepare PDB context for commands.  Returns true on success with pdb and
mod_start/mod_end set which are used to mark the modules to be processed
(either all or just one).  */
static bool
prepare_info_command (const char *args,
struct pdb_per_objfile **out_pdb,
uint32_t *out_mod_start,
uint32_t *out_mod_end,
const std::vector<std::string> &extra_keys = {})
{
std::string path;
std::string modi_str;

std::vector<std::string> valid = {"path", "modi"};
valid.insert (valid.end (), extra_keys.begin (), extra_keys.end ());

if (!validate_args (args, valid))
{
gdb_printf ("Usage: [path=<pdb-path>] [modi=N]\n");
return false;
}

/* Get PDB: use path= if provided, else use default.  */
struct pdb_per_objfile *pdb = default_pdb;
if (get_info_arg (args, "path", &path))
{
pdb = pdb_require_loaded_path (path);
if (pdb == NULL)
return false;
}

if (pdb == NULL)
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
gdb_printf ("Module index %lu out of range (0..%u).\n",
modi, pdb->num_modules - 1);
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
struct pdb_line_block_info *info = (struct pdb_line_block_info *) cb_data;
struct CV_LineSection *ls = info->line_sect;
struct CV_Line *lines = info->lines;
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
(unsigned) ls->seci,
range_start, range_end, num_lines);

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
pdb_info_modules_command (const char *args, int from_tty)
{
std::vector<struct pdb_per_objfile *> pdbs_to_dump;
std::string path = "";

if (!validate_args (args, {"path"}))
{
gdb_printf ("Usage: info pdb-modules [path=<pdb-path>]\n");
return;
}

/* If pdb path is provided it must point to one of the loaded PDBs.
Otherwise, dump all loaded PDBs.  */
if (get_info_arg (args, "path", &path))
{
struct pdb_per_objfile *pdb = pdb_require_loaded_path (path);
if (pdb == NULL)
{
gdb_printf ("PDB path not found in loaded PDBs: %s\n", path.c_str ());
return;
}
pdbs_to_dump.push_back (pdb);
}
else
{
pdbs_to_dump = loaded_pdbs;
}

if (pdbs_to_dump.empty ())
{
gdb_printf ("No PDB loaded.\n");
return;
}

/* Dump module info for each selected PDB.  */
for (struct pdb_per_objfile *pdb : pdbs_to_dump)
{
if (pdb == NULL)
continue;

gdb_printf ("  PDB '%s'\n",
pdb->pdb_file_path ? pdb->pdb_file_path : "<unknown>");

for (uint32_t i = 0; i < pdb->num_modules; i++)
{
struct pdb_module_info *mod = &pdb->modules[i];
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
pdb_info_loaded_files_command (const char *args, int from_tty)
{
if (args != NULL && *args != '\0')
{
gdb_printf ("Usage: info pdb-loaded-files\n");
return;
}

if (loaded_pdbs.empty ())
{
gdb_printf ("No PDB loaded.\n");
return;
}

for (const struct pdb_per_objfile *pdb : loaded_pdbs)
{
if (pdb == NULL || pdb->pdb_file_path == NULL
|| pdb->pdb_file_path[0] == '\0')
continue;
gdb_printf ("%s\n", pdb->pdb_file_path);
}
}

/* 'info pdb-lines' command.  */
static void
pdb_info_lines_command (const char *args, int from_tty)
{
struct pdb_per_objfile *pdb;
uint32_t mod_start, mod_end;

if (!prepare_info_command (args, &pdb, &mod_start, &mod_end))
return;

for (uint32_t i = mod_start; i < mod_end; i++)
{
/* pdb_read_dbi_stream already allocated pdb->modules array.  */
struct pdb_module_info *mod = &pdb->modules[i];
auto stream_buf = pdb_read_stream (pdb, mod->stream_number);

gdb_printf ("  Mod %04u | `%s`:\n", i,
mod->module_name ? mod->module_name : "<unnamed>");

if (stream_buf != nullptr && mod->c13_byte_size > 0)
{
struct pdb_line_block_info dcb;
pdb_walk_c13_line_blocks (pdb, mod, stream_buf.get (),
pdb_dump_line_block, &dcb);
}
}
}

/* 'info pdb-files' command.  */
static void
pdb_info_files_command (const char *args, int from_tty)
{
struct pdb_per_objfile *pdb;
uint32_t mod_start, mod_end;

if (!prepare_info_command (args, &pdb, &mod_start, &mod_end))
return;

gdb_printf ("  PDB '%s'\n",
pdb->pdb_file_path ? pdb->pdb_file_path : "<unknown>");

for (auto i = mod_start; i < mod_end; i++)
{
struct pdb_module_info *mod = &pdb->modules[i];
pdb_read_module_files (pdb, mod);

gdb_printf ("  Mod %04u | `%s`:\n", i,
mod->module_name ? mod->module_name : "<unnamed>");

for (auto j = 0; j < mod->num_files; j++)
{
const char *fname = mod->files ? mod->files[j] : NULL;
gdb_printf ("    - %s\n", fname ? fname : "<unknown>");
}
}
}

/* 'info pdb-files-c13' command.  */
static void
pdb_info_files_c13_command (const char *args, int from_tty)
{
struct pdb_per_objfile *pdb;
uint32_t mod_start, mod_end;

if (!prepare_info_command (args, &pdb, &mod_start, &mod_end))
return;

gdb_printf ("  PDB '%s'\n", pdb->pdb_file_path ?
pdb->pdb_file_path : "<unknown>");

for (uint32_t i = mod_start; i < mod_end; i++)
{
struct pdb_module_info *mod = &pdb->modules[i];
pdb_read_module_stream (pdb, mod);
pdb_read_module_files_c13 (pdb, mod);

gdb_printf ("  Mod %04u | `%s`:\n", i,
mod->module_name ? mod->module_name : "<unnamed>");

for (uint32_t j = 0; j < mod->num_files_c13; j++)
{
size_t checksum_len = 0;
struct pdb_file_info *fi = &mod->files_c13[j];

if (fi->checksum_size > 0 && fi->checksum != NULL)
{
/* Print hash type & hash - also record the printed length
so that we can correctly format entries w/o hash.  */
std::string hash_str = bin2hex (fi->checksum, fi->checksum_size);
auto len = snprintf (NULL, 0, " [%s: %s] ",
pdb_checksum_type_name (fi->checksum_type),
hash_str.c_str ());
if(len > 0)
checksum_len = len;

gdb_printf (" [%s: %s] ",
pdb_checksum_type_name (fi->checksum_type),
hash_str.c_str ());
}
else if (checksum_len > 0)
gdb_printf ("%*s", (int) checksum_len, "");

gdb_printf ("%s", fi->filename ? fi->filename : "<unknown>");
gdb_printf ("\n");
}
}
}


/* Dump a single symbol record with an optional prefix string.
Prefix serves different callers as they dump the record along
with other information.  */
static void
pdb_dump_sym_record (struct pdb_per_objfile *pdb,
gdb_byte *rec, uint16_t reclen, uint16_t rectype,
const std::string &prefix)
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
pdb_parse_sym_record_stream (struct pdb_per_objfile *pdb, uint32_t flags)
{
gdb_byte *syms_data = pdb->sym_record_data;
auto stream_size = pdb->sym_record_size;

if (syms_data == nullptr || stream_size == 0)
{
pdb_warning ("No symbol record stream data available");
return;
}

gdb_byte *syms_end = syms_data + stream_size;
gdb_byte *p = syms_data;
uint32_t record_num = 0;

/* We iterate by grabbing two values at a time: reclen and rectype. In each
loop check if there is 2 values available.  */
while (p + PDB_RECORD_HDR_SIZE <= syms_end)
{
uint16_t reclen, rectype;
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
oss << "    [" << record_num << "] " << std::hex
<< std::setfill ('0') << std::setw (4) << rectype
<< std::dec << " " << std::setfill (' ')
<< std::setw (18) << std::left
<< rec_name
<< " len=" << reclen;
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
pdb_dump_gsi_hash_records (struct pdb_per_objfile *pdb,
gdb_byte *hr_data, uint32_t hr_size)
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
gdb_byte *rec = hr_data + i * GSI_HASH_RECORD_SIZE;
uint32_t offs = UINT32_CAST (rec + GSI_HASH_RECORD_SYMOFFS_OFFS) - 1;

/* Check if offset is within SymRecordStream bounds.  */
if (offs >= syms_size || syms_size - offs < 4)
{
gdb_printf ("    [%u] offset=%u (out of bounds)\n", i, offs);
continue;
}

/* Now we got the pointer into the symbol record stream - first, analyze
the header.  */
gdb_byte *data = syms_data + offs;
gdb_byte *syms_end = syms_data + syms_size;

uint16_t reclen, rectype;
size_t rec_size;
if (!pdb_parse_sym_record_hdr (data, syms_end, reclen, rectype, rec_size))
{
gdb_printf ("    [%u] failed to parse symbol record header\n", i);
continue;
}

auto rec_name = pdb_sym_rec_type_name (rectype);
std::ostringstream oss;
oss << "    [" << i << "] offset:" << std::setw(10) << std::left
<< offs << " " << rec_name;
std::string prefix = oss.str();

pdb_dump_sym_record (pdb, data, reclen, rectype, prefix);
}
}

/* Dump GSI hash record header info.  */
static void
pdb_dump_gsi_hdr (const struct pdb_gsi_hdr &gsi)
{
gdb_printf ("    GSI header: sig:0x%08X ver:0x%08X hr_bytes:%u bucket_bytes:%u\n",
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

auto sym_hash_size = UINT32_CAST (psi_data + PSGSI_HDR_SYM_HASH_OFFS);
auto addr_map_size = UINT32_CAST (psi_data + PSGSI_HDR_ADDR_MAP_OFFS);

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
auto sym_offs = UINT32_CAST (addr_map_data + i * sizeof (uint32_t));

if (sym_offs >= pdb->sym_record_size ||
pdb->sym_record_size - sym_offs < 4)
{
gdb_printf ("    [%u] offset=%u (out of bounds)\n", i, sym_offs);
continue;
}

gdb_byte *data = pdb->sym_record_data + sym_offs;
gdb_byte *syms_end = pdb->sym_record_data + pdb->sym_record_size;

uint16_t reclen, rectype;
size_t rec_size;

if (!pdb_parse_sym_record_hdr (data, syms_end, reclen, rectype, rec_size))
{
gdb_printf ("    [%u] offset=%u (record extends past stream, reclen=%u)\n",
i, sym_offs, reclen);
continue;
}

auto rec_name = pdb_sym_rec_type_name (rectype);

std::ostringstream oss;
oss << "    [" << i << "] offset:" << std::setw(10) << std::left
<< sym_offs << " " << rec_name;
std::string prefix = oss.str();

pdb_dump_sym_record (pdb, data, reclen, rectype, prefix);
data += rec_size;
}
}

/* 'info pdb-symbols' command.  */
static void
pdb_info_symbols_command (const char *args, int from_tty)
{
struct pdb_per_objfile *pdb;
uint32_t mod_start, mod_end;

if (!prepare_info_command (args, &pdb, &mod_start, &mod_end, {"stream"}))
return;

/* If stream= is given, dispatch to the appropriate stream dumper.  */
std::string stream_str;
if (get_info_arg (args, "stream", &stream_str))
{
gdb_printf ("  PDB '%s'\n",
pdb->pdb_file_path ? pdb->pdb_file_path : "<unknown>");

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

gdb_printf ("  PDB '%s'\n",
pdb->pdb_file_path ? pdb->pdb_file_path : "<unknown>");

for (auto i = mod_start; i < mod_end; i++)
{
struct pdb_module_info *mod = &pdb->modules[i];
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
pdb_info_sym_records_command (const char *args, int from_tty)
{
struct pdb_per_objfile *pdb;
uint32_t mod_start, mod_end;

if (!prepare_info_command (args, &pdb, &mod_start, &mod_end))
return;

gdb_printf ("  PDB '%s'\n",
pdb->pdb_file_path ? pdb->pdb_file_path : "<unknown>");

pdb_parse_sym_record_stream (pdb, PDB_DUMP_SYM);
}

/* 'info pdb-gsi' command — dump GSI (global symbol index) stream.  */
static void
pdb_info_gsi_command (const char *args, int from_tty)
{
struct pdb_per_objfile *pdb;
uint32_t mod_start, mod_end;

if (!prepare_info_command (args, &pdb, &mod_start, &mod_end))
return;

gdb_printf ("  PDB '%s'\n",
pdb->pdb_file_path ? pdb->pdb_file_path : "<unknown>");

pdb_dump_gsi_stream (pdb);
}

/* 'info pdb-psi' command — dump PSGSI (public symbol index) stream.  */
static void
pdb_info_psi_command (const char *args, int from_tty)
{
struct pdb_per_objfile *pdb;
uint32_t mod_start, mod_end;

if (!prepare_info_command (args, &pdb, &mod_start, &mod_end))
return;

gdb_printf ("  PDB '%s'\n",
pdb->pdb_file_path ? pdb->pdb_file_path : "<unknown>");

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

if (symbol_filter != nullptr
&& strcmp (name, symbol_filter) != 0)
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
const char *regname
= (e->gdb_regnum >= 0)
? gdbarch_register_name (gdbarch, e->gdb_regnum)
: "???";

if (e->is_full_scope)
{
if (e->is_register)
gdb_printf ("    [full]  %s\n", regname);
else
gdb_printf ("    [full]  [%s+%d]\n",
regname, (int) e->offset);
}
else
{
if (e->is_register)
gdb_printf ("      [0x%s-0x%s) %s",
phex_nz (e->start, 8),
phex_nz (e->end, 8),
regname);
else
gdb_printf ("      [0x%s-0x%s) [%s+%d]",
phex_nz (e->start, 8),
phex_nz (e->end, 8),
regname, (int) e->offset);
}

if (!e->is_full_scope && e->num_gaps > 0)
{
gdb_printf (" gaps={");
for (int i = 0; i < e->num_gaps; i++)
{
if (i > 0) gdb_printf (", ");
gdb_printf ("0x%s-0x%s",
phex_nz (e->gaps[i].start, 8),
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
pdb_info_locations_command (const char *args, int from_tty)
{
struct pdb_per_objfile *pdb;
uint32_t mod_start, mod_end;

if (!prepare_info_command (args, &pdb, &mod_start, &mod_end, {"symbol"}))
{
gdb_printf ("Usage: info pdb-locations modi=N [symbol=NAME]\n");
return;
}

std::string symbol_str;
const char *symbol_filter = nullptr;
if (get_info_arg (args, "symbol", &symbol_str))
symbol_filter = symbol_str.c_str ();

gdb_printf ("  PDB '%s'\n",
pdb->pdb_file_path ? pdb->pdb_file_path : "<unknown>");

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

/* Register PDB info commands.  */
INIT_GDB_FILE (pdb_info)
{
add_info ("pdb-loaded-files", pdb_info_loaded_files_command,
_("List currently loaded PDB file paths."));

add_info ("pdb-modules", pdb_info_modules_command,
_("List PDB modules.\n\
Usage: info pdb-modules [path=<pdb-path>] [modi=N]\n\
If path is omitted, dumps all modules from all loaded PDBs.\n\
If modi is omitted, dumps all modules."));

add_info ("pdb-files", pdb_info_files_command,
_("List source files in PDB modules (from DBI File Info).\n\
Usage: info pdb-files [path=<pdb-path>] [modi=N]\n\
If path is omitted, uses the default (main program) PDB.\n\
If modi is omitted, dumps all modules."));

add_info ("pdb-files-c13", pdb_info_files_c13_command,
_("List source files in PDB modules with checksums\n\
(from module streams).\n\
Usage: info pdb-files-c13 [path=<pdb-path>] [modi=N]\n\
If path is omitted, uses the default (main program) PDB.\n\
If modi is omitted, dumps all modules."));

add_info ("pdb-lines", pdb_info_lines_command,
_("List line/address entries for files in PDB modules.\n\
Usage: info pdb-lines [path=<pdb-path>] [modi=N]\n\
If path is omitted, uses the default (main program) PDB.\n\
If modi is omitted, dumps all modules."));

add_info ("pdb-symbols", pdb_info_symbols_command,
_("Dump raw CodeView symbol records from PDB module streams.\n\
Usage: info pdb-symbols [path=<pdb-path>] [modi=N]\n\
If path is omitted, uses the default (main program) PDB.\n\
If modi is omitted, dumps all modules."));

add_info ("pdb-sym-records", pdb_info_sym_records_command,
_("Dump raw CodeView records from the global symbol record stream.\n\
Usage: info pdb-sym-records [path=<pdb-path>]\n\
If path is omitted, uses the default (main program) PDB."));

add_info ("pdb-gsi", pdb_info_gsi_command,
_("Dump GSI (global symbol index) hash records.\n\
Usage: info pdb-gsi [path=<pdb-path>]\n\
If path is omitted, uses the default (main program) PDB."));

add_info ("pdb-psi", pdb_info_psi_command,
_("Dump PSGSI (public symbol index) hash records.\n\
Usage: info pdb-psi [path=<pdb-path>]\n\
If path is omitted, uses the default (main program) PDB."));

add_info ("pdb-locations", pdb_info_locations_command,
_("Dump resolved PDB variable location batons.\n\
Usage: info pdb-locations modi=N [symbol=NAME]\n\
If symbol is given, only that variable is shown."));
}
