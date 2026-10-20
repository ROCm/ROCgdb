/* PDB debugging format support for GDB - Header file.

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

#ifndef GDB_PDB_PDB_INTERNAL_H
#define GDB_PDB_PDB_INTERNAL_H

#include "pdb/pdb.h"
#include "gdbtypes.h"
#include "symtab.h"
#include "buildsym.h"
#include "gdbarch.h"
#include "gdbsupport/filestuff.h"
#include "gdbsupport/array-view.h"
#include "gdbsupport/byte-vector.h"
#include "gdbsupport/function-view.h"

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>

struct buildsym_compunit;
struct objfile;

namespace pdb
{

/* Half-open PC ranges [first, second).  */
using pdb_range_pair_vec = std::vector<std::pair<CORE_ADDR, CORE_ADDR>>;

struct pdb_per_objfile;

/* Open scope while walking a module's symbols.  */
struct pdb_scope_frame
{
  /* End address handed to buildsym pop_context.  */
  CORE_ADDR end;
};

/* Scopes opened and not yet closed while walking a module's symbols,
   innermost last.  */
using pdb_scope_stack = std::vector<pdb_scope_frame>;

inline uint32_t
read_u32 (const void *p)
{
  return bfd_getl32 (p);
}

inline uint16_t
read_u16 (const void *p)
{
  return bfd_getl16 (p);
}

inline uint8_t
read_u8 (const void *p)
{
  return *static_cast<const uint8_t *> (p);
}

/* Read a CodeView signature (e.g. CV_SIGNATURE_C13) from the start of a
   C13 debug section.  The signature occupies CV_SIGNATURE_SIZE bytes.  */

inline uint32_t
read_cv_signature (const void *p)
{
  return read_u32 (p);
}

/* Round a ULONGEST value up to the next N-byte boundary.  */

inline ULONGEST
align_up (ULONGEST val, int n)
{
  return ::align_up (val, n);
}

/* Round P up to an N-byte boundary, preserving its type.  */

template<typename T>
inline T *
align_up (T *p, int n)
{
  return (T *) ::align_up ((uintptr_t) p, n);
}

/* Return true if N is a power of 2.  N must be non-zero.  */

inline bool
is_power_of_2 (uint32_t n)
{
  return (n & (n - 1)) == 0;
}

#define CSTR(data) ((const char *) (data))

/* "set debug pdb-read": reader debug messages when non-zero; at 2 and
   above also symbol lookup and source path conversion traces.  */
extern unsigned int pdb_read_debug;

/* Print a "pdb" debug statement if pdb_read_debug is non-zero.  */
#define pdb_dbg_printf(fmt, ...)                                       \
  debug_prefixed_printf_cond_func (pdb_read_debug >= 1, "pdb", nullptr, \
				   fmt, ##__VA_ARGS__)

/* Error — fatal issue that stops the operation.  */
#define pdb_error(fmt, ...) error (_("PDB Error: " fmt), ##__VA_ARGS__)

/* Warning — important one-time message that allows continuation.  */
#define pdb_warning(fmt, ...) warning (_("PDB Warning: " fmt), ##__VA_ARGS__)

/* Complaint — recoverable diagnostic issue (repeatable, suppressible).  */
#define pdb_complaint(fmt, ...) \
  complaint (_("PDB Complaint: " fmt), ##__VA_ARGS__)

/* Use a warning for OBJF_MAINLINE and a complaint for other objfiles,
   allowing missing dependency PDB diagnostics to be suppressed.  */
#define pdb_load_diagnostic(objfile, fmt, ...)		\
  do							\
    {							\
      if (((objfile)->flags & OBJF_MAINLINE) != 0)	\
	pdb_warning (fmt, ##__VA_ARGS__);		\
      else						\
	pdb_complaint (fmt, ##__VA_ARGS__);		\
    }							\
  while (0)

/* MSF SuperBlock is the first block in the PDB file and contains some basic
   information like the block size, number of blocks, and most importantly
   the location of the stream directory, which is used to locate all the
   other streams in the file.  The SuperBlock is 64 bytes long, but we only
   need a few fields from it.  Each stream is split across multiple blocks
   as recorded in this directory.  The directory itself might be split into
   multiple blocks which is described by the block map in the MSF SuperBlock.
   https://llvm.org/docs/PDB/MsfFile.html
*/
inline constexpr std::string_view PDB_MSF_MAGIC
  = "Microsoft C/C++ MSF 7.00\r\n\x1A\x44\x53\x00\x00";
inline constexpr auto PDB_MSF_MAGIC_SIZE = 32;

inline constexpr auto MSF_HEADER_SIZE = 64;

/* Block size in bytes.  */
inline constexpr auto MSF_BLOCK_SIZE_OFFS = 32;
/* Number of blocks.  */
inline constexpr auto MSF_NUM_BLOCKS_OFFS = 40;
/* Number of directory bytes.  */
inline constexpr auto MSF_NUM_DIRECTORY_BYTES_OFFS = 44;
/* Block number of the directory block map.  */
inline constexpr auto MSF_BLOCK_MAP_ADDR_OFFS = 52;

/* PDB streams.  */
inline constexpr auto PDB_STREAM_PDB = 1;
inline constexpr auto PDB_STREAM_TPI = 2;
inline constexpr auto PDB_STREAM_DBI = 3;
/* Invalid stream - microsoft-pdb/msf.cpp.  */
inline constexpr auto PDB_NO_STREAM = 0xFFFFFFFF;

/* Size of the per-record header: uint16_t RecordLen + uint16_t RecordKind.  */
inline constexpr auto CV_REC_HDR_SIZE = 4;

/* The PDB info stream (stream 1) contains version/age/GUID followed by
   the "named stream map" which is a hash table that maps stream identifiers
   (e.g. "/names", "/LinkInfo") to MSF stream numbers.  Except for a few fixed
   streams like PDB info, TPI, IPI and DBI, all other streams have arbitrary
   Id and need to be found by name.
   Here we only need the /names stream as that is where the global filename
   table resides, used to resolve filenames in C13 line info.  To find the
   stream we need to access the named stream map at the end of the header.
   See: https://llvm.org/docs/PDB/HashTable.html.

   PDB info stream layout:
     0:  Version
     4:  Signature
     8:  Age
     12: GUID
     28: Named stream map:
	StringBuffer:
	- Size (4 bytes)
	- Buffer (variable - null-terminated stream names)
	HashTable:
	- Size (4 bytes)
	- Capacity (4 bytes)
	- present bit vector:
	- PresentWordCount  (4 bytes)
	- PresentWords[PresentWordCount]
	- deleted bit vector:
	- DeletedWordCount (4 bytes)
	- DeletedWords[DeletedWordCount]
	- Table: (key=uint32 StringBuffer offset, value=uint32 stream number)
   Named stream map contains a hash table (HashTable) with keys being offsets
   into a string buffer (StringBuffer) where all the stream names are stored.
   So, to find the stream number for a given stream name (e.g. "/names") we need
   to walk the hash table entries whose present bit is set until the key points
   to the StringBuffer entry that matches the name we are looking for.
   The HashTable size should match the number of bits set in present bit vector.
   Note: Each bit in PresentWords field corresponds to one HashTable entry.
   Field PresentWordCount just tells how many words are in the vector.  E.g. if
   the HashTable capacity is 70 entries PresentWordCount is 3, while
   PresentWords[2] will contain present bits for entries 64–70.

   /names stream layout:
     0: Signature (4) — 0xEFFEEFFE
     4: Hash version (4) — 1
     8: String data size (4)
     12: String data/table (variable)
     Then hash table follows (we don't need it)
*/

/* Offset of the named stream map.  */
inline constexpr auto INFO_STREAM_HASH_MAP_OFFS = 28;
/* Fixed Info header size.  */
inline constexpr auto INFO_STREAM_MIN_SIZE = 28;
/* StringBuffer size */
inline constexpr auto NAMES_MAP_STRBUF_SIZE_OFFS = 0;
/* StringBuffer */
inline constexpr auto NAMES_MAP_STRBUF_OFFS = 4;
/* uint32_t size */
inline constexpr auto NAMES_MAP_UINT32_SIZE = 4;
/* Hash table size */
inline constexpr auto HASH_TABLE_SIZE_OFFS = 0;
/* Hash table capacity */
inline constexpr auto HASH_TABLE_CAPACITY_OFFS = 4;
/* PresentWordCount (4 bytes) */
inline constexpr auto HASH_TABLE_PRESENT_CNT_OFFS = 8;
/* Hash table header size */
inline constexpr auto HASH_TABLE_PRESENT_WORD_OFFS = 12;
/* Bits per uint32_t word */
inline constexpr auto HASH_TABLE_BIT_WORD_SIZE = 32;
/* Bytes per uint32_t word */
inline constexpr auto HASH_TABLE_BYTES_PER_WORD = 4;
/* Hash entry size (key + value) */
inline constexpr auto HASH_TABLE_ENTRY_SIZE = 8;
/* Hash entry key size */
inline constexpr auto HASH_TABLE_KEY_SIZE = 4;
/* Signature (4 bytes) */
inline constexpr auto NAMES_STREAM_SIGNATURE_OFFS = 0;
/* String data byte count (4 bytes), excluding header and hash table.  */
inline constexpr auto NAMES_STREAM_DATA_SIZE_OFFS = 8;
/* Expected signature */
inline constexpr auto NAMES_STREAM_SIGNATURE_VAL = 0xEFFEEFFE;
/* Minimum size (12 bytes) */
inline constexpr auto NAMES_STREAM_MIN_SIZE = 12;

/* DBI stream contains the debug information (line numbers, symbols, etc.)
   for all the modules (object files) linked into the program.
   See: https://llvm.org/docs/PDB/DbiStream.html
   The stream layout:
      - DBI stream header (fixed 64 bytes)
      - Module Info Headers: array of variable length headers, one per module,
	The length of each header is 64 bytes plus the variable
	size for the names of the module and object file, padded to
	4-byte boundary.
      - Contribution section: maps PE sections to modules.
      - Section Map: not used here.
      - File Info substream: defines the mapping from module to the source
	files that contribute to that module
      - Type Server Map Substream: not used here
      - EC Substream: not used here.
      - Optional Debug Header Stream.
      - Module streams.  Each module stream contains the debug information
	for the corresponding module.  Each stream contains 3 debug records:
	- Symbol records.
	  - C11 line info records.  Ignored (old line info format).
	  - C13 line info records.
   See: https://llvm.org/docs/PDB/DbiStream.html
*/

/* DBI stream header layout (NewDBIHdr in microsoft-pdb)
   We only need few fields from the header, and use it for checks only.
   The header also holds the Version Header (4 bytes) at 4, the Age
   (4 bytes) at 8, the Build Num. (2 bytes) at 14 and the Dll Ver.
   (2 bytes) at 18.
*/
/* Version Signature (4 bytes) */
inline constexpr auto DBI_HDR_SIGNATURE_OFFS = 0;
/* Public Symbol Stream Index (2 bytes) */
inline constexpr auto DBI_HDR_PSGSI_STREAM_OFFS = 16;
/* Symbol Record Stream (2 bytes) */
inline constexpr auto DBI_HDR_SYM_RECORD_STREAM_OFFS = 20;
/* Module Info Size (4 bytes) */
inline constexpr auto DBI_HDR_MODULE_SIZE_OFFS = 24;
/* Section Contrib Size (4 bytes) */
inline constexpr auto DBI_HDR_SECT_CONTRIB_OFFS = 28;
/* Section Map Size (4 bytes) */
inline constexpr auto DBI_HDR_SECT_MAP_OFFS = 32;
/* File Info Size (4 bytes) */
inline constexpr auto DBI_HDR_SOURCE_INFO_OFFS = 36;

/* Total header size */
inline constexpr auto DBI_HDR_SIZE = 64;

/* Expected signature value */
inline constexpr auto DBI_HDR_SIGNATURE_VAL = 0xFFFFFFFF;
/* Substreams start here.  */
#define DBI_HDR_MODULE_INFO_OFFS DBI_HDR_SIZE

/* File Info substream — contains info on each module's files.
   It follows the Section Map substream; DBI_HDR_SOURCE_INFO_OFFS holds its
   size.  A 4-byte header holds the number of modules and of source files,
   followed by:
      Module index array: num modules × 2 bytes
      File count array: num modules × 2 bytes
      File name offset array: one 4-byte offset per file, summed over all
	modules
      Names buffer: the file names the offsets point into
   Each module has its slice of the file offset arrays, however, the size of
   each slice is not explicitly stored anywhere.  Instead, we have to read the
   file count for each module and sum them up to know where the next module's
   slice starts.  */

/* Offset of NumModules (uint16_t) in the header.  */
inline constexpr auto FILE_INFO_HDR_NUM_MODULES_OFFS = 0;
/* Header size: NumModules and NumSourceFiles.  */
inline constexpr auto FILE_INFO_HDR_SIZE = 4;
/* The module index array starts right after the header.  */
inline constexpr auto FILE_INFO_MOD_INDICES_OFFS = FILE_INFO_HDR_SIZE;
/* Size of one module index or file count.  */
inline constexpr auto FILE_INFO_ELEMENT_SIZE = 2;

/* Section Contribution offsets within a Module Info header.  */

/* Section identifier (2 bytes)  */
inline constexpr auto MODI_SC_ISECT_OFFS = 4;
/* Offset in the section (4 bytes) */
inline constexpr auto MODI_SC_OFFSET_OFFS = 8;
/* Size of the contribution (4 bytes) */
inline constexpr auto MODI_SC_SIZE_OFFS = 12;

/* Module Info Header (ModInfo in microsoft-pdb) specifies the sizes
   of various debug data blocks in the corresponding module stream.
   Total size is 64 bytes plus the variable-length names.  It also holds
   the Flags (2 bytes) at 32 and the Num. of Source files (2 bytes) at
   48.  */

/* ModuleSymStream (2 bytes) */
inline constexpr auto MODI_STREAM_NUM_OFFS = 34;
/* Symbol stream size (4 bytes) */
inline constexpr auto MODI_SYM_BYTES_OFFS = 36;
/* C11 Bytes Size (4 bytes) */
inline constexpr auto MODI_C11_BYTES_OFFS = 40;
/* C13 Bytes Size (4 bytes) */
inline constexpr auto MODI_C13_BYTES_OFFS = 44;
/* Size of the fixed part of the header */
inline constexpr auto MODI_FIXED_SIZE = 64;

/* Module stream layout: starts with a 4-byte CV signature, then
   symbol records.  */
/* Offset past CV signature to symbols */
inline constexpr auto PDB_MODULE_SYMBOLS_OFFS = 4;

/* C13 line data, after the C11 area of a module stream: it maps the
   module's code addresses to source files and lines.  Sizes below are
   in bytes.  A leading 4-byte CV_SIGNATURE_C13 is accepted before it.

   It is a sequence of subsections, each 4-byte aligned:
      4: type  (DEBUG_S_LINES, DEBUG_S_FILECHKSMS, ...)
      4: size
      var: data (size bytes, then padding to a 4-byte boundary)

   DEBUG_S_FILECHKSMS is the module's file table, one CV_FileChecksum
   per source file.  Line data names a file by the byte offset of its
   entry there (a file id); the entry gives the name's offset in the
   /names string table.

   DEBUG_S_LINES is the line table of one contiguous code range; with
   clang-cl, one function's code.  A line entry (CV_Line) maps a code
   address in the range to a source line.  It holds no file, so the
   range's entries are stored in consecutive groups by source file,
   each headed by a CV_FileBlock naming the file.  A file gets a new
   group each time the code returns to it.

      CV_LineSection (12 bytes): the code range
	 4: offs    — start of the range within the PE section
	 2: seci    — PE section number (1-based, e.g. .text)
	 2: flags   — bit 0: CV_LINES_HAVE_COLUMNS
	 4: code_sz — size of the range

      then one or more groups, each:
      CV_FileBlock (12 bytes)
	 4: file_id    — the file's byte offset in DEBUG_S_FILECHKSMS
	 4: num_lines  — number of CV_Line entries in the group
	 4: block_size — size of the group: header, lines and columns
      CV_Line[num_lines] (8 bytes each)
	 4: offs       — code offset from CV_LineSection.offs
	 4: line_start — bits 0-23: start line; bits 24-30: end-line delta;
			 bit 31: statement (1) or expression (0)
      With CV_LINES_HAVE_COLUMNS, CV_Column[num_lines] follows:
	 2: start column
	 2: end column  */

/* CV_LineSection flags bit: each file block's lines are followed by
   CV_Column records.  */
inline constexpr uint16_t CV_LINES_HAVE_COLUMNS = 0x0001;

/* DEBUG_S subsection types (C13 format) */

inline constexpr auto C13_SUBSECT_HEADER_SIZE = 8;

/* CV Signatures */
inline constexpr auto CV_SIGNATURE_SIZE = 4;
inline constexpr auto CV_SIGNATURE_C13 = 4L;

/* Supported C13 subsection types.  */
inline constexpr auto DEBUG_S_LINES = 0xF2;
inline constexpr auto DEBUG_S_FILECHKSMS = 0xF4;

/* RSDS data from the PE CodeView record.  */
struct pdb_rsds_info
{
  std::array<gdb_byte, 16> guid = {};
  uint32_t age = 0;
  std::string pdb_path;
};

/* Read the RSDS record from the PE Debug Directory.
   Returns nullopt if the binary has no CODEVIEW/RSDS entry or the entry exists
   but uses an unsupported format.  */
extern std::optional<pdb_rsds_info>
pdb_read_rsds_info (bfd *abfd, const objfile *objfile);

/* Format 16-byte CodeView GUID as canonical
   XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX string.  */
extern std::string pdb_format_guid (const gdb_byte *guid);

/* Find the PDB file for the given objfile.  Locations are tried in
   the following order:
    1. <EXE dir>/<PDB basename from RSDS>.
    2. <EXE path with extension replaced by .pdb>.
    3. Full embedded PDB path from the RSDS record, as-is.  */
extern std::string pdb_find_pdb_file (const objfile *objfile,
				      const pdb_rsds_info &rsds);

/* Convert a Unix/MSYS2-style PATH to a native path.  A path that is not
   Unix-style is returned unchanged.  */
std::string pdb_convert_path (const char *path);

/* PDB Module information */
struct pdb_module_info
{
  /* Module name borrowed from pdb_per_objfile::dbi_data.  */
  char *module_name;

  /* Module debug stream index; 0xFFFF means no stream.  */
  uint16_t stream_number;
  /* Symbol-area byte count, including the leading CodeView signature.  */
  uint32_t sym_byte_size;
  /* Size of C11 records in the module stream.  */
  uint32_t c11_byte_size;
  /* Size of C13 records in the module stream.  */
  uint32_t c13_byte_size;

  /* The section contribution in the module's Module Info record.  It tells
     whether the module has code, and is the compunit's range only for a
     module without functions; otherwise its functions give that range.
     1-based section index; 0 means none.  */
  uint16_t sc_section;
  /* Offset within section.  */
  uint32_t sc_offset;
  /* Size of contribution in bytes.  */
  uint32_t sc_size;

  /* The module's DEBUG_S_FILECHKSMS subsection.  A line block or an inline
     site names its source file by the byte offset of the file's entry
     here, and the entry holds the file name's offset in /names.  */
  gdb_byte *file_checksums;
  uint32_t file_checksums_size;

  /* Set when pdb_build_module starts on this module, so a lookup made
     during the build does not build it again.  It stays set when the
     module yields no compunit; CU is the result, or null.  */
  bool expanded;
  compunit_symtab *cu;

  /* Source language of this module, cached on first use.  */
  enum language language = language_unknown;
  /* Pointers to the files in the Names Buffer (in pdb_per_objfile) that belong
     to this module, assigned from the module offset array
     (file_name_offsets below).  That array is parsed eagerly because the
     offset calculations are easier to do for all modules at once than per
     module.  */
  uint16_t num_files;
  const char **files;

  /* Offset array provides the offsets into the Names Buffer for each file
     in this module.  Populated during the initialization - this is because
     the location of each module's offset slice depends on the sizes of the
     slices before it.  */
  const gdb_byte *file_name_offsets;
};

/* Indexed raw TPI record.  Payload bytes borrow the PDB mapping or an
   objfile-obstack copy when the payload spans MSF blocks.  */
struct pdb_tpi_type
{
  /* CodeView record type (LF_*).  */
  uint16_t leaf;
  /* Payload after RecordKind, excluding both header words.  */
  const gdb_byte *data;
  /* Size of data: RecordLen less the 2 bytes of RecordKind.  */
  uint32_t data_len;
};

/* TPI record index.  */
struct pdb_tpi_context
{
  /* The raw TPI records, one per type index, read when the PDB is loaded.
     A type index finds its record here; the GDB type is built from the
     record later, the first time it is needed.  */
  pdb_tpi_type *types = nullptr;
  /* Type index of the first record; smaller indices are built-in types.  */
  uint32_t type_idx_begin = 0;
  /* One past the last type index.  */
  uint32_t type_idx_end = 0;

  /* Cache of the GDB types built so far, by type index; null until a type
     is built.  Built-in types are cached here too.  */
  type **type_cache = nullptr;

  /* Type used for unsupported types.  */
  type *undefined_type = nullptr;

  /* Type indices whose types are being built right now.  Reaching one of
     them again means the types refer to each other in a cycle; the reader
     then reports it and uses a placeholder type instead of recursing
     forever.  */
  std::vector<uint32_t> types_in_progress;
};

/* MSF stream.  */
struct msf_stream
{
  /* Size in bytes of the stream.  */
  uint32_t size;

  /* List of blocks that, when concatenated, form the stream.  */
  std::vector<uint32_t> block_nums;
};

/* Read-only memory mapping of the whole PDB file.
   Mapped for the objfile's lifetime.  */

struct pdb_file_mapping
{
  pdb_file_mapping () = default;
  ~pdb_file_mapping ();
  pdb_file_mapping (const pdb_file_mapping &) = delete;
  pdb_file_mapping &operator= (const pdb_file_mapping &) = delete;

  bool ok () const { return base != nullptr; }

  /* Mapping base, or null when unmapped.  */
  const gdb_byte *base = nullptr;
  /* Mapped length in bytes (the PDB file size).  */
  uint64_t size = 0;
  /* Windows file-mapping object handle, unused on POSIX.  */
  void *map_handle = nullptr;
};

/* PDB per-objfile data */
struct pdb_per_objfile
{
  explicit pdb_per_objfile (::objfile *objfile)
    : objfile (objfile)
  {
  }

  /* Associated objfile, not owned by this context.  */
  ::objfile *objfile;

  /* Path of the loaded PDB file on disk.  */
  std::string pdb_file_path;

  /* File handle owned by this context for seek-based block reads.  */
  gdb_file_up pdb_file;

  /* MSF file size  */
  size_t msf_size;

  /* MSF SuperBlock */
  uint32_t block_size;
  uint32_t num_blocks;
  uint32_t num_dir_bytes;
  uint32_t block_map_addr;

  /* Stream directory */
  std::vector<msf_stream> streams;

  /* Owned, cached stream buffers.  These are read once at PDB load
     time and never reassigned or resized afterwards, so interior
     pointers (string_table, file_info_names_buffer, per-module
     file_name_offsets, GSI/PSI record pointers) stay valid for the
     objfile's lifetime.  */
  /* PDB_STREAM_PDB  */
  gdb::byte_vector info_data;
  /* /names stream  */
  gdb::byte_vector names_data;
  /* PDB_STREAM_DBI  */
  gdb::byte_vector dbi_data;
  /* Symbol record stream  */
  gdb::byte_vector sym_record_data;
  /* PDB file mapping.  */
  pdb_file_mapping file_mapping;

  /* The name of an S_PUB32 public symbol at each unrelocated address, the
     last one read when several share it.  A procedure or data record
     gives only a symbol's natural name; its linkage name is that of a
     public at its address.  Keys are unrelocated because the map is
     filled when the PDB is loaded, before the image is relocated.  Values
     point into sym_record_data.  */
  std::unordered_map<CORE_ADDR, const char *> pub_mangled_names;

  /* Module info */
  std::vector<pdb_module_info> modules;

  /* Section mapping: per-section VMA from the PE/COFF BFD, indexed by
     (CodeView section number - 1).  */
  std::vector<CORE_ADDR> section_addresses;

  /* Back-link to the symbol parser currently walking a symbol stream
     (if any).  Set for the parser's lifetime by the function that
     creates it.  */
  class pdb_sym_parser *cur_parser = nullptr;
  /* The File Info substream is per PDB, not per module, so it is loaded
     with the PDB.  Since it is the larger part of the DBI stream, it is
     kept as a pointer into the DBI stream, which stays alive.  */
  gdb_byte *file_info_names_buffer;
  uint32_t file_info_names_buffer_size;

  /* Symbol record stream number from DBI header.  */
  uint16_t sym_record_stream;
  /* Public Symbol Information stream index.  */
  uint16_t psgsi_stream;

  /* TODO: free after PSI/GSI load.  Blocked: GSI records and the
     sym-records dump command both point into this buffer.  */

  /* The /names stream's string table: null-terminated strings that
     records refer to by byte offset, such as the source file name of a
     CV_FileChecksum entry.  Points into names_data.  */
  gdb_byte *string_table;
  uint32_t string_table_size;

  /* Parsed TPI stream.  Check tpi.types to verify parsing succeeded.  */
  pdb_tpi_context tpi;

  /* Read a stream by index.  Returns an owned buffer.  To cache a
     stream for the lifetime of the objfile, move the result into one
     of the dedicated owning fields above (info_data, names_data,
     dbi_data, sym_record_data).  Returns an empty vector if
     the stream doesn't exist.  */
  gdb::byte_vector read_stream (uint32_t stream_idx) const;

  /* Map a (section, offset) pair to a relocated PC.  Returns 0 on
     failure (invalid section index or missing section addresses).  */
  CORE_ADDR map_section_offset_to_pc (uint16_t section, uint32_t offset);

  CORE_ADDR map_section_offset_unrelocated (uint16_t section, uint32_t offset);

  /* True if SECTION is valid (maps to a known section address).
     Section 0 is reserved.  */
  bool section_valid (uint16_t section) const
  {
    return section != 0 && section <= section_addresses.size ();
  }
};

/* Header of a DEBUG_S_LINES subsection (12 bytes): the code range its
   line entries cover.  See microsoft-pdb cvinfo.h.  */
struct CV_LineSection
{
  static constexpr uint32_t SIZE = 12;

  explicit CV_LineSection (const gdb_byte *p)
    : offs (read_u32 (p + 0)),
      seci (read_u16 (p + 4)),
      flags (read_u16 (p + 6)),
      code_sz (read_u32 (p + 8))
  {
  }

  uint32_t offs;
  uint16_t seci;
  uint16_t flags;
  uint32_t code_sz;
};

/* Header (12 bytes) of a group of consecutive line entries in a
   DEBUG_S_LINES range whose code comes from one source file; it names
   that file, and the entries follow it.  See microsoft-pdb cvinfo.h.  */
struct CV_FileBlock
{
  static constexpr uint32_t SIZE = 12;

  explicit CV_FileBlock (const gdb_byte *p)
    : file_id (read_u32 (p + 0)),
      num_lines (read_u32 (p + 4)),
      block_size (read_u32 (p + 8))
  {
  }

  uint32_t file_id;
  uint32_t num_lines;
  uint32_t block_size;
};

/* An entry of the module's file table, DEBUG_S_FILECHKSMS: the source
   file's name, as an offset in /names, and its checksum.  The 6-byte
   header is followed by checksum_size checksum bytes.  See microsoft-pdb
   cvinfo.h.  */
struct CV_FileChecksum
{
  static constexpr uint32_t HDR_SIZE = 6;

  explicit CV_FileChecksum (const gdb_byte *p)
    : file_name_offs (read_u32 (p + 0))
  {
  }

  uint32_t file_name_offs;
};

/* A line entry: maps a code address, given as an offset from the start
   of its DEBUG_S_LINES range, to a source line.  The code from that
   address up to the next entry's comes from that line.  8 bytes; the
   second word packs linenumStart (bits 0-23), deltaLineEnd (bits
   24-30) and fStatement (bit 31), and the constructor keeps
   linenumStart.  See microsoft-pdb cvinfo.h.  */
struct CV_Line
{
  static constexpr uint32_t SIZE = 8;

  explicit CV_Line (const gdb_byte *p)
    : offs (read_u32 (p + 0)),
      line_start (read_u32 (p + 4) & 0x00FFFFFF)
  {
  }

  uint32_t offs;
  uint32_t line_start;
};

/* Callback type for pdb_walk_c13_line_blocks.  Invoked once per
   DEBUG_S_LINES file block with the resolved filename, the line section
   header and a pointer to NUM_LINES consecutive on-disk CV_Line records
   that the callback can wrap as needed.  */
using pdb_line_block_fn
  = gdb::function_view<void (const char *filename, CV_LineSection line_sect,
			     const gdb_byte *lines, uint32_t num_lines)>;

/* Return the pdb_per_objfile associated with OBJFILE, or nullptr if
   no PDB data has been loaded for this objfile.  */
extern pdb_per_objfile *get_pdb_per_objfile (objfile *objfile);

/* Source files for a module's procedure symbols, and the module's line
   table.

   A procedure record (S_GPROC32, S_LPROC32) gives a function's name,
   type, address and size, but not the source file it is defined in;
   only the line table shows that, by the file of the line entry at the
   function's entry PC.  AT_PC maps the PC of every line entry in the
   module to its file's subfile, and each procedure symbol is queued in
   TO_APPLY with the subfile at its entry PC.  A subfile gets its symtab
   only when the compunit is built, so each queued symbol is given that
   symtab afterwards (otherwise GDB's buildsym puts the symbol in the
   module's main source file).

   ROWS are the line table entries of the whole module: one per CV_Line
   that maps to a PC, plus one the reader adds at the end of each code
   range.  They are sorted by PC and recorded in the compunit once all
   line blocks are read.  */

struct pdb_symbol_lines
{
  struct pending
  {
    struct symbol *sym;
    struct ::subfile *subfile;
  };

  /* Row PC -> the file of the first row there.  */
  std::unordered_map<CORE_ADDR, struct ::subfile *> at_pc;

  /* Symbols awaiting their symtab, applied after the compunit is built.  */
  std::vector<pending> to_apply;

  /* Every row of the module, across all files.  Collected first and emitted
     in address order, so a file's run can be closed where the next file's
     code starts.  */
  struct row
  {
    CORE_ADDR pc;
    /* 0 ends the line before: no source line from PC on.  */
    int line;
    struct ::subfile *subfile;
    linetable_entry_flags flags;
  };
  std::vector<row> rows;
};

/* Parse CodeView symbol records from a module stream into GDB symbols.
   MODULE_STREAM is the raw module stream data (caller owns it).
   If FUNC_RANGES is non-null, each function's [start, end) is appended
   so the caller can register discontiguous block ranges.  */
extern void pdb_parse_symbols (pdb_per_objfile *pdb, pdb_module_info *mod_info,
			       gdb_byte *module_stream, buildsym_compunit *cu,
			       pdb_range_pair_vec *func_ranges,
			       enum language lang,
			       pdb_symbol_lines *sym_lines);

/* Return a module's language from its first S_COMPILE2/S_COMPILE3 record,
   cached in the module; C++ when there is none.  */

extern enum language pdb_module_language (pdb_per_objfile *pdb,
					  pdb_module_info *mod_info);

/* Read the symbol record stream into pdb->sym_record_data.  */
extern void pdb_read_sym_record_stream (pdb_per_objfile *pdb);

/* Build minimal symbols from S_PUB32 records in the PSI stream.
   Called during PDB init after sections are available.  */
extern void pdb_build_minsyms (pdb_per_objfile *pdb);

/* Read LEN bytes from stream STREAM_IDX starting at OFF, where OFF is a
   logical offset into the stream (the stream seen as one flat array).

   If the range lies within a single block a simple pointer to the file mapping
   is returned and SCRATCH is left untouched.  If the range spans multiple
   blocks, the bytes are gathered, block by block, into SCRATCH and a pointer
   into SCRATCH is returned; in that case the returned pointer equals
   scratch.data ().

   PDB->file_mapping must be mapped.  */
extern const gdb_byte *pdb_stream_bytes (pdb_per_objfile *pdb,
					 uint32_t stream_idx, uint32_t off,
					 uint32_t len,
					 gdb::byte_vector &scratch);

/* Read a module's debug stream and extract file checksums onto the
   objfile obstack.  Returns an owned buffer with the raw stream data;
   the caller keeps it alive as long as the data is needed.  Returns
   an empty vector if the module has no stream.  */
extern gdb::byte_vector pdb_read_module_stream (pdb_per_objfile *pdb,
						pdb_module_info *mod);

/* CodeView simple types (cvinfo.h).  Indices below 0x1000 encode kind in
   bits 0-7 and pointer mode in bits 8-10; bit 11 is reserved.  */

/* Extract kind (bits 0-7) from a simple type index.  */
static inline uint32_t
cv_simple_kind (uint32_t ti)
{
  return ti & 0xFF;
}

/* Extract mode (bits 8-11) from a simple type index.  */
static inline uint32_t
cv_simple_mode (uint32_t ti)
{
  return (ti >> 8) & 0x0F;
}

/* True if TI is a simple (built-in) type index.  */
static inline bool
cv_ti_is_simple (uint32_t ti)
{
  return ti < 0x1000;
}

/* SimpleTypeKind values */
inline constexpr auto CV_NONE = 0x00;
inline constexpr auto CV_VOID = 0x03;
inline constexpr auto CV_HRESULT = 0x08;
inline constexpr auto CV_SIGNED_CHAR = 0x10;
inline constexpr auto CV_UNSIGNED_CHAR = 0x20;
inline constexpr auto CV_NARROW_CHAR = 0x70;
inline constexpr auto CV_WIDE_CHAR = 0x71;
inline constexpr auto CV_CHAR16 = 0x7a;
inline constexpr auto CV_CHAR32 = 0x7b;
inline constexpr auto CV_CHAR8 = 0x7c;
inline constexpr auto CV_SHORT = 0x11;
inline constexpr auto CV_USHORT = 0x21;
inline constexpr auto CV_INT8 = 0x68;
inline constexpr auto CV_UINT8 = 0x69;
inline constexpr auto CV_LONG = 0x12;
inline constexpr auto CV_ULONG = 0x22;
inline constexpr auto CV_QUAD = 0x13;
inline constexpr auto CV_UQUAD = 0x23;
inline constexpr auto CV_INT16 = 0x72;
inline constexpr auto CV_UINT16 = 0x73;
inline constexpr auto CV_INT32 = 0x74;
inline constexpr auto CV_UINT32 = 0x75;
inline constexpr auto CV_INT64 = 0x76;
inline constexpr auto CV_UINT64 = 0x77;
inline constexpr auto CV_OCT = 0x14;
inline constexpr auto CV_UOCT = 0x24;
inline constexpr auto CV_INT128 = 0x78;
inline constexpr auto CV_UINT128 = 0x79;
inline constexpr auto CV_FLOAT32 = 0x40;
inline constexpr auto CV_FLOAT64 = 0x41;
inline constexpr auto CV_FLOAT80 = 0x42;
inline constexpr auto CV_COMPLEX32 = 0x50;
inline constexpr auto CV_COMPLEX64 = 0x51;
inline constexpr auto CV_COMPLEX80 = 0x52;
inline constexpr auto CV_BOOL8 = 0x30;

/* SimpleTypeMode values */
/* No indirection.  */
inline constexpr auto CV_TM_DIRECT = 0;
/* 32-bit near pointer.  */
inline constexpr auto CV_TM_NPTR32 = 4;
/* 32-bit far pointer.  */
inline constexpr auto CV_TM_FPTR32 = 5;
/* 64-bit near pointer.  */
inline constexpr auto CV_TM_NPTR64 = 6;
/* 128-bit near pointer.  */
inline constexpr auto CV_TM_NPTR128 = 7;

/* Global Symbol Streams.

   The DBI header references three stream indices that together form
   the global symbol infrastructure:

   1. Symbol Records Stream: index at DBI_HDR_SYM_RECORD_STREAM_OFFS.
      Array of symbol records that GSI and PSI streams reference.
      Layout of each symbol record matches the layout of CodeView symbol
      records in module streams:
	  uint16_t reclen    — length of data after this field
	  uint16_t rectype   — CodeView record type
	  uint8_t  data[]    — record-type-specific payload
   2. Global Symbol Index (GSI stream) - its index is in the DBI header.
      GSI is a hash table that maps the symbol name to the location in the
      module symbol records where the symbol is fully described.
      Contains mostly S_PROCREF / S_LPROCREF / S_DATAREF records
      (which are cross-references pointing into module streams),
      plus S_UDT, S_CONSTANT, etc.  Layout:
	GSI Header (GSIHashHdr in microsoft-pdb gsi.h):
	  uint32_t VerSignature  (0)  always 0xFFFFFFFF
	  uint32_t VerHdr        (4)  always 0xeffe0000 + 19990810
	  uint32_t cbHr          (8)  byte count of hash records
	  uint32_t cbBuckets     (12) byte count of bucket data
	HashRecord[cbHr/8]:
	  uint32_t Off   (0)  1-based byte offset into SymRecordStream
	  uint32_t CRef  (4)  reference count (vestigial, always 1)
	Bucket data (cbBuckets bytes):
	  uint32_t bitmap[ceil(4097/32)] — one bit per bucket
	  uint32_t offsets[popcount(bitmap)] — compressed array,
	    one byte-offset-into-HR-array per present bucket

      Name lookup algorithm:
	1. Hash the name with hashStringV1 (XOR 4-byte LE chunks,
	   fold remainder, apply bit mixing), then mod 4097.
	2. Check bitmap[hash/32] bit (hash%32).  If unset → miss.
	3. Count set bits before that position (rank query) to get
	   the index into the compressed offsets[] array.
	4. offsets[rank] gives the byte offset into the HR array.
	   Walk contiguous HashRecords from that point; re-hash each
	   record's symbol name and stop when the bucket changes.
	5. strcmp the name to find an exact match.

   3. PSGSI stream (DBI_HDR_PSGSI_STREAM_OFFS) — Public Symbol Index
      Indexes the publicly exported symbols (S_PUB32 only).
      Embeds a complete GSI hash table after a metadata header.
      Layout (byte offsets from stream start):

	Offset 0-27: PublicsStreamHeader (28 bytes = PSGSI_HDR_SIZE):
	  0-3:   uint32_t SymHash;         byte size of embedded GSI hash data
	  4-7:   uint32_t AddrMap;        byte size of address map
	  8-11:  uint32_t NumThunks;      number of thunk records
	  12-15: uint32_t SizeOfThunk;    size of each thunk
	  16-17: uint16_t ISectThunkTable; PE section of thunk table
	  18-19: uint16_t padding;        reserved
	  20-23: uint32_t OffThunkTable;  offset to thunk table
	  24-27: uint32_t NumSections;    number of sections

	Offset 28+: GSI Hash Data (SymHash bytes):
	  GSI Header (see GSI).
	  HashRecord[] (see GSI):
	  Bucket data (see GSI):

	Offset 28 + SymHash+: Address map (AddrMap bytes)
	  uint32_t AddrMap[]             (sorted address-order indices)

   Usage during init (pdb_read_pdb_file):
     - pdb_read_sym_record_stream: caches SymRecordStream into
       pdb->sym_record_data.
     - pdb_build_minsyms: walks PSGSI hash records, resolves each
       S_PUB32 in sym_record_data, creates minimal_symbol entries.  */
/* On-disk size of the GSI hash header (first 4 uint32_t fields).  */

inline constexpr auto GSI_HASH_HDR_SIZE = 16;

inline constexpr auto PSGSI_HDR_SIZE = 28;
inline constexpr auto PSGSI_HDR_SYM_HASH_OFFS = 0;

inline constexpr auto GSI_HASH_RECORD_SIZE = 8;
inline constexpr auto GSI_HASH_RECORD_SYMOFFS_OFFS = 0;

/* GSI hash header (GSIHashHdr in microsoft-pdb gsi.h).
   For PSGSI, it comes after PublicsStreamHeader main header.  */
struct pdb_gsi_hdr
{
  /* HashRecord array.  */
  gdb::array_view<gdb_byte> hr;
};

/* CodeView symbol record kinds.  */
/* Global procedure */
inline constexpr auto S_GPROC32 = 0x1110;
/* Local procedure */
inline constexpr auto S_LPROC32 = 0x110f;
/* Global procedure with ID */
inline constexpr auto S_GPROC32_ID = 0x1147;
/* Local procedure with ID */
inline constexpr auto S_LPROC32_ID = 0x1146;
/* Global data */
inline constexpr auto S_GDATA32 = 0x110d;
/* Local data */
inline constexpr auto S_LDATA32 = 0x110c;
/* Public symbol */
inline constexpr auto S_PUB32 = 0x110e;
/* End of scope */
inline constexpr auto S_END = 0x0006;
/* Closes procedure scope */
inline constexpr auto S_PROC_ID_END = 0x114f;
/* Thunk (opens scope) */
inline constexpr auto S_THUNK32 = 0x1102;
/* Compiler info v2 */
inline constexpr auto S_COMPILE2 = 0x1116;
/* Compiler info v3 */
inline constexpr auto S_COMPILE3 = 0x113c;

/* Symbol record.  Layout:
     uint16_t reclen    (length of data following reclen)
     uint16_t rectype   (symbol type)
     ...  record-type-specific data ...
*/
inline constexpr auto PDB_RECORD_LEN_OFFS = 0;
inline constexpr auto PDB_RECORD_TYPE_OFFS = 2;
inline constexpr auto PDB_RECORD_DATA_OFFS = 4;
inline constexpr auto PDB_RECORD_HDR_SIZE = 4;

/* Name/field offsets in module symbol record bodies.  */
inline constexpr auto PDB_SYMBOL_FUNC_NAME_OFFS = 35;
inline constexpr auto PDB_SYMBOL_VAR_NAME_OFFS = 10;

/* CodeView type and id leaf kinds.  */

/* Virtual function table shape */
inline constexpr auto LF_VTSHAPE = 0x000a;
/* Label type */
inline constexpr auto LF_LABEL = 0x000e;

inline constexpr auto LF_PROCEDURE = 0x1008;
/* Argument list for procedures */
inline constexpr auto LF_ARGLIST = 0x1201;
/* Field list for structs/classes */
inline constexpr auto LF_FIELDLIST = 0x1203;
/* Method overload list */
inline constexpr auto LF_METHODLIST = 0x1206;

/* LF_PROCEDURE payload (after RecordKind).
       uint32_t rvtype      return type index
       uint8_t  calltype    calling convention (CV_call_e)
       uint8_t  funcattr    attributes (unused)
       uint16_t parmcount   number of parameters
       uint32_t arglist     type index of argument list  */
inline constexpr auto LF_PROC_RVTYPE_OFFS = 0;
inline constexpr auto LF_PROC_CALLTYPE_OFFS = 4;
inline constexpr auto LF_PROC_ARGLIST_OFFS = 8;
/* Minimum record data size.  */
inline constexpr auto LF_PROC_SIZE = 12;

/* CV_call_e value for the platform's ordinary C calling convention.  */
inline constexpr uint8_t CV_CALL_NEAR_C = 0x00;

/* LF_ARGLIST sub-record layout.
       uint32_t count         number of type indices
       uint32_t arg[count]    array of type indices.
   A trailing type index 0 marks an ellipsis, not a void parameter.  The
   count includes that marker; it excludes an implicit 'this' parameter.  */
inline constexpr auto LF_ARGLIST_COUNT_OFFS = 0;
inline constexpr auto LF_ARGLIST_ARGS_OFFS = 4;
inline constexpr auto LF_ARGLIST_MIN_SIZE = 4;

/* Index raw TPI records (stream 2) and allocate the type cache.  Missing
   streams and detected format errors raise pdb_error.  Record indexing
   does not construct GDB types.  */

void pdb_read_tpi_stream (pdb_per_objfile *pdb);

/* Extract a NUL-terminated string from a bounded buffer [P, END).
   Returns the string pointer if a NUL byte exists in the range,
   nullptr otherwise.  Used to safely read variable-length name fields
   from PDB records without risking out-of-bounds reads.  */
const char *pdb_extract_string (const gdb_byte *p, const gdb_byte *end);

/* Resolve a TPI index to a GDB type after TPI initialization.  Unsupported
   indices/leaves return an error placeholder.  */
type &pdb_tpi_resolve_type (pdb_per_objfile *pdb, uint32_t type_index);

/* Return NAME, the name of a function of type FUNCTION_TYPE, as GDB names
   it: a C++ function's name is followed by its parameter list and
   canonicalized.  NAME is returned unchanged for other languages.  */
const char *pdb_function_name (pdb_per_objfile *pdb, const char *name,
			       type *function_type, enum language lang);

/* Return the canonical C++ spelling of NAME for tag and symbol lookup.
   Callers retain the returned pointer: unchanged names must already have
   persistent storage, and rewritten names must last for the objfile's
   lifetime.  Canonicalization reconciles spellings such as "Buffer<int,3>"
   and "Buffer<int, 3>".  */

const char *pdb_canonical_name (pdb_per_objfile *pdb, const char *name);

/* Header of a symbol record in the PDB symbol stream.  */

struct pdb_sym_record_hdr
{
  /* Length of the data following LEN.  */
  uint16_t len;

  /* Symbol record type.  */
  uint16_t type;

  /* Size of the full record.  */
  size_t rec_size () const
  {
    return len + sizeof (len);
  }
};

} /* namespace pdb */

/* GDB_PDB_PDB_INTERNAL_H */
#endif
