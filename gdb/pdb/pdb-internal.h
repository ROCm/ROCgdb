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
#include "gdbsupport/filestuff.h"
#include "gdbsupport/array-view.h"
#include "gdbsupport/byte-vector.h"
#include "gdbsupport/function-view.h"

#include <array>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

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

/* Return the little-endian 32-bit value at P.  */

inline uint32_t
read_u32 (const void *p)
{
  return bfd_getl32 (p);
}

/* Return the little-endian 16-bit value at P.  */

inline uint16_t
read_u16 (const void *p)
{
  return bfd_getl16 (p);
}

/* Return the byte at P.  */

inline uint8_t
read_u8 (const void *p)
{
  return *static_cast<const uint8_t *> (p);
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

/* DATA, the bytes of a NUL-terminated name, as a C string.  */
#define CSTR(data) ((const char *) (data))

/* "set debug pdb-read": reader debug messages when non-zero; at 2 and
   above also symbol lookup and source path conversion traces.  */
extern unsigned int pdb_read_debug;

/* Print a "pdb" debug statement if pdb_read_debug is non-zero.  */
#define pdb_dbg_printf(fmt, ...)                                       \
  debug_prefixed_printf_cond_func (pdb_read_debug >= 1, "pdb", nullptr, \
				   fmt, ##__VA_ARGS__)

/* Error: fatal issue that stops the operation.  */
#define pdb_error(fmt, ...) error (_("PDB Error: " fmt), ##__VA_ARGS__)

/* Warning: important one-time message that allows continuation.  */
#define pdb_warning(fmt, ...) warning (_("PDB Warning: " fmt), ##__VA_ARGS__)

/* Complaint: recoverable diagnostic issue (repeatable, suppressible).  */
#define pdb_complaint(fmt, ...) \
  complaint (_("PDB Complaint: " fmt), ##__VA_ARGS__)

/* Use a warning for OBJF_MAINLINE and a complaint for other objfiles,
   allowing missing dependency PDB diagnostics to be suppressed.  */
#define pdb_load_diagnostic(objfile, fmt, ...)		\
  do							\
    {							\
      if ((objfile) != nullptr				\
	  && ((objfile)->flags & OBJF_MAINLINE) != 0)	\
	pdb_warning (fmt, ##__VA_ARGS__);		\
      else						\
	pdb_complaint (fmt, ##__VA_ARGS__);		\
    }							\
  while (0)

/* MSF SuperBlock is the first block in the PDB file and contains some basic
   information like the block size, number of blocks, and most importantly
   the location of the stream directory, which is used to locate all the
   other streams in the file.  The SuperBlock is 56 bytes long, but we only
   need a few fields from it.  Each stream is split across multiple blocks
   as recorded in this directory.  The directory itself might be split into
   multiple blocks which is described by the block map in the MSF SuperBlock.
   https://llvm.org/docs/PDB/MsfFile.html  */
inline constexpr std::string_view PDB_MSF_MAGIC
  ("Microsoft C/C++ MSF 7.00\r\n\x1A" "DS\0\0\0", 32);
inline constexpr auto PDB_MSF_MAGIC_SIZE = PDB_MSF_MAGIC.size ();

inline constexpr auto MSF_HEADER_SIZE = 56;

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
	- PresentWordCount (4 bytes)
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
   PresentWords[2] will contain present bits for entries 64 to 69.

   /names stream layout:
     0: Signature (4): 0xEFFEEFFE
     4: Hash version (4): 1
     8: String data size (4)
     12: String data/table (variable)
     Then hash table follows (we don't need it).  */

/* Offset of the named stream map.  */
inline constexpr auto INFO_STREAM_HASH_MAP_OFFS = 28;
/* Fixed Info header size.  */
inline constexpr auto INFO_STREAM_MIN_SIZE = 28;
/* StringBuffer size.  */
inline constexpr auto NAMES_MAP_STRBUF_SIZE_OFFS = 0;
/* StringBuffer.  */
inline constexpr auto NAMES_MAP_STRBUF_OFFS = 4;
/* uint32_t size.  */
inline constexpr auto NAMES_MAP_UINT32_SIZE = 4;
/* Hash table size.  */
inline constexpr auto HASH_TABLE_SIZE_OFFS = 0;
/* Hash table capacity.  */
inline constexpr auto HASH_TABLE_CAPACITY_OFFS = 4;
/* PresentWordCount (4 bytes).  */
inline constexpr auto HASH_TABLE_PRESENT_CNT_OFFS = 8;
/* Hash table header size.  */
inline constexpr auto HASH_TABLE_PRESENT_WORD_OFFS = 12;
/* Bits per uint32_t word.  */
inline constexpr auto HASH_TABLE_BIT_WORD_SIZE = 32;
/* Bytes per uint32_t word.  */
inline constexpr auto HASH_TABLE_BYTES_PER_WORD = 4;
/* Hash entry size (key + value).  */
inline constexpr auto HASH_TABLE_ENTRY_SIZE = 8;
/* Hash entry key size.  */
inline constexpr auto HASH_TABLE_KEY_SIZE = 4;
/* Signature (4 bytes).  */
inline constexpr auto NAMES_STREAM_SIGNATURE_OFFS = 0;
/* String data byte count (4 bytes), excluding header and hash table.  */
inline constexpr auto NAMES_STREAM_DATA_SIZE_OFFS = 8;
/* Expected signature.  */
inline constexpr auto NAMES_STREAM_SIGNATURE_VAL = 0xEFFEEFFE;
/* Minimum size (12 bytes).  */
inline constexpr auto NAMES_STREAM_MIN_SIZE = 12;

/* DBI stream contains the debug information (line numbers, symbols, etc.)
   for all the modules (object files) linked into the program.
   See: https://llvm.org/docs/PDB/DbiStream.html
   The stream layout:
      - DBI stream header (fixed 64 bytes)
      - Module Info Headers: array of variable length headers, one per module.
	The length of each header is 64 bytes plus the variable
	size for the names of the module and object file, padded to
	4-byte boundary.
      - Contribution section: maps PE sections to modules.
      - Section Map: not used here.
      - File Info substream: defines the mapping from module to the source
	files that contribute to that module.
      - Type Server Map Substream: not used here.
      - EC Substream: not used here.
      - Optional Debug Header Stream.
      - Module streams.  Each module stream contains the debug information
	for the corresponding module, in three parts:
	- Symbol records.
	- C11 line info records.  Ignored (old line info format).
	- C13 line info records.
   See: https://llvm.org/docs/PDB/DbiStream.html  */

/* DBI stream header layout (NewDBIHdr in microsoft-pdb)
   We only need a few fields from the header, and use it for checks only.
   The header also holds the Version Header (4 bytes) at 4, the Build
   Num. (2 bytes) at 14 and the Dll Ver. (2 bytes) at 18.  */
/* Version Signature (4 bytes).  */
inline constexpr auto DBI_HDR_SIGNATURE_OFFS = 0;
/* Age (4 bytes).  */
inline constexpr auto DBI_HDR_AGE_OFFS = 8;
/* Public Symbol Stream Index (2 bytes).  */
inline constexpr auto DBI_HDR_PSGSI_STREAM_OFFS = 16;
/* Symbol Record Stream (2 bytes).  */
inline constexpr auto DBI_HDR_SYM_RECORD_STREAM_OFFS = 20;
/* Module Info Size (4 bytes).  */
inline constexpr auto DBI_HDR_MODULE_SIZE_OFFS = 24;
/* Section Contrib Size (4 bytes).  */
inline constexpr auto DBI_HDR_SECT_CONTRIB_OFFS = 28;
/* Section Map Size (4 bytes).  */
inline constexpr auto DBI_HDR_SECT_MAP_OFFS = 32;
/* File Info Size (4 bytes).  */
inline constexpr auto DBI_HDR_SOURCE_INFO_OFFS = 36;

/* Total header size.  */
inline constexpr auto DBI_HDR_SIZE = 64;

/* Expected signature value.  */
inline constexpr auto DBI_HDR_SIGNATURE_VAL = 0xFFFFFFFF;
/* Substreams start here.  */
inline constexpr auto DBI_HDR_MODULE_INFO_OFFS = DBI_HDR_SIZE;

/* File Info substream: contains info on each module's files.
   It follows the Section Map substream; DBI_HDR_SOURCE_INFO_OFFS holds its
   size.  A 4-byte header holds the number of modules and of source files,
   followed by:
      Module index array: num modules x 2 bytes
      File count array: num modules x 2 bytes
      File name offset array: one 4-byte offset per file, summed over all
	modules
      Names buffer: the file names the offsets point into
   Each module has its slice of the file name offset array.  The size of
   each slice is not stored anywhere: the file counts of the modules before
   it are summed to find where a module's slice starts.  */

/* Offset of NumModules (uint16_t) in the header.  */
inline constexpr auto FILE_INFO_HDR_NUM_MODULES_OFFS = 0;
/* Header size: NumModules and NumSourceFiles.  */
inline constexpr auto FILE_INFO_HDR_SIZE = 4;
/* The module index array starts right after the header.  */
inline constexpr auto FILE_INFO_MOD_INDICES_OFFS = FILE_INFO_HDR_SIZE;
/* Size of one module index or file count.  */
inline constexpr auto FILE_INFO_ELEMENT_SIZE = 2;

/* Section Contribution offsets within a Module Info header.  */

/* Section identifier (2 bytes).  */
inline constexpr auto MODI_SC_ISECT_OFFS = 4;
/* Offset in the section (4 bytes).  */
inline constexpr auto MODI_SC_OFFSET_OFFS = 8;
/* Size of the contribution (4 bytes).  */
inline constexpr auto MODI_SC_SIZE_OFFS = 12;

/* Section Contribution substream, after Module Info: a version word, then
   one entry per contribution.  A V2 entry appends a 4-byte COFF section
   index to the Ver60 layout.  */
inline constexpr uint32_t DBI_SC_VER_60 = 0xeffe0000 + 19970605;
inline constexpr uint32_t DBI_SC_VER_2 = 0xeffe0000 + 20140516;
inline constexpr auto DBI_SC_ENTRY_SIZE = 28;
inline constexpr auto DBI_SC2_ENTRY_SIZE = 32;
/* Section index (2 bytes).  */
inline constexpr auto DBI_SC_ISECT_OFFS = 0;
/* Offset in the section (4 bytes).  */
inline constexpr auto DBI_SC_OFFSET_OFFS = 4;
/* Size of the contribution (4 bytes).  */
inline constexpr auto DBI_SC_SIZE_OFFS = 8;
/* Characteristics of the section the contribution is in (4 bytes).  */
inline constexpr auto DBI_SC_CHARACTERISTICS_OFFS = 12;
/* Module index, 0-based (2 bytes).  */
inline constexpr auto DBI_SC_IMOD_OFFS = 16;

/* Flags of a section's Characteristics (IMAGE_SCN_* in the PE/COFF
   specification), which a section contribution repeats: the section holds
   code, or may be executed.  */
inline constexpr uint32_t PDB_SCN_CODE = 0x00000020;
inline constexpr uint32_t PDB_SCN_EXECUTE = 0x20000000;

/* Module Info Header (ModInfo in microsoft-pdb) specifies the sizes
   of various debug data blocks in the corresponding module stream.
   Total size is 64 bytes plus the variable-length names.  It also holds
   the Flags (2 bytes) at 32 and the Num. of Source files (2 bytes) at
   48.  */

/* ModuleSymStream (2 bytes).  */
inline constexpr auto MODI_STREAM_NUM_OFFS = 34;
/* Symbol stream size (4 bytes).  */
inline constexpr auto MODI_SYM_BYTES_OFFS = 36;
/* C11 Bytes Size (4 bytes).  */
inline constexpr auto MODI_C11_BYTES_OFFS = 40;
/* C13 Bytes Size (4 bytes).  */
inline constexpr auto MODI_C13_BYTES_OFFS = 44;
/* Size of the fixed part of the header.  */
inline constexpr auto MODI_FIXED_SIZE = 64;

/* Module stream layout: starts with a 4-byte CV signature, then
   symbol records.  */
/* Offset past CV signature to symbols.  */
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
	 4: offs    : start of the range within the PE section
	 2: seci    : PE section number (1-based, e.g. .text)
	 2: flags   : bit 0: CV_LINES_HAVE_COLUMNS
	 4: code_sz : size of the range

      then one or more groups, each:
      CV_FileBlock (12 bytes)
	 4: file_id    : the file's byte offset in DEBUG_S_FILECHKSMS
	 4: num_lines  : number of CV_Line entries in the group
	 4: block_size : size of the group: header, lines and columns
      CV_Line[num_lines] (8 bytes each)
	 4: offs       : code offset from CV_LineSection.offs
	 4: line_start : bits 0-23: start line; bits 24-30: end-line delta;
			 bit 31: statement (1) or expression (0)
      With CV_LINES_HAVE_COLUMNS, CV_Column[num_lines] follows:
	 2: start column
	 2: end column  */

/* CV_LineSection flags bit: each file block's lines are followed by
   CV_Column records.  */
inline constexpr uint16_t CV_LINES_HAVE_COLUMNS = 0x0001;

/* A C13 subsection's header: its type and its size, 4 bytes each.  */
inline constexpr auto C13_SUBSECT_HEADER_SIZE = 8;

/* CV Signatures.  */
inline constexpr auto CV_SIGNATURE_SIZE = 4;
inline constexpr uint32_t CV_SIGNATURE_C13 = 4;

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
   Returns nullopt if ABFD is not a PE file, has no CODEVIEW/RSDS entry, or
   the entry exists but uses an unsupported format.  */
extern std::optional<pdb_rsds_info> pdb_read_rsds_info (bfd *abfd);

/* What pdb_parse_rsds reads of a PE image: its image base, the RVA and
   size of its debug data directory, the size of the file (0 when
   unknown), the address and size of each section, READ, which reads
   LEN bytes at OFFSET of section I into BUF and returns whether it
   could, and READ_FILE, which does so at OFFSET of the file.  */

struct pdb_pe_image
{
  uint64_t image_base = 0;
  uint32_t debug_dir_rva = 0;
  uint32_t debug_dir_size = 0;
  uint64_t file_size = 0;

  struct section
  {
    uint64_t vma;
    uint64_t size;
  };
  std::vector<section> sections;

  gdb::function_view<bool (size_t i, gdb_byte *buf, uint64_t offset,
			   uint64_t len)> read;

  gdb::function_view<bool (uint64_t offset, gdb_byte *buf,
			   uint64_t len)> read_file;
};

/* The RSDS record of the debug directory of IMAGE, as pdb_read_rsds_info
   returns it.  */
extern std::optional<pdb_rsds_info> pdb_parse_rsds (const pdb_pe_image &image);

/* The number of bytes SECT spans in memory.  BFD gives an initialized PE
   section its raw size, which is less than its virtual size when the
   section ends in uninitialized data, as .data does when lld merges .bss
   into it; the objects in that tail are still the section's.  */
extern bfd_size_type pdb_section_extent (bfd *abfd, asection *sect);

/* Format 16-byte CodeView GUID as canonical
   XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX string.  */
extern std::string pdb_format_guid (const gdb_byte *guid);

/* Find the PDB file for the given objfile.  Locations are tried in
   the following order:
    1. <EXE dir>/<PDB basename from RSDS>.
    2. <EXE path with extension replaced by .pdb>.
    3. The full RSDS path, unless it is a network or device path or is not
       absolute on this host.  */
extern std::string pdb_find_pdb_file (const objfile *objfile,
				      const pdb_rsds_info &rsds);

/* Whether a usable file exists at a path.  */
using pdb_file_present_ftype = gdb::function_view<bool (const std::string &)>;

/* Find the PDB that the RSDS record RSDS names for the executable at
   EXE_PATH.  EXE_PATH must not be a null pointer.  PRESENT is called with
   each candidate path and returns true if a file exists there.  MAINLINE
   says whether to report that no PDB was found.  Return the path of the
   PDB, or an empty string if there is none.  */
extern std::string pdb_find_pdb_file (const char *exe_path,
				      const pdb_rsds_info &rsds, bool mainline,
				      pdb_file_present_ftype present);

/* Convert a Unix/MSYS2-style PATH to a native path.  A path that is not
   Unix-style is returned unchanged.  */
std::string pdb_convert_path (const char *path);

/* A module's stream number when the module has no stream.  */
inline constexpr uint16_t MODI_NO_STREAM = 0xFFFF;

/* PDB Module information.  */
struct pdb_module_info
{
  /* Module name borrowed from pdb_per_objfile::dbi_data.  */
  const char *module_name = nullptr;

  /* Module debug stream index; 0xFFFF means no stream.  */
  uint16_t stream_number = MODI_NO_STREAM;
  /* Symbol-area byte count, including the leading CodeView signature.  */
  uint32_t sym_byte_size = 0;
  /* Size of C11 records in the module stream.  */
  uint32_t c11_byte_size = 0;
  /* Size of C13 records in the module stream.  */
  uint32_t c13_byte_size = 0;

  /* The section contribution in the module's Module Info record.  It tells
     whether the module has code, and is the compunit's range only for a
     module without functions; otherwise its functions give that range.
     1-based section index; 0 means none.  */
  uint16_t sc_section = 0;
  /* Offset within section.  */
  uint32_t sc_offset = 0;
  /* Size of contribution in bytes.  */
  uint32_t sc_size = 0;

  /* The module's DEBUG_S_FILECHKSMS subsection.  A line block or an inline
     site names its source file by the byte offset of the file's entry
     here, and the entry holds the file name's offset in /names.  */
  const gdb_byte *file_checksums = nullptr;
  uint32_t file_checksums_size = 0;

  /* Set when pdb_build_module starts on this module, so a lookup made
     during the build does not build it again.  It stays set when the
     module yields no compunit; CU is the result, or null.  */
  bool expanded = false;
  compunit_symtab *cu = nullptr;

  /* Source language of this module, cached on first use.  */
  enum language language = language_unknown;
  /* The names of this module's NUM_FILES files in the Names Buffer (in
     pdb_per_objfile), read on first use (pdb_read_module_files) from the
     module's slice of the file name offset array, FILE_NAME_OFFSETS
     below.  */
  uint16_t num_files = 0;
  const char **files = nullptr;

  /* Offset array provides the offsets into the Names Buffer for each file
     in this module.  Populated during the initialization - this is because
     the location of each module's offset slice depends on the sizes of the
     slices before it.  */
  const gdb_byte *file_name_offsets = nullptr;
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

  /* Type of a procedure whose type index names no procedure type; see
     pdb_tpi_resolve_function_type.  */
  type *untyped_function_type = nullptr;

  /* The parameter list pdb_function_name printed for each function type,
     or null when it was too long to print, so that it is printed once
     however many procedures have the type.  */
  std::unordered_map<const type *, const char *> printed_param_lists;

  /* The copy of each function type that marks a function that does not
     return (CV_PFLAG_NEVER), shared by the functions of the type.  */
  std::unordered_map<type *, type *> no_return_types;

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
};

/* The names of the S_PUB32 public symbols at one address, indexed so that
   finding a record's public costs the same however many share the
   address.  Names point into the symbol record stream.  */

struct pdb_publics_at
{
  /* Add the public NAME.  */
  void add (const char *name);

  /* Every public at the address.  */
  std::unordered_set<std::string_view> names;

  /* The decorated publics ("?..."), each under the unqualified name its
     decoration starts with ("?name@"); the first one added wins.  */
  std::unordered_map<std::string_view, const char *> decorated_by_name;

  /* The last decorated public added, and the number of them.  */
  const char *decorated = nullptr;
  size_t num_decorated = 0;
};

/* PDB per-objfile data.  */

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

  /* MSF SuperBlock.  */
  uint32_t block_size = 0;
  /* log2 (block_size).  */
  uint32_t block_shift = 0;
  uint32_t num_blocks = 0;
  uint32_t num_dir_bytes = 0;
  uint32_t block_map_addr = 0;

  /* Stream directory.  */
  std::vector<msf_stream> streams;

  /* Owned, cached stream buffers.  These are read once at PDB load
     time and never reassigned or resized afterwards, so interior
     pointers (string_table, file_info_names_buffer, per-module
     file_name_offsets, GSI/PSI record pointers) stay valid for the
     objfile's lifetime.  */
  /* PDB_STREAM_PDB.  */
  gdb::byte_vector info_data;
  /* /names stream.  */
  gdb::byte_vector names_data;
  /* PDB_STREAM_DBI.  */
  gdb::byte_vector dbi_data;
  /* Symbol record stream.  */
  gdb::byte_vector sym_record_data;
  /* PDB file mapping.  */
  pdb_file_mapping file_mapping;

  /* The S_PUB32 public symbols at each unrelocated address; identical
     code folding (/OPT:ICF) gives several functions one address.  A
     procedure record gives only a function's natural name; its linkage
     name is that of a public at its address.  Keys are unrelocated
     because the map is filled when the PDB is loaded, before the image is
     relocated.  */
  std::unordered_map<CORE_ADDR, pdb_publics_at> pub_mangled_names;

  /* Module info.  */
  std::vector<pdb_module_info> modules;

  /* Section mapping: per-section VMA from the PE/COFF BFD, indexed by
     (CodeView section number - 1).  */
  std::vector<CORE_ADDR> section_addresses;
  /* Size of each section in section_addresses, the larger of its raw
     and virtual sizes.  */
  std::vector<bfd_size_type> section_sizes;

  /* A section contribution: SIZE bytes at OFFSET in SECTION of the image
     that module MODULE_INDEX's object file contributed, from the DBI
     Section Contribution substream.  */
  struct pdb_section_contrib
  {
    uint16_t section;
    uint16_t module_index;
    uint32_t offset;
    uint32_t size;
  };

  /* Every code contribution; empty when the substream is absent or
     unreadable.  */
  std::vector<pdb_section_contrib> code_contribs;

  /* Back-link to the symbol parser currently walking a symbol stream
     (if any).  Set for the parser's lifetime by the function that
     creates it.  */
  class pdb_sym_parser *cur_parser = nullptr;

  /* The File Info substream's Names Buffer.  It is per PDB, not per
     module, so it is read with the PDB, and points into the DBI stream,
     which stays alive.  */
  const gdb_byte *file_info_names_buffer = nullptr;
  uint32_t file_info_names_buffer_size = 0;

  /* Age from the DBI header; 0 in old PDBs.  */
  uint32_t dbi_age = 0;
  /* Symbol record stream number from DBI header.  */
  uint16_t sym_record_stream = 0;
  /* Public Symbol Information stream index.  */
  uint16_t psgsi_stream = 0;

  /* The /names stream's string table: null-terminated strings that
     records refer to by byte offset, such as the source file name of a
     CV_FileChecksum entry.  Points into names_data.  */
  const gdb_byte *string_table = nullptr;
  uint32_t string_table_size = 0;

  /* Parsed TPI stream; empty when the PDB has none or it is malformed.  */
  pdb_tpi_context tpi;

  /* Copy a stream by index out of the file mapping.  Returns an owned
     buffer.  To cache a stream for the lifetime of the objfile, move the
     result into one of the dedicated owning fields above (info_data,
     names_data, dbi_data, sym_record_data).  Returns an empty vector if
     the stream doesn't exist.  */
  gdb::byte_vector read_stream (uint32_t stream_idx) const;

  /* Map a (section, offset) pair to a relocated PC.  Returns 0 on
     failure (invalid section index, missing section addresses, or an
     OFFSET that a 32-bit section offset cannot hold, as a sum of record
     fields can be).  */
  CORE_ADDR map_section_offset_to_pc (uint16_t section,
				      uint64_t offset) const;

  CORE_ADDR map_section_offset_unrelocated (uint16_t section, uint32_t offset);

  /* True if SECTION is valid (maps to a known section address).
     Section 0 is reserved.  */
  bool section_valid (uint16_t section) const
  {
    return section != 0 && section <= section_addresses.size ();
  }

  /* True if SECTION is valid and holds the SIZE bytes at OFFSET.  */
  bool range_in_section (uint16_t section, uint32_t offset,
			 uint32_t size) const
  {
    if (!section_valid (section))
      return false;
    bfd_size_type section_size = section_sizes[section - 1];
    return offset <= section_size && size <= section_size - offset;
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
   header (offFileName, cbChecksum, ChecksumKind) is followed by cbChecksum
   checksum bytes, and the entry is padded to 4 bytes.  See microsoft-pdb
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

/* The module that contributed the code at relocated PC, from PDB's code
   section contributions, or nullptr.  A module compiled with a section
   per function (/Gy) has a contribution per function, while its header,
   and so its compunit's range, names only the first.  */
extern pdb_module_info *pdb_module_at_pc (pdb_per_objfile *pdb,
					  CORE_ADDR pc);

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
   so the caller can register discontiguous block ranges.  LANG is the
   module's language, and SYM_LINES the module's line rows, which give
   each function its source file.  */
extern void pdb_parse_symbols (pdb_per_objfile *pdb, pdb_module_info *mod_info,
			       gdb_byte *module_stream, buildsym_compunit *cu,
			       pdb_range_pair_vec *func_ranges,
			       enum language lang,
			       pdb_symbol_lines *sym_lines);

/* Return a module's language from its first S_COMPILE2/S_COMPILE3 record,
   cached in the module; C++ when there is none or its language code is
   not one GDB tells apart.  */

extern enum language pdb_module_language (pdb_per_objfile *pdb,
					  pdb_module_info *mod_info);

/* Read the symbol record stream into pdb->sym_record_data.  */
extern void pdb_read_sym_record_stream (pdb_per_objfile *pdb);

/* Build minimal symbols from S_PUB32 records in the PSGSI stream.
   Called during PDB init after sections are available.  */
extern void pdb_build_minsyms (pdb_per_objfile *pdb);

/* Read LEN bytes from stream STREAM_IDX starting at OFF, where OFF is a
   logical offset into the stream (the stream seen as one flat array).

   If the range lies within a single block a simple pointer to the file mapping
   is returned and SCRATCH is left untouched.  If the range spans multiple
   blocks, the bytes are gathered, block by block, into SCRATCH and a pointer
   into SCRATCH is returned; in that case the returned pointer equals
   scratch.data ().

   PDB->file_mapping must be mapped.  STREAM_IDX must be less than
   pdb->streams.size (): it is asserted, not checked, so a stream number
   from the file needs checking first.  A range past the end of the stream
   is an error, thrown with pdb_error.  */
extern const gdb_byte *pdb_stream_bytes (const pdb_per_objfile *pdb,
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
inline uint32_t
cv_simple_kind (uint32_t ti)
{
  return ti & 0xFF;
}

/* Extract mode (bits 8-11, of which bit 11 is reserved) from a simple
   type index.  */
inline uint32_t
cv_simple_mode (uint32_t ti)
{
  return (ti >> 8) & 0x0F;
}

/* True if TI is a simple (built-in) type index.  */
inline bool
cv_ti_is_simple (uint32_t ti)
{
  return ti < 0x1000;
}

/* SimpleTypeKind values.  */
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

/* SimpleTypeMode values.  */
/* No indirection.  */
inline constexpr auto CV_TM_DIRECT = 0;
/* 16-bit near, far and huge pointers.  */
inline constexpr auto CV_TM_NPTR = 1;
inline constexpr auto CV_TM_FPTR = 2;
inline constexpr auto CV_TM_HPTR = 3;
/* 32-bit near pointer.  */
inline constexpr auto CV_TM_NPTR32 = 4;
/* 32-bit far pointer.  */
inline constexpr auto CV_TM_FPTR32 = 5;
/* 64-bit near pointer.  */
inline constexpr auto CV_TM_NPTR64 = 6;
/* 128-bit near pointer.  */
inline constexpr auto CV_TM_NPTR128 = 7;

/* Global Symbol Streams.

   The DBI header references the streams of the global symbol
   infrastructure.  Two of them are read here:

   1. Symbol Records Stream: index at DBI_HDR_SYM_RECORD_STREAM_OFFS.
      Array of symbol records that the PSGSI stream references.
      Layout of each symbol record matches the layout of CodeView symbol
      records in module streams:
	  uint16_t reclen    : length of data after this field
	  uint16_t rectype   : CodeView record type
	  uint8_t  data[]    : record-type-specific payload
   2. PSGSI stream (DBI_HDR_PSGSI_STREAM_OFFS): Public Symbol Index
      Indexes the publicly exported symbols (S_PUB32 only).
      Embeds a GSI hash table after a metadata header.
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
	  GSI Header (GSIHashHdr in microsoft-pdb gsi.h):
	    uint32_t VerSignature  (0)  always 0xFFFFFFFF
	    uint32_t VerHdr        (4)  always 0xeffe0000 + 19990810
	    uint32_t cbHr          (8)  byte count of hash records
	    uint32_t cbBuckets     (12) byte count of bucket data
	  HashRecord[cbHr/8]:
	    uint32_t Off   (0)  1-based byte offset into SymRecordStream
	    uint32_t CRef  (4)  reference count (vestigial, always 1)
	  Bucket data (cbBuckets bytes).

	Offset 28 + SymHash+: Address map (AddrMap bytes)
	  uint32_t AddrMap[]             (sorted address-order indices)

   Usage during init (pdb_read_pdb_file):
     - pdb_read_sym_record_stream: caches SymRecordStream into
       pdb->sym_record_data.
     - pdb_build_minsyms: walks PSGSI hash records, resolves each
       S_PUB32 in sym_record_data, creates minimal_symbol entries.  */

/* On-disk size of the GSI hash header (first 4 uint32_t fields).  */
inline constexpr auto GSI_HASH_HDR_SIZE = 16;
/* VerSignature and VerHdr of the only layout read (GSIHashSCImpvV70).  */
inline constexpr uint32_t GSI_HASH_HDR_SIGNATURE = 0xFFFFFFFF;
inline constexpr uint32_t GSI_HASH_HDR_VERSION = 0xeffe0000 + 19990810;

inline constexpr auto PSGSI_HDR_SIZE = 28;
inline constexpr auto PSGSI_HDR_SYM_HASH_OFFS = 0;

inline constexpr auto GSI_HASH_RECORD_SIZE = 8;
inline constexpr auto GSI_HASH_RECORD_SYMOFFS_OFFS = 0;

/* GSI hash header (GSIHashHdr in microsoft-pdb gsi.h).
   For PSGSI, it comes after PublicsStreamHeader main header.  */
struct pdb_gsi_hdr
{
  /* VerSignature (always 0xFFFFFFFF).  */
  uint32_t sig;
  /* VerHdr.  */
  uint32_t ver;
  /* HashRecord array.  */
  gdb::array_view<gdb_byte> hr;
};

/* CodeView symbol record kinds.  */
/* Global procedure.  */
inline constexpr auto S_GPROC32 = 0x1110;
/* Local procedure.  */
inline constexpr auto S_LPROC32 = 0x110f;
/* Global procedure with ID.  */
inline constexpr auto S_GPROC32_ID = 0x1147;
/* Local procedure with ID.  */
inline constexpr auto S_LPROC32_ID = 0x1146;
/* Global data.  */
inline constexpr auto S_GDATA32 = 0x110d;
/* Local data.  */
inline constexpr auto S_LDATA32 = 0x110c;
/* Public symbol.  */
inline constexpr auto S_PUB32 = 0x110e;
/* End of scope.  */
inline constexpr auto S_END = 0x0006;
/* Closes procedure scope.  */
inline constexpr auto S_PROC_ID_END = 0x114f;
/* Thunk (opens scope).  */
inline constexpr auto S_THUNK32 = 0x1102;
/* Block scope.  */
inline constexpr auto S_BLOCK32 = 0x1103;
/* WITH scope.  */
inline constexpr auto S_WITH32 = 0x1104;
/* Managed global procedure.  */
inline constexpr auto S_GMANPROC = 0x112a;
/* Managed local procedure.  */
inline constexpr auto S_LMANPROC = 0x112b;
/* Separated code.  */
inline constexpr auto S_SEPCODE = 0x1132;
/* DPC local procedure.  */
inline constexpr auto S_LPROC32_DPC = 0x1155;
/* DPC local procedure with ID.  */
inline constexpr auto S_LPROC32_DPC_ID = 0x1156;
/* Compiler info v2.  */
inline constexpr auto S_COMPILE2 = 0x1116;
/* Compiler info v3.  */
inline constexpr auto S_COMPILE3 = 0x113c;

/* Symbol record.  Layout:
     uint16_t reclen    (length of data following reclen)
     uint16_t rectype   (symbol type)
     ...  record-type-specific data ...  */
inline constexpr auto PDB_RECORD_LEN_OFFS = 0;
inline constexpr auto PDB_RECORD_TYPE_OFFS = 2;
inline constexpr auto PDB_RECORD_DATA_OFFS = 4;
inline constexpr auto PDB_RECORD_HDR_SIZE = 4;

/* Name/field offsets in module symbol record bodies.  */
inline constexpr auto PDB_SYMBOL_FUNC_NAME_OFFS = 35;
inline constexpr auto PDB_SYMBOL_VAR_NAME_OFFS = 10;

/* CodeView type and id leaf kinds.  */

/* Virtual function table shape.  */
inline constexpr auto LF_VTSHAPE = 0x000a;
/* Label type.  */
inline constexpr auto LF_LABEL = 0x000e;

/* Procedure type.  */
inline constexpr auto LF_PROCEDURE = 0x1008;
/* Argument list for procedures.  */
inline constexpr auto LF_ARGLIST = 0x1201;
/* Field list for structs/classes.  */
inline constexpr auto LF_FIELDLIST = 0x1203;
/* Method overload list.  */
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
extern void pdb_read_tpi_stream (pdb_per_objfile *pdb);

/* Extract a NUL-terminated string from a bounded buffer [P, END).
   Returns the string pointer if a NUL byte exists in the range,
   nullptr otherwise.  Used to safely read variable-length name fields
   from PDB records without risking out-of-bounds reads.  */
const char *pdb_extract_string (const gdb_byte *p, const gdb_byte *end);

/* Resolve a TPI index to a GDB type after TPI initialization.  Unsupported
   indices/leaves return an error placeholder.  */
type &pdb_tpi_resolve_type (pdb_per_objfile *pdb, uint32_t type_index);

/* Resolve TYPE_INDEX as the type of a procedure.  An index that names no
   procedure type (none, a simple type, another kind of record) gives a
   non-prototyped function returning void, as GDB requires a function's
   symbol to have a function type.  */
type &pdb_tpi_resolve_function_type (pdb_per_objfile *pdb,
				     uint32_t type_index);

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
extern const char *pdb_canonical_name (pdb_per_objfile *pdb,
				       const char *name);

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

/* The header of the record that PSGSI hash record I of HR names in PDB's
   symbol record stream, with its offset there in *OFFSET.  A hash record
   stores the offset plus one, 0 marking an empty slot.  Return
   std::nullopt for an empty slot and, after a complaint, for a record
   that does not lie in the stream.  */

std::optional<pdb_sym_record_hdr>
pdb_psgsi_hash_record (const pdb_per_objfile *pdb,
		       gdb::array_view<const gdb_byte> hr, uint32_t i,
		       uint32_t *offset);

/* A public symbol of the PSGSI: its name, its CV_PUBSYMFLAGS, and the
   section and offset it is at.  */

struct pdb_public
{
  const char *name;
  uint32_t flags;
  uint16_t section;
  uint32_t offset;
};

/* The public that PSGSI hash record I of HR names.  Return std::nullopt
   for an empty slot or a record of another kind, and, after a complaint,
   for an S_PUB32 cut short, one whose name has no terminator or is empty,
   and one at an offset past the end of its section.  One in no section
   of the image is passed over silently.  */

std::optional<pdb_public>
pdb_psgsi_public (const pdb_per_objfile *pdb,
		  gdb::array_view<const gdb_byte> hr, uint32_t i);

#if GDB_SELF_TEST

/* The reader's self tests check what it does with the bytes of a PDB, so
   they need PDBs, and need no objfile for that: the MSF container and the
   streams in it are read from the file mapping alone.  A test builds the
   file in memory as an MSF 7.00 file holding the streams it lists, and
   reads it with the reader's own functions.  */

namespace selftests {

/* A PDB built in memory.  BLOCK_SIZE is that of the MSF file; STREAMS[I]
   is the content of stream I.  */

struct test_pdb
{
  uint32_t block_size = 4096;
  std::vector<gdb::byte_vector> streams;

  /* The file's bytes, after build.  */
  gdb::byte_vector bytes;

  /* The block that holds the stream directory, after build.  */
  uint32_t directory_block = 0;

  /* Lay the streams out in MSF 7.00 form into BYTES: the SuperBlock in
     block 0, the free block maps in blocks 1 and 2, each stream in blocks
     of its own, then the stream directory's two blocks and the block
     that lists them.  */
  void build ();

  /* Replace the stream directory with the uint32_t values WORDS, and set
     the SuperBlock's directory size to match.  */
  void set_directory (const std::vector<uint32_t> &words);
};

/* A pdb_per_objfile without an objfile, reading the bytes of a test_pdb
   as its file mapping.  */

struct test_pdb_reader
{
  explicit test_pdb_reader (const test_pdb &file);
  ~test_pdb_reader ();

  /* Read the SuperBlock and the stream directory, as loading a PDB does.
     Throws as the reader does on a malformed file.  */
  void read_msf ();

  pdb_per_objfile pdb;
};

/* An objfile with no file, as the JIT reader makes, for the reader to
   build types and compunits on: an x86-64 Windows program of NUM_SECTIONS
   sections, none relocated.  OBJFILE is null when GDB has no such
   architecture.  The objfile is removed with the object.  */

struct test_objfile
{
  explicit test_objfile (size_t num_sections);
  ~test_objfile ();

  ::objfile *objfile = nullptr;
};

/* The symbol records and C13 line data of a module, built in memory.  */

struct test_module
{
  /* Append a symbol record of KIND whose body is BODY.  */
  void symbol (uint16_t kind, const gdb::byte_vector &body);

  /* Append an S_GPROC32 named NAME, of type TYPE and CV_PROCFLAGS FLAGS,
     for the SIZE bytes at SECTION:OFFSET, and its S_END.  */
  void procedure (const char *name, uint16_t section, uint32_t offset,
		  uint32_t size, uint32_t type = 0, uint8_t flags = 0);

  /* Add a file to the checksum table, its name at NAME_OFFSET in the
     /names stream; return its file id.  */
  uint32_t file (uint32_t name_offset);

  /* Add a DEBUG_S_LINES subsection for the SIZE bytes at SECTION:OFFSET,
     one block of FILE_ID whose rows are the (offset, line) pairs ROWS.  */
  void lines (uint16_t section, uint32_t offset, uint32_t size,
	      uint32_t file_id,
	      const std::vector<std::pair<uint32_t, uint32_t>> &rows);

  /* The module stream: the symbols, then the C13 subsections, the file
     checksums first unless there are none.  Sets the symbol and C13 sizes
     of MODULE.  */
  gdb::byte_vector stream (pdb_module_info *module) const;

  /* The symbols, starting with the C13 signature.  */
  gdb::byte_vector syms { 4, 0, 0, 0 };
  gdb::byte_vector checksums;
  std::vector<gdb::byte_vector> line_subsections;
};

/* A pdb_per_objfile reading the PDB FILE for a test_objfile of
   NUM_SECTIONS sections, registered on it as for a loaded program.  Its
   MSF is read; section I + 1 is at 0x140001000 + 0x1000 * I, 0x1000
   bytes long.  PDB is null when OBJFILE is.  */

struct test_module_reader
{
  test_module_reader (const test_pdb &file, size_t num_sections);
  ~test_module_reader ();

  test_objfile obj;
  pdb_per_objfile *pdb = nullptr;
};

/* The records of a TPI or IPI stream, built in memory.  */

struct test_tpi
{
  /* Append a record of LEAF whose body is BODY, padded as the linker
     pads it; return its type index.  */
  uint32_t add (uint16_t leaf, const gdb::byte_vector &body);

  /* An LF_ARGLIST of ARGS; return its type index.  */
  uint32_t arglist (const std::vector<uint32_t> &args);

  /* An LF_PROCEDURE returning RET, with the parameters ARGLIST lists
     (COUNT of them); return its type index.  */
  uint32_t procedure (uint32_t ret, uint32_t arglist, uint16_t count);

  /* The stream: a TPI_VERSION_V80 header, then the records.  */
  gdb::byte_vector stream () const;

  gdb::byte_vector records;
  uint32_t next = 0x1000;
};

/* Read MODULES, whose streams are STREAMS, with the types of TPI, into
   compunits as expanding them does, in order, for a test_objfile of one
   section; call SETUP before reading them and CHECK after.  The files of
   the modules are named from the /names data "\0a.cpp\0b.cpp\0", at
   offsets 1 and 7.  Nothing is called when GDB has no x86-64 Windows
   architecture.  */

void test_read_modules
  (std::vector<pdb_module_info> modules,
   const std::vector<gdb::byte_vector> &streams, const test_tpi &tpi,
   gdb::function_view<void (pdb_per_objfile *pdb)> setup,
   gdb::function_view<void (pdb_per_objfile *pdb)> check);

/* Write V to BYTES at OFFSET, little-endian.  */
void put_u32 (gdb::byte_vector &bytes, size_t offset, uint32_t v);

/* Run BODY and return the message of the error it throws, without the
   "PDB Error: " prefix, or "" when it throws none.  */
std::string error_of (gdb::function_view<void ()> body);

} /* namespace selftests */

#endif /* GDB_SELF_TEST */

} /* namespace pdb */

/* GDB_PDB_PDB_INTERNAL_H.  */
#endif
