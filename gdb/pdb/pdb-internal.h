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

#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

struct buildsym_compunit;
struct objfile;

namespace pdb
{

/* Half-open PC ranges [first, second).  */
using pdb_range_pair_vec = std::vector<std::pair<CORE_ADDR, CORE_ADDR>>;

struct pdb_per_objfile;
class pdb_cooked_index;

/* A code origin is a location in the executable: a section SECT plus a byte
   offset OFF within it.

   An S_INLINESITE gives its code chunks as offsets from a base; the origin is
   that base.  */
struct pdb_code_origin
{
  uint16_t sect;
  uint32_t off;
};

/* One chunk of an inlined function, decoded from an S_INLINESITE record.
   A single inlined call can be split into several chunks.  The offsets and line
   numbers are relative, later turned into real PCs and line numbers.  */
struct pdb_inline_chunk
{
  /* Code range [start_off, end_off), as byte offsets from a base code origin
     (a pdb_code_origin selected by base_index below).  */
  uint32_t start_off;
  uint32_t end_off;

  /* This chunk's source line, as a delta from the inlinee's first line.
     The base line comes from DEBUG_S_INLINEELINES.  */
  int32_t line_delta;

  /* This chunk's source file: a DEBUG_S_FILECHKSMS offset, or
     PDB_INLINE_FILE_ID_BASE (the inlinee's base file).  */
  uint32_t file_id;

  /* Selects which base code origin start_off/end_off are measured from.
     The list of pdb_code_origin bases is passed by the caller to
     pdb_inline_chunk_ranges / pdb_record_inline_lines.  */
  uint32_t base_index;
};

/* Special value for pdb_inline_chunk::file_id (see above).  */
inline constexpr uint32_t PDB_INLINE_FILE_ID_BASE = 0xFFFFFFFF;

/* A code range of an inline site and the source line it maps to.  */
struct pdb_inline_line_span
{
  /* Relocated half-open range [START, END).  */
  CORE_ADDR start;
  CORE_ADDR end;
  int line;
  struct ::subfile *subfile;
};

/* Open scope while walking a module's symbols.  */
struct pdb_scope_frame
{
  /* End address handed to buildsym pop_context.  */
  CORE_ADDR end;
  /* Explicit half-open ranges for the block, as relocated PC pairs.
     Inline scopes can have one or several ranges.  Empty leaves the range
     chosen by pop_context unchanged.  Applied after the block is built.  */
  pdb_range_pair_vec ranges;
  /* Enclosing C++ scope of a procedure, copied onto the objfile obstack.
     Null for an unqualified procedure or a non-procedure scope.  */
  const char *scope = nullptr;
  /* For an inline site, the lines recorded for its chunks.  A site nested
     in it restores these at the end of each of its own chunks.  */
  std::vector<pdb_inline_line_span> line_spans;
};

using pdb_scope_stack = std::vector<pdb_scope_frame>;

/* Integer readers.  */

inline uint64_t
read_u64 (const void *p)
{
  return bfd_getl64 (p);
}

inline uint32_t
read_u32 (const void *p)
{
  return bfd_getl32 (p);
}

inline int32_t
read_i32 (const void *p)
{
  return static_cast<int32_t> (bfd_getl_signed_32 (p));
}

inline uint16_t
read_u16 (const void *p)
{
  return bfd_getl16 (p);
}

inline int16_t
read_i16 (const void *p)
{
  return static_cast<int16_t> (bfd_getl_signed_16 (p));
}

inline uint8_t
read_u8 (const void *p)
{
  return *static_cast<const uint8_t *> (p);
}

inline int8_t
read_i8 (const void *p)
{
  return *static_cast<const int8_t *> (p);
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

/* Reader debug verbosity: 0 = off, 1 = basic, 2 = verbose, 3 = trace.  */
extern unsigned int pdb_read_debug;
/* Path-conversion debug verbosity.  */
extern unsigned int pdb_convert_debug;
/* Quick-functions debug verbosity.  */
extern unsigned int pdb_qf_debug;

/* Force modules indexing even when GSI is present:
   "maintenance set pdb-force-module-index".  */
extern bool pdb_force_module_index;

/* Don't build <pdb-globals> eagerly at load; index globals
   and create a GDB symbol after a QF lookup.
   "maintenance set pdb-lazy-globals".  */
extern bool pdb_lazy_globals;

/* Print a "pdb" debug statement if pdb_read_debug is >= 1.  */
#define pdb_dbg_printf(fmt, ...) \
  debug_prefixed_printf_cond_func (pdb_read_debug >= 1, "pdb", nullptr, fmt, ##__VA_ARGS__)

#define pdb_dbg_printf_v(fmt, ...) \
  debug_prefixed_printf_cond_func (pdb_read_debug >= 2, "pdb", nullptr, fmt, ##__VA_ARGS__)

#define pdb_dbg_printf_t(fmt, ...) \
  debug_prefixed_printf_cond_func (pdb_read_debug >= 3, "pdb", nullptr, fmt, ##__VA_ARGS__)

/* Error — fatal issue that stops the operation.  */
#define pdb_error(fmt, ...) error (_ ("PDB Error: " fmt), ##__VA_ARGS__)

/* Warning — important one-time message that allows continuation.  */
#define pdb_warning(fmt, ...) warning (_ ("PDB Warning: " fmt), ##__VA_ARGS__)

/* Complaint — recoverable diagnostic issue (repeatable, suppressible).  */
#define pdb_complaint(fmt, ...) \
  complaint (_ ("PDB Complaint: " fmt), ##__VA_ARGS__)

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
inline constexpr auto PDB_MSF_MAGIC_OFFS = 0;

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
inline constexpr auto PDB_STREAM_IPI = 4;
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

/* Offset of the Named hash map  */;
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
/* Hash entry value size */
inline constexpr auto HASH_TABLE_VAL_SIZE = 4;
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
*/
/* Version Signature (4 bytes) */
inline constexpr auto DBI_HDR_SIGNATURE_OFFS = 0;
/* Version Header (4 bytes) */
inline constexpr auto DBI_HDR_VERSION_OFFS = 4;
/* Age (4 bytes) */
inline constexpr auto DBI_HDR_AGE_OFFS = 8;
/* Global Symbol Stream Index (2 bytes) */
inline constexpr auto DBI_HDR_GSI_STREAM_OFFS = 12;
/* Build Num. (2 bytes) */
inline constexpr auto DBI_HDR_BUILD_NUMBER_OFFS = 14;
/* Public Symbol Stream Index (2 bytes) */
inline constexpr auto DBI_HDR_PSGSI_STREAM_OFFS = 16;
/* Dll Ver. (2 bytes) */
inline constexpr auto DBI_HDR_PDB_DLL_VERSION_OFFS = 18;
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
   Stored in DBI stream at offset DBI_HDR_SOURCE_INFO_OFFS.
   Header is 4 bytes and contains the number of modules, followed by
   arrays of indices, file counts, and file offsets for each module.
      Indices array: num modules × 2 bytes
      File count array: num modules × 2 bytes
      File offset array: num modules × 2 bytes
   Each module has its slice of the file offset arrays, however, the size of
   each slice is not explicitly stored anywhere.  Instead, we have to read the
   file count for each module and sum them up to know where the next module's
   slice starts.  */
/* Total header size  */
inline constexpr auto FILE_INFO_HDR_SIZE = 4;
#define FILE_INFO_MOD_INDICES_OFFS FILE_INFO_HDR_SIZE
/* Number of modules (2 bytes).  */
inline constexpr auto FILE_INFO_HDR_NUM_MODULES_OFFS = 0;
/* Each module entry (2 bytes) */
inline constexpr auto FILE_INFO_ELEMENT_SIZE = 2;

/* Section Contribution offsets within a Module Info header.  */

/* Section identifier (2 bytes)  */
inline constexpr auto MODI_SC_ISECT_OFFS = 4;
/* Offset in the section (4 bytes) */
inline constexpr auto MODI_SC_OFFSET_OFFS = 8;
/* Size of the contribution (4 bytes) */
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
/* IMAGE_SCN_* characteristics (4 bytes).  */
inline constexpr auto DBI_SC_CHARACTERISTICS_OFFS = 12;
/* Module index, 0-based (2 bytes).  */
inline constexpr auto DBI_SC_IMOD_OFFS = 16;

/* IMAGE_SCN_CNT_CODE and IMAGE_SCN_MEM_EXECUTE.  */
inline constexpr uint32_t PDB_SCN_CODE = 0x00000020;
inline constexpr uint32_t PDB_SCN_EXECUTE = 0x20000000;

/* Module Info Header (ModInfo in microsoft-pdb) specifies the sizes
   of various debug data blocks in the corresponding module stream.
   Total size is 64 bytes plus the variable-length names.  */

/* Flags (2 bytes) */
inline constexpr auto MODI_FLAGS_OFFS = 32;
/* ModuleSymStream (2 bytes) */
inline constexpr auto MODI_STREAM_NUM_OFFS = 34;
/* Symbol stream size (4 bytes) */
inline constexpr auto MODI_SYM_BYTES_OFFS = 36;
/* C11 Bytes Size (4 bytes) */
inline constexpr auto MODI_C11_BYTES_OFFS = 40;
/* C13 Bytes Size (4 bytes) */
inline constexpr auto MODI_C13_BYTES_OFFS = 44;
/* Num. of Source files (2 bytes) */
inline constexpr auto MODI_SRC_FILE_COUNT_OFFS = 48;
/* Size of the fixed part of the header */
inline constexpr auto MODI_FIXED_SIZE = 64;

/* Module stream layout: starts with a 4-byte CV signature, then
   symbol records.  */
/* Offset past CV signature to symbols */
inline constexpr auto PDB_MODULE_SYMBOLS_OFFS = 4;

/* C13 line data.  Sizes below are in bytes.  A leading 4-byte
   CV_SIGNATURE_C13 is accepted before the subsection sequence.
   subsection[]  — sequence of subsections, each 4-byte aligned
   Subsection:
      4: type  (DEBUG_S_LINES, DEBUG_S_FILECHKSMS, ...)
	 DEBUG_S_LINES records identify the lines and map them to
	 the files in PDB /names section using DEBUG_S_FILECHKSMS records.
      4: size
      var: data (size bytes, followed by padding to 4-byte boundary)

   DEBUG_S_LINES subsection data:
      CV_LineSection header (12 bytes) — identifies the code range
      that the following file blocks map line numbers into:
	 4: offs   — starting code offset within the PE section
	 2: seci   — PE section number (1-based, e.g. .text)
	 2: flags  — bit 0: CV_LINES_HAVE_COLUMNS
	 4: code_sz — code size covered by this subsection

      CV_FileBlock[] — one or more file blocks:
	 4: file_id    — byte offset into FILECHKSMS subsection
	 4: num_lines  — number of CV_Line records
	 4: block_size — total size of this file block

      CV_Line[num_lines] (8 bytes each):
	 4: offs       — code offset relative to CV_LineSection.offs
	 4: line_start — bits 0-23: start line; bits 24-30: end-line delta;
	                 bit 31: statement (1) or expression (0)

   With CV_LINES_HAVE_COLUMNS, CV_Column[num_lines] follows the line array:
      2: start column
      2: end column
   block_size includes the file-block header, lines and any columns.  */

/* CV_LineSection flags bit: each file block's lines are followed by
   CV_Column records.  */
inline constexpr uint16_t CV_LINES_HAVE_COLUMNS = 0x0001;

/* DEBUG_S subsection types (C13 format) */

inline constexpr auto C13_SUBSECT_HEADER_SIZE = 8;

/* CV Signatures */
inline constexpr auto CV_SIGNATURE_SIZE = 4;
inline constexpr auto CV_SIGNATURE_C6 = 0L;
inline constexpr auto CV_SIGNATURE_C7 = 1L;
inline constexpr auto CV_SIGNATURE_C11 = 2L;
inline constexpr auto CV_SIGNATURE_C13 = 4L;

/* Supported C13 subsection types.  */
inline constexpr auto DEBUG_S_LINES = 0xF2;
inline constexpr auto DEBUG_S_FILECHKSMS = 0xF4;
inline constexpr auto DEBUG_S_INLINEELINES = 0xF6;

/* InlineeLines subsection signatures.  */
inline constexpr auto CV_INLINEE_SOURCE_LINE_SIGNATURE = 0x0;
inline constexpr auto CV_INLINEE_SOURCE_LINE_SIGNATURE_EX = 0x1;

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
    3. For each entry of 'set debug-file-directory':
       <entry>/<PDB basename from RSDS>.
    4. Full embedded PDB path from the RSDS record, as-is.
    5. Search path from _NT_ALT_SYMBOL_PATH then _NT_SYMBOL_PATH.
       See https://learn.microsoft.com/en-us/windows/win32/debug/symbol-paths
       and microsoft-pdb locator.cpp for path expansion details.
       TODO: SRV, SYMSRV, CACHE entries need symbol server fetch via HTTP.
    6. Windows registry SymbolSearchPath.  Same docs as above.
       TODO: SRV, SYMSRV, CACHE entries need symbol server fetch via HTTP.  */
extern std::string pdb_find_pdb_file (const objfile *objfile,
				      const pdb_rsds_info &rsds);

/* Per-file info from DEBUG_S_FILECHKSMS.  */
struct pdb_file_info
{
  /* Filename borrowed from names_data; null if unresolved.  */
  const char *filename;
  /* Checksum kind: 0=none, 1=MD5, 2=SHA1, 3=SHA256.  */
  uint8_t checksum_type;
  /* Valid checksum length in bytes.  */
  uint8_t checksum_size;
  /* Checksum borrowed from the module's obstack-backed file_checksums.  */
  const gdb_byte *checksum;
};

/* PDB Module information */
struct pdb_module_info
{
  /* Module name borrowed from pdb_per_objfile::dbi_data.  */
  char *module_name;
  /* Object filename borrowed from pdb_per_objfile::dbi_data.  */
  char *obj_file_name;

  /* Module debug stream index; 0xFFFF means no stream.  */
  uint16_t stream_number;
  /* Symbol-area byte count, including the leading CodeView signature.  */
  uint32_t sym_byte_size;
  /* Size of C11 records in the module stream.  */
  uint32_t c11_byte_size;
  /* Size of C13 records in the module stream.  */
  uint32_t c13_byte_size;

  uint16_t flags;

  /* One Module Info section contribution, not necessarily the module's
     full address coverage.  1-based section index; 0 means none.  */
  uint16_t sc_section;
  /* Offset within section.  */
  uint32_t sc_offset;
  /* Size of contribution in bytes.  */
  uint32_t sc_size;

  /* First DEBUG_S_FILECHKSMS copied onto the objfile obstack; null until
     read or when absent.  */
  gdb_byte *file_checksums;
  uint32_t file_checksums_size;

  /* DEBUG_S_INLINEELINES subsection (copied onto objfile obstack).  Maps each
     inlined function's IPI id to its source file and base line; nullptr when
   the subsection has not been read or is absent.  */
  gdb_byte *inlinee_lines;
  uint32_t inlinee_lines_size;

  /* Set before expansion to prevent reentry.  Remains set if no CU is
     produced, so it is not a completion test.  */
  bool expanded;
  compunit_symtab *cu;

  /* Source language of this module, cached on first use.  */
  enum language language = language_unknown;

  /* Pointers to the files in the Names Buffer (in pdb_per_objfile) that belong
     to this module.  These don't have to be loaded lazily since the Names
     buffer is already loaded.  The work is in parsing the module offset array
     (file_name_offsets below) from which the file pointers are assigned; that
     array is parsed eagerly because the offset calculations are easier to do
     for all modules at once than per module.
     TODO: assign the file pointers eagerly too once lazy loading is complete.  */
  uint16_t num_files;
  const char **files;

  /* Offset array provides the offsets into the Names Buffer for each file
     in this module.  Populated during the initialization - this is because
     the location of each module's offset slice depends on the sizes of the
     slices before it.  */
  const gdb_byte *file_name_offsets;

  /* C13 file list for maintenance info pdb-files-c13.  The entry array
     uses the objfile obstack and borrows names/checksums via pdb_file_info.
     Its count is not limited to a 16-bit field.  */
  bool files_c13_read;
  pdb_file_info *files_c13;
  uint32_t num_files_c13;
};

/* Indexed raw TPI/IPI record.  Payload bytes borrow the PDB mapping or an
   objfile-obstack copy when the payload spans MSF blocks.  */
struct pdb_tpi_type
{
  /* CodeView record type (LF_*).  */
  uint16_t leaf;
   /* RecordLen includes RecordKind and payload, excluding the length word.  */
  uint16_t length;
   /* Payload after RecordKind, excluding both header words.  */
  const gdb_byte *data;
  /* Size of data (length - 2 for RecordKind).  */
  uint32_t data_len;
};

/* Counts and timing from pdb_build_parent_map.  */

struct pdb_nesting_stats
{
   /* Tagged records with a nonzero fieldlist reference.  */
  size_t tags_scanned = 0;
  size_t fieldlists_walked = 0;
  size_t subrecords_walked = 0;
   /* Additional direct tag references to an already-owned fieldlist.  */
  size_t shared_fieldlists = 0;
   /* All fieldlists with no selected owner, not only continuations.  */
  size_t unowned_continuations = 0;
  size_t edges = 0;
  /* Inserted edges from fieldlists directly referenced by several tags;
     the first referring tag is retained as owner.  */
  size_t ambiguous_edges = 0;
   /* Named nonzero LF_NESTTYPE bindings rejected by the owner/name check.  */
  size_t alias_edges = 0;
  size_t static_members = 0;
  double build_ms = 0;
};

/* TPI/IPI record index.  IPI uses the array and index bounds only; the
   GDB type cache and name/ownership maps belong to TPI.  */
struct pdb_tpi_context
{
   /* Objfile-obstack array; TI N indexes types[N - type_idx_begin].  */
  pdb_tpi_type *types = nullptr;
   /* First record index in the stream.  */
  uint32_t type_idx_begin = 0;
  /* One past last type index.  */
  uint32_t type_idx_end = 0;

  /* Objfile-obstack array indexed by TI, covering
     [0, max(0x1000, type_idx_end)).  A forward record with a definition
     has no entry of its own; its references use the definition's.  An
     entry can be a forward stub for a tag the TPI never defines, or, while
     members are being read, a struct or union shell whose members are
     still pending.  */
  type **type_cache = nullptr;

  /* Type used for unsupported types.  */
  type *undefined_type = nullptr;

  /* Canonical tag spelling (LF_CLASS/STRUCTURE/UNION/ENUM) -> one TI.
     A forward record never displaces an existing entry; the last
     non-forward record wins.  Keys borrow record names or persistent
     rewritten names, whose storage must outlive this context.  */
  std::unordered_map<std::string_view, uint32_t> tagged_type_names;

  /* Forward record -> definition TI, indexed by TI - type_idx_begin; 0
     for other records and for tags the TPI never defines.  A forward
     record carries only the tag's name, so the definition is the one
     tagged_type_names selects for that name.  */
  std::vector<uint32_t> fwdref_definition;

  /* Struct and union shells whose field lists are not read yet, keyed by
     the shell, with the field list TI.  A shell has its name, size and
     kind but no members.  */
  std::unordered_map<type *, uint32_t> deferred_structs;

  /* The keys of deferred_structs in creation order.  A key already read
     through a base-class reference is no longer in the map.  */
  std::vector<type *> deferred_struct_order;

  /* Nonzero while a field list is being read.  Structs and unions built at
     that depth are shells; the outermost pdb_tpi_resolve_type reads their
     members before returning.  */
  int struct_read_depth = 0;

  /* Records being built, each with the struct_read_depth at which its
     build started.  Reaching one again at the same depth means the
     records refer to each other in a cycle.  */
  std::vector<std::pair<uint32_t, int>> records_in_progress;

  /* Method type -> " &" or " &&" when its 'this' pointer carries the
     LF_POINTER islref or isrref bit.  GDB's method type has no field for
     this ref-qualifier.  */
  std::unordered_map<const type *, const char *> method_ref_qualifier;

  /* Enumerator name -> canonical tag name of an LF_ENUM that declares it,
     one entry per declaring enum.  An enumerator has no record of its own;
     it exists only as an LF_ENUMERATE inside the enum's field list, so a
     by-name lookup for one cannot find anything until that enum has been
     built.  */
  std::unordered_multimap<std::string_view, const char *> enumerator_names;

  /* Nested TI -> selected declaring tag TI.  pdb_build_parent_map accepts
     exact "owner::nested" raw-name matches, excluding aliases to other
     tags.  Forward and definition TIs remain separate keys.  */
  std::unordered_map<uint32_t, uint32_t> nested_owner;

  /* "Tag::member" -> selected declaring tag TI for static data members.
     Built with nested_owner from raw tag/member spellings.  This map owns
     its keys; namespace inference queries it by symbol name.  */
  std::unordered_map<std::string, uint32_t> static_member_owner;

  /* What building nested_owner cost, for maintenance info pdb-nesting.  */
  pdb_nesting_stats nesting_stats;

  /* Parsed fields reusable by tags sharing an LF_FIELDLIST.  Static-member
     physnames stay bare and are qualified on each owner's copy.  Lists
     with methods or a vptr are excluded.  Names borrow record data and
     referenced types use GDB type storage.  */
  struct pdb_fieldlist_parse
  {
    std::vector<field> baseclasses;
    std::vector<field> fields;
    std::vector<decl_field> nested_types;
  };
  std::unordered_map<uint32_t, pdb_fieldlist_parse> fieldlist_cache;
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
  explicit pdb_per_objfile (::objfile *objfile);
  ~pdb_per_objfile ();

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
  uint32_t free_block_map_block;
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

  /* Map an unrelocated address to the mangled name of the S_PUB32 public
     symbol at that address.  A module stream's S_GPROC32 / S_GDATA32 carries
     only a readable name for a symbol while S_PUB32 carries only the mangled
     name. This map joins the names for an address.
     Keys are unrelocated because they are filled at load time before the image
     has been relocated.  Values point into symbol names in sym_record_data.  */
  std::unordered_map<CORE_ADDR, const char *> pub_mangled_names;

  /* Cooked index owned by this context for name/address lookup to modules.  */
  std::unique_ptr<pdb_cooked_index> cooked_index;

  /* Names of tagged types (LF_CLASS/STRUCTURE/UNION/ENUM) already
   built as <pdb-types> GDB symbols.  A tagged type is built
   the first time it is needed: QF name lookup in type domain, or when
   all the CUs are expanded (readnow or using QFs).  Elements point at
   the keys of tagged_type_names.  */
  std::unordered_set<std::string_view> built_type_names;

  /* Set when the full tagged-type sweep runs, so it runs once and the
     by-name path then has nothing left to build.  */
  bool all_tagged_types_built = false;

  /* Scopes of tagged-type names already given a namespace symbol in
     <pdb-types>.  Keys are owned: a scope is a substring of a name.  */
  std::unordered_set<std::string> built_namespace_names;

  /* An entry in the lazy-global name index.  */
  struct pdb_lazy_global
  {
    /* Name borrowed from sym_record_data.  */
    const char *name;
    /* Record-header byte offset in sym_record_data.  */
    uint32_t sym_offset;
    uint16_t rectype;
  };
  std::vector<pdb_lazy_global> lazy_globals;

  /* Names already built as GDB symbols in <pdb-globals>, so the per-name
     and build-all paths each add a name only once.
     Keys point into sym_record_data.  */
  std::unordered_set<std::string_view> built_global_names;

  /* Names already bound by a global S_UDT.  The linker can leave several
     global S_UDT records with one name and different type indices (a TU's
     own typedef plus the CRT's), and a global block holds one binding per
     name, so without this the winner would depend on how many records the
     mode happened to build.  Keys point into sym_record_data.  */
  std::unordered_set<std::string_view> global_udt_names;

  /* Set once every lazy global has been built.  */
  bool all_globals_built = false;

  /* Module info */
  std::vector<pdb_module_info> modules;

  /* A code contribution from the DBI Section Contribution substream.  */
  struct pdb_code_contrib
  {
    uint16_t section;
    uint16_t module_index;
    uint32_t offset;
    uint32_t size;
  };

  /* Every code contribution with its module; empty when the substream is
     absent or unreadable.  */
  std::vector<pdb_code_contrib> code_contribs;

  /* Section mapping: per-section VMA from the PE/COFF BFD, indexed by
     (CodeView section number - 1).  */
  std::vector<CORE_ADDR> section_addresses;

  /* Back-link to the symbol parser currently walking a symbol stream
     (if any).  Set by pdb_sym_parser's ctor/dtor.  Used by helpers
     like cv_reg_to_gdb_regnum to reach the live frame register without
     it being threaded through every signature.  */
  class pdb_sym_parser *cur_parser = nullptr;

  /* Loading File Info Substream details:
     This substream is per PDB and not per module, thus there is not much sense
     loading it lazily - any command like break or info sources will try to
     expand that module which will require this substream to be loaded.  So, we
     can just load it eagerly.  Since the File Info substream is the larger
     part of the DBI stream, we keep the file info substream as a pointer into
     DBI stream which is kept alive.  */
  gdb_byte *file_info_names_buffer;
  uint32_t file_info_names_buffer_size;

  /* Symbol record stream number from DBI header.  */
  uint16_t sym_record_stream;
  /* Global Symbol Information stream index.  */
  uint16_t gsi_stream;
  /* Public Symbol Information stream index.  */
  uint16_t psgsi_stream;

  /* TODO: free after PSI/GSI load.  Blocked: GSI records and the
     sym-records dump command both point into this buffer.  */

  /* Global string table from /names stream.  It is a simple array
     of null-terminated strings; names are accessed by offset in
     CV_FileChecksum which is how the records are stored in the PDB.
     Points into names_data.  */
  gdb_byte *string_table;
  uint32_t string_table_size;

  /* Parsed TPI & IPI streams.  Check tpi.types to verify parsing succeeded.  */
  pdb_tpi_context tpi;

  /* Parsed IPI (id) stream.  Holds LF_FUNC_ID / LF_MFUNC_ID records that name
     inlined functions; ipi.types is null when the stream is absent.  */
  pdb_tpi_context ipi;

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
  CORE_ADDR section_reloc (uint16_t section) const;

  /* True if SECTION is valid (maps to a known section address).
     Section 0 is reserved.  */
  bool section_valid (uint16_t section) const
  {
    return section != 0 && section <= section_addresses.size ();
  }

  /* Get the BFD section name for a CodeView section number.
     Returns "(none)" for 0 and "(unknown)" if out of range.  */
  const char *get_section_name (uint16_t sect_num) const;
};

/* C13 DEBUG_S_LINES subsection header (12 bytes).
   See microsoft-pdb cvinfo.h.  */
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

/* C13 file block header (12 bytes), one per file inside a DEBUG_S_LINES
   subsection.  See microsoft-pdb cvinfo.h.  */
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

/* C13 file checksum entry header (6 bytes), one per file in a
   DEBUG_S_FILECHKSMS subsection.  See microsoft-pdb cvinfo.h.
   CHECKSUM points at the checksum_size bytes that follow the header in
   the source buffer (nullptr if checksum_size == 0).  */
struct CV_FileChecksum
{
  static constexpr uint32_t HDR_SIZE = 6;

  explicit CV_FileChecksum (const gdb_byte *p)
    : file_name_offs (read_u32 (p + 0)),
      checksum_size (read_u8 (p + 4)),
      checksum_type (read_u8 (p + 5)),
      checksum (checksum_size != 0 ? p + HDR_SIZE : nullptr)
  {
  }

  uint32_t file_name_offs;
  uint8_t checksum_size;
  uint8_t checksum_type;
  const gdb_byte *checksum;
};

/* C13 line entry (8 bytes).  See microsoft-pdb cvinfo.h.  The
   constructor unpacks the bit-packed line/flags word.  */
struct CV_Line
{
  static constexpr uint32_t SIZE = 8;

  explicit CV_Line (const gdb_byte *p)
    : offs (read_u32 (p + 0)),
      line_start (read_u32 (p + 4) & 0x00FFFFFF),
      delta_end ((read_u32 (p + 4) >> 24) & 0x7F),
      is_statement (((read_u32 (p + 4) >> 31) & 0x1) != 0)
  {
  }

  uint32_t offs;
  uint32_t line_start;
  uint32_t delta_end;
  bool is_statement;
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

/* Expand module into its own compunit_symtab.  Returns the cached CU on
   subsequent calls.  Returns nullptr if the module has no useful debug
   data.  */
extern compunit_symtab *pdb_build_module (pdb_per_objfile *pdb,
					  pdb_module_info *mod);

/* Flags for pdb_parse_symbols.  */
enum pdb_sym_flag : unsigned
{
  /* Dump records (GDB symbols not created).  */
  PDB_DUMP_SYM = 1 << 0,
};
DEF_ENUM_FLAGS_TYPE (enum pdb_sym_flag, pdb_sym_flags);

/* Parse CodeView symbol records from a module stream into GDB symbols.
   MODULE_STREAM is the raw module stream data (caller owns it).
   When flags includes PDB_DUMP_SYM, records are dumped to stdout
   instead of creating GDB symbols.
   If FUNC_RANGES is non-null, each function's [start, end) is appended
   so the caller can register discontiguous block ranges.  */
/* Declaration lines for function symbols, taken from the C13 line table.
   A line number means nothing without the file it counts within, so each
   symbol is recorded with the subfile its entry PC belonged to and both
   are applied together once that subfile has a symtab -- the pairing
   new_symbol_file_line makes for DWARF.  */

struct pdb_symbol_lines
{
  struct site
  {
    int line;
    struct ::subfile *subfile;
  };

  struct pending
  {
    struct symbol *sym;
    struct ::subfile *subfile;
    int line;
    /* Also set SYM's line, not just its symtab.  */
    bool with_line;
  };

  /* Procedure entry PC -> the line and file recorded there.  */
  std::unordered_map<CORE_ADDR, site> at_pc;

  /* Symbols awaiting their line, applied after the compunit is built.  */
  std::vector<pending> to_apply;

  /* Every row of the module, across all files.  Collected first and emitted
     in address order, so a file's run can be closed where the next file's
     code starts.  */
  struct row
  {
    CORE_ADDR pc;
    int line;
    struct ::subfile *subfile;
    linetable_entry_flags flags;
  };
  std::vector<row> rows;
};

extern void pdb_parse_symbols (pdb_per_objfile *pdb, pdb_module_info *mod_info,
			       gdb_byte *module_stream, buildsym_compunit *cu,
			       pdb_sym_flags flags,
			       pdb_range_pair_vec *func_ranges,
			       enum language lang,
			       pdb_symbol_lines *sym_lines = nullptr);

/* Where a procedure's prologue ends.  INFERRED tells the line table that PC
   came from the parameters' location ranges rather than from DbgStart, so it
   must not displace a real line entry.  */
struct pdb_prologue_end
{
  CORE_ADDR pc;
  bool inferred;
};

/* Map each procedure's start PC to the PC just past its prologue, from the
   DbgStart field of S_GPROC32/S_LPROC32, or from where its parameters first
   become readable when DbgStart is absent.  Procedures with neither are
   omitted.  The line table needs this before symbols are parsed, so it is a
   separate scan of MODULE_STREAM rather than a by-product of parsing.  */
extern void pdb_collect_prologue_ends (pdb_per_objfile *pdb,
				       pdb_module_info *mod_info,
				       gdb_byte *module_stream,
				       std::unordered_map<CORE_ADDR,
							  pdb_prologue_end> *out);

/* Find MOD_INFO's source language by scanning its symbols for the first
   S_COMPILE2/S_COMPILE3 record, caching it in mod_info->language.  Missing
   records and unrecognized language codes default to language_cplus.
   Pass an existing MODULE_STREAM to avoid rereading it; with nullptr,
   this function reads the stream itself.  */

extern enum language pdb_module_language (pdb_per_objfile *pdb,
					  pdb_module_info *mod_info,
					  const gdb_byte *module_stream
					    = nullptr);

/* Return MOD_INFO's compiler version string from its S_COMPILE2/S_COMPILE3
   record, allocated on the objfile obstack, or nullptr when absent.  GDB
   gates several compiler-specific workarounds on this.  */
extern const char *pdb_module_producer (pdb_per_objfile *pdb,
					pdb_module_info *mod_info,
					const gdb_byte *module_stream);

/* Read the symbol record stream into pdb->sym_record_data.  */
extern void pdb_read_sym_record_stream (pdb_per_objfile *pdb);

/* Create a <pdb-globals> CU, load global symbols into it via
   pdb_load_global_syms, and finalize the compunit.  */
extern void pdb_load_global_syms_cu (pdb_per_objfile *pdb);

/* Build the name -> SymRecordStream offset index (pdb->lazy_globals) so
   globals can be built lazily during QF lookups.  */
extern void pdb_build_lazy_globals_index (pdb_per_objfile *pdb);

/* Give each namespace that encloses a lazy global a namespace symbol, as
   building the globals eagerly would.  */
extern void pdb_register_global_namespaces (pdb_per_objfile *pdb);

/* Build the global(s) named NAME (from pdb->lazy_globals index) into the
   <pdb-globals> CU.  */
extern void pdb_build_global (pdb_per_objfile *pdb, std::string_view name);

/* Build all remaining globals.  */
extern void pdb_build_all_globals (pdb_per_objfile *pdb);

/* Dump records from the global symbol record stream selected by DBI.
   Records include S_PUB32 and other global kinds.  FLAGS is unused.  */

extern void pdb_parse_sym_record_stream (pdb_per_objfile *pdb,
					 uint32_t = 0);

/* Dump GSI hash records — reads the GSI stream and resolves each hash
   record's offset into the SymRecordStream, dumping the referenced symbol.  */
extern void pdb_dump_gsi_stream (pdb_per_objfile *pdb);

/* Half-open gap [start, end) in one entry, in linked PCs before relocation.
   Another entry may still describe the variable at these addresses.  */
struct pdb_loc_gap
{
  /* Start of gap.  */
  CORE_ADDR start;
  /* End of gap.  */
  CORE_ADDR end;
};

/* A gap as encoded in CodeView S_DEFRANGE_* records: relative byte
   offset from the defrange's start, and the gap's byte length.
   Converted to linked PC bounds by pdb_add_loc_entry.  */
struct pdb_defrange_gap
{
  uint16_t offset;
  uint16_t length;
};

/* One parsed location from a DEFRANGE or a full-scope register record.
   The entry and its trailing gap array use the objfile obstack.  PC bounds
   are stored as linked; selection applies the current section relocation.  */
struct pdb_loc_entry
{
  pdb_loc_entry *next;
  /* Start PC as linked, before relocation (ignored when is_full_scope).  */
  CORE_ADDR start;
  /* End PC as linked, before relocation (ignored when is_full_scope).  */
  CORE_ADDR end;
   /* 1-based section for relocation of START, END and gaps at lookup.  */
  uint16_t section;
  /* GDB register number (-1 = unsupported).  */
  int gdb_regnum;
  /* Byte offset from register.  */
  int32_t offset;
  /* Value is the register itself (not memory).  */
  bool is_register;
  /* When set, the entry is valid for the whole enclosing function.  */
  bool is_full_scope;
  /* Number of gaps in gaps[] (ignored when is_full_scope).  */
  int num_gaps;
  /* Inline gap array.  */
  pdb_loc_gap gaps[];
};

/* Per-symbol location baton.
   Contains a linked list of parsed location entries, each fully
   resolved at parse time.  At read time we walk the list and find
   the entry matching the current PC.  Allocated on the objfile obstack.  */
struct pdb_loclist_baton
{
  /* Linked list of parsed location entries (nullptr if none).  */
  pdb_loc_entry *entries;

  /* Context for current section relocation, not owned by this baton.  */
  pdb_per_objfile *pdb;

  /* The location holds the address of the value, not the value.  */
  bool deref;
};

/* Return true if SYM has a PDB-computed location (pdb_loclist_funcs).  */
extern bool pdb_is_pdb_location (const symbol *sym);

/* Which entry of BATON describes the variable at PC, or nullptr when none
   does.  *GAPPED reports that a range covering PC was interrupted there,
   which no whole-scope entry overrides.  */
extern const pdb_loc_entry *pdb_loclist_select (const pdb_loclist_baton *baton,
						CORE_ADDR pc, bool *gapped);

/* Dump resolved variable location batons from a built compunit_symtab.
   Walks all blocks, finds symbols with pdb_loclist_baton, and prints
   each pdb_loc_entry (range, register, offset, gaps).
   If SYMBOL_FILTER is non-null, only prints the matching symbol.
   With HAVE_PC, reports the entry answering at SEL_PC instead.  */
extern void pdb_dump_locations (compunit_symtab *cust, gdbarch *gdbarch,
				const char *symbol_filter, bool have_pc,
				CORE_ADDR sel_pc);

/* Parse and optionally dump a single symbol record from the SymRecordStream.
   If PDB_DUMP_SYM is set in FLAGS, calls the record's dump() method.  */
extern void pdb_dump_parse_record (pdb_per_objfile *pdb, gdb_byte *rec_data,
				   uint16_t rectype, uint16_t reclen,
				   pdb_sym_flags flags);

/* Dump PSGSI (public symbol) hash records — reads the PSGSI stream and
   resolves each hash record into the SymRecordStream.  */
extern void pdb_dump_psgsi_stream (pdb_per_objfile *pdb);

/* Build minimal symbols from S_PUB32 records in the PSI stream.
   Called during PDB init after sections are available.  */
extern void pdb_build_minsyms (pdb_per_objfile *pdb);

/* Map the whole PDB file read-only.
   Returns true on success.  */
extern bool pdb_map_file (pdb_per_objfile *pdb);

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

/* Index type streams. Make a pass over a type stream by reading REC_BYTES bytes
   of records starting at REC_START_OFF and fill tpi.types[] with one entry per
   record contain the TI details (leaf kind, length, and a pointer to the
   record's data.  */
extern void pdb_parse_tpi_records (pdb_per_objfile *pdb, uint32_t stream_idx,
				   uint32_t rec_start_off, uint32_t rec_bytes,
				   const char *stream_name,
				   pdb_tpi_context &tpi);

/* Install the lazy cooked-index quick_symbol_functions on OBJFILE.
   Build globals eagerly (or their index with pdb_lazy_globals), start the
   cooked-index build, then register the lazy reader.  Tagged-type symbols
   remain on demand; globals may resolve their own referenced types.  */

extern void pdb_install_cooked_index (objfile *objfile, pdb_per_objfile *pdb);

/* Get module's files.  Parses the DBI File Info substream and resolves
   file names into module->files[].  Populates module->num_files and
   module->files on first call; subsequent calls are no-ops.  */
extern void pdb_read_module_files (pdb_per_objfile *pdb, pdb_module_info *mod);

/* Read a module's debug stream and extract file checksums onto the
   objfile obstack.  Returns an owned buffer with the raw stream data;
   the caller keeps it alive as long as the data is needed.  Returns
   an empty vector if the module has no stream.  */
extern gdb::byte_vector pdb_read_module_stream (pdb_per_objfile *pdb,
						pdb_module_info *mod);

/* Resolve the i-th filename for a module from the shared File Info
   substream.  Returns nullptr if index is out of range or the substream
   is not loaded.  Caller must call pdb_read_module_files first.  */
extern const char *pdb_module_file_name (pdb_per_objfile *pdb,
					 const pdb_module_info *mod,
					 uint32_t index);

/* Resolve a DEBUG_S_FILECHKSMS file id to a source filename via the /names
   string table.  Returns nullptr if MODULE has no checksums or FILE_ID is
   out of range.  */
const char *pdb_get_filename_from_file_id (const pdb_per_objfile *pdb,
					   pdb_module_info *module,
					   uint32_t file_id);

/* TODO: Symbol server fetcher */
std::string pdb_symserver_fetch (const std::string &entry,
				 const std::string &pdb_name,
				 const pdb_rsds_info &rsds);

/* Try to open PATH for reading.  Returns true if the file exists.  */
bool pdb_try_open (const char *path);

/* Extract the directory portion of PATH (including trailing separator).
   Returns empty string if PATH has no directory component.  */
std::string pdb_dirname (const char *path);

/* Convert a Unix/MSYS2-style PATH to a native path.  A path that is not
   Unix-style is returned unchanged.  */
std::string pdb_convert_path (const char *path);

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
/* Near pointer.  */
inline constexpr auto CV_TM_NPTR = 1;
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

   2. Global Symbol Index (GSI stream) - Index at DBI_HDR_GSI_STREAM_OFFS.
      GSI is a hash table that maps the symbol name to the location in the
      module symbol records where the symbol is fully described.  As such,
      GSI can be used as cooked index.
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
       S_PUB32 in sym_record_data, creates minimal_symbol entries.
     - the GSI also seeds the lazy cooked index (see pdb-index.c).  */
/* On-disk size of the GSI hash header (first 4 uint32_t fields).  */

inline constexpr auto GSI_HASH_HDR_SIZE = 16;

inline constexpr auto PSGSI_HDR_SIZE = 28;
inline constexpr auto PSGSI_HDR_SYM_HASH_OFFS = 0;
inline constexpr auto PSGSI_HDR_ADDR_MAP_OFFS = 4;
//#define PSGSI_HDR_NUM_THUNKS_OFFS   8
//#define PSGSI_HDR_SIZE_OF_THUNK_OFFS 12
//#define PSGSI_HDR_ISECT_THUNK_TABLE_OFFS 16
//#define PSGSI_HDR_PADDING_OFFS       18
//#define PSGSI_HDR_OFF_THUNK_TABLE_OFFS 20
//#define PSGSI_HDR_NUM_SECTIONS_OFFS 24

inline constexpr auto GSI_HASH_HDR_SIG_VAL = 0xFFFFFFFF;
#define GSI_HASH_HDR_VER_VAL (0xeffe0000 + 19990810)
inline constexpr auto GSI_HASH_RECORD_SIZE = 8;
inline constexpr auto GSI_HASH_RECORD_SYMOFFS_OFFS = 0;
//#define GSI_HASH_RECORD_CREF_OFFS   4

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
  /* Bucket data (after records).  */
  gdb::array_view<gdb_byte> bucket;
};

/* Validate and parse a GSI hash region at DATA with DATA_SIZE bytes.
   Returns the parsed header on success, std::nullopt on failure.  The data
   pointer must point to the GSIHashHdr (i.e. past any PublicsStreamHeader).  */
extern std::optional<pdb_gsi_hdr>
pdb_parse_gsi_hash_header (gdb_byte *data, uint32_t data_size);

/* GSI hash bucket constants (microsoft-pdb GSI1.h).  */
/* IPHR_HASH + 1.  */
inline constexpr auto GSI_NUM_BUCKETS = 4097;
/* ceil(GSI_NUM_BUCKETS / 32).  */
inline constexpr auto GSI_BITMAP_WORDS = 129;

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
/* Local variable */
inline constexpr auto S_LOCAL = 0x113e;
/* Local var (same as S_LOCAL) */
inline constexpr auto S_LOCAL32 = 0x113e;
/* User-defined type */
inline constexpr auto S_UDT = 0x1108;
/* Block scope */
inline constexpr auto S_BLOCK32 = 0x1103;
/* Register-relative var.  */
inline constexpr auto S_REGREL32 = 0x1111;
/* Public symbol */
inline constexpr auto S_PUB32 = 0x110e;
/* Register variable */
inline constexpr auto S_REGISTER = 0x1106;
/* Constant value */
inline constexpr auto S_CONSTANT = 0x1107;
/* Code label */
inline constexpr auto S_LABEL32 = 0x1105;
/* End of scope */
inline constexpr auto S_END = 0x0006;
/* Opens inlined function scope  */
inline constexpr auto S_INLINESITE = 0x114d;
/* Close inlined func. scope  */
inline constexpr auto S_INLINESITE_END = 0x114e;
/* Closes procedure scope */
inline constexpr auto S_PROC_ID_END = 0x114f;
/* Thunk (opens scope) */
inline constexpr auto S_THUNK32 = 0x1102;
/* Reference to procedure  */
inline constexpr auto S_PROCREF = 0x1125;
/* Reference to local procedure  */
inline constexpr auto S_LPROCREF = 0x1127;

/* CodeView symbol types recognised only well enough to dump.  */
/* Object file name */
inline constexpr auto S_OBJNAME = 0x1101;
/* Compiler info v2 */
inline constexpr auto S_COMPILE2 = 0x1116;
/* Reference to data in module stream */
inline constexpr auto S_DATAREF = 0x1126;
/* Reference to annotation */
inline constexpr auto S_ANNOTATIONREF = 0x1128;
/* Compiler info v3 */
inline constexpr auto S_COMPILE3 = 0x113c;
/* Environment block (key/value pairs) */
inline constexpr auto S_ENVBLOCK = 0x113d;
/* Using namespace */
inline constexpr auto S_UNAMESPACE = 0x1124;
/* BP-relative variable */
inline constexpr auto S_BPREL32 = 0x110b;
/* Local thread-local storage */
inline constexpr auto S_LTHREAD32 = 0x1112;
/* Global thread-local storage */
inline constexpr auto S_GTHREAD32 = 0x1113;
/* Managed local data */
inline constexpr auto S_LMANDATA = 0x111c;
/* Managed global data */
inline constexpr auto S_GMANDATA = 0x111d;
/* Build information */
inline constexpr auto S_BUILDINFO = 0x114c;
/* Frame procedure info */
inline constexpr auto S_FRAMEPROC = 0x1012;
/* Call site information */
inline constexpr auto S_CALLSITEINFO = 0x1139;
/* File-scoped static */
inline constexpr auto S_FILESTATIC = 0x1153;
/* Exported symbol */
inline constexpr auto S_EXPORT = 0x1138;
/* PE section info */
inline constexpr auto S_SECTION = 0x1136;
/* COFF group info */
inline constexpr auto S_COFFGROUP = 0x1137;
/* Trampoline thunk */
inline constexpr auto S_TRAMPOLINE = 0x112c;
/* Security cookie on stack frame */
inline constexpr auto S_FRAMECOOKIE = 0x113a;
/* Heap allocation site */
inline constexpr auto S_HEAPALLOCSITE = 0x115e;
/* Callee list */
inline constexpr auto S_CALLEES = 0x115a;
/* Caller list */
inline constexpr auto S_CALLERS = 0x115b;
/* Profile-guided optimization data */
inline constexpr auto S_POGODATA = 0x115c;
/* Inline site v2 */
inline constexpr auto S_INLINESITE2 = 0x115d;
/* Managed token reference */
inline constexpr auto S_TOKENREF = 0x1129;
/* Managed global procedure */
inline constexpr auto S_GMANPROC = 0x112a;
/* Managed local procedure */
inline constexpr auto S_LMANPROC = 0x112b;
/* COBOL UDT */
inline constexpr auto S_COBOLUDT = 0x1109;
/* Managed constant */
inline constexpr auto S_MANCONSTANT = 0x112d;
/* Separated code */
inline constexpr auto S_SEPCODE = 0x1132;
/* Discarded symbol */
inline constexpr auto S_DISCARDED = 0x113b;
/* Annotation */
inline constexpr auto S_ANNOTATION = 0x1019;
/* Def range */
inline constexpr auto S_DEFRANGE = 0x113f;
/* Def range subfield */
inline constexpr auto S_DEFRANGE_SUBFIELD = 0x1140;
/* Local variable (2005 format) */
inline constexpr auto S_LOCAL_2005 = 0x1133;
/* Def range (2005 format) */
inline constexpr auto S_DEFRANGE_2005 = 0x1134;
/* Def range v2 (2005 format) */
inline constexpr auto S_DEFRANGE2_2005 = 0x1135;
/* ARM switch table / jump table */
inline constexpr auto S_ARMSWITCHTABLE = 0x1159;
/* Module type reference */
inline constexpr auto S_MOD_TYPEREF = 0x115f;
/* Reference to mini PDB */
inline constexpr auto S_REF_MINIPDB = 0x1160;
/* PDB path mapping */
inline constexpr auto S_PDBMAP = 0x1161;
/* DPC local procedure */
inline constexpr auto S_LPROC32_DPC = 0x1155;
/* DPC local procedure with ID */
inline constexpr auto S_LPROC32_DPC_ID = 0x1156;
/* Inlinee list */
inline constexpr auto S_INLINEES = 0x1168;
/* Fast link info */
inline constexpr auto S_FASTLINK = 0x1167;
/* Hot-patch function */
inline constexpr auto S_HOTPATCHFUNC = 0x1169;
/* Frame register */
inline constexpr auto S_FRAMEREG = 0x1166;
/* Attributed frame-relative */
inline constexpr auto S_ATTR_FRAMEREL = 0x112e;
/* Attributed register */
inline constexpr auto S_ATTR_REGISTER = 0x112f;
/* Attributed register-relative */
inline constexpr auto S_ATTR_REGREL = 0x1130;
/* Attributed many-register */
inline constexpr auto S_ATTR_MANYREG = 0x1131;
/* Virtual function table */
inline constexpr auto S_VFTABLE32 = 0x100c;
/* WITH scope */
inline constexpr auto S_WITH32 = 0x1104;
/* Many-register variable */
inline constexpr auto S_MANYREG = 0x110a;
/* Many-register variable v2 */
inline constexpr auto S_MANYREG2 = 0x1117;
/* Local managed slot */
inline constexpr auto S_LOCALSLOT = 0x111a;
/* Parameter managed slot */
inline constexpr auto S_PARAMSLOT = 0x111b;

/* Variable in register */
inline constexpr auto S_DEFRANGE_REGISTER = 0x1141;
/* FP-relative */
inline constexpr auto S_DEFRANGE_FRAMEPOINTER_REL = 0x1142;
/* Sub-field in register */
inline constexpr auto S_DEFRANGE_SUBFIELD_REGISTER = 0x1143;
/* FP-relative, valid entire function */
inline constexpr auto S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE = 0x1144;
/* Register-relative */
inline constexpr auto S_DEFRANGE_REGISTER_REL = 0x1145;

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
inline constexpr auto PDB_SYMBOL_UDT_NAME_OFFS = 4;
/* S_CONSTANT: the numeric leaf follows the type index; the name follows
   the leaf.  */
inline constexpr auto PDB_SYMBOL_CONST_VALUE_OFFS = 4;
inline constexpr auto PDB_SYMBOL_REF_MOD_INDEX_OFFS = 8;
inline constexpr auto PDB_SYMBOL_REF_NAME_OFFS = 10;

/* CodeView type and id leaf kinds.  */

/* Virtual function table shape */
inline constexpr auto LF_VTSHAPE = 0x000a;
/* Label type */
inline constexpr auto LF_LABEL = 0x000e;
/* modifier (const, volatile)  */
inline constexpr auto LF_MODIFIER = 0x1001;
/* Pointer to another type */
inline constexpr auto LF_POINTER = 0x1002;
/* IPI id of a non-member function.  */
inline constexpr auto LF_FUNC_ID = 0x1601;
/* IPI id of a member function.  */
inline constexpr auto LF_MFUNC_ID = 0x1602;
inline constexpr auto LF_SUBSTR_LIST = 0x1604;
inline constexpr auto LF_STRING_ID = 0x1605;

/* LF_FUNC_ID: scopeId (u32 IPI id, 0 for global scope),
   type (u32 TPI signature index), NUL-terminated name.  */
inline constexpr auto LF_FUNC_ID_TYPE_OFFS = 4;
inline constexpr auto LF_FUNC_ID_NAME_OFFS = 8;
/* LF_MFUNC_ID: parentType (u32 TPI class index),
   type (u32 TPI signature index), NUL-terminated name.  */
inline constexpr auto LF_MFUNC_ID_TYPE_OFFS = 4;
inline constexpr auto LF_MFUNC_ID_NAME_OFFS = 8;

inline constexpr auto LF_PROCEDURE = 0x1008;
/* Member function type */
inline constexpr auto LF_MFUNCTION = 0x1009;
/* Argument list for procedures */
inline constexpr auto LF_ARGLIST = 0x1201;
/* Field list for structs/classes */
inline constexpr auto LF_FIELDLIST = 0x1203;
/* Bit field */
inline constexpr auto LF_BITFIELD = 0x1205;
/* Method overload list */
inline constexpr auto LF_METHODLIST = 0x1206;
/* Array type */
inline constexpr auto LF_ARRAY = 0x1503;
/* Class type */
inline constexpr auto LF_CLASS = 0x1504;
/* Structure type */
inline constexpr auto LF_STRUCTURE = 0x1505;
/* Union type */
inline constexpr auto LF_UNION = 0x1506;
/* Enumeration type */
inline constexpr auto LF_ENUM = 0x1507;
/* Interface type */
inline constexpr auto LF_INTERFACE = 0x1519;
/* Numeric leaf encoding values.  These are prefix tags for variable-length
   integers embedded inside type records (struct sizes, member offsets,
   enum values).  See pdb_cv_read_numeric().  */
inline constexpr auto LF_NUMERIC = 0x8000;
inline constexpr auto LF_CHAR = 0x8000;
inline constexpr auto LF_SHORT = 0x8001;
inline constexpr auto LF_USHORT = 0x8002;
inline constexpr auto LF_LONG = 0x8003;
inline constexpr auto LF_ULONG = 0x8004;
inline constexpr auto LF_QUADWORD = 0x8009;
inline constexpr auto LF_UQUADWORD = 0x800a;

/* Compound type leaf types.  */

/* LF_FIELDLIST sub-record leaf types.
   Describe the members of a compound type (class, struct, union, enum).  */
/* Direct base class  */
inline constexpr auto LF_BCLASS = 0x1400;
/* Direct virtual base class  */
inline constexpr auto LF_VBCLASS = 0x1401;
/* Indirect virtual base class  */
inline constexpr auto LF_IVBCLASS = 0x1402;
/* Enum const (name + value)  */
inline constexpr auto LF_ENUMERATE = 0x1502;
/* Data member/field  */
inline constexpr auto LF_MEMBER = 0x150d;
/* Static data member  */
inline constexpr auto LF_STMEMBER = 0x150e;
/* Overloaded method (ptr to methodlist).  */
inline constexpr auto LF_METHOD = 0x150f;
/* Nested type definition  */
inline constexpr auto LF_NESTTYPE = 0x1510;
/* Non-overloaded method  */
inline constexpr auto LF_ONEMETHOD = 0x1511;
/* Virtual func. table pointer  */
inline constexpr auto LF_VFUNCTAB = 0x1409;
/* Continuation to another LF_FIELDLIST.  */
inline constexpr auto LF_INDEX = 0x1404;
/* Direct base interface; laid out as LF_BCLASS.  */
inline constexpr auto LF_BINTERFACE = 0x151a;
/* Nested type definition carrying access bits; laid out as LF_NESTTYPE.  */
inline constexpr auto LF_NESTTYPEEX = 0x1512;
/* Virtual function table offset.  */
inline constexpr auto LF_VFUNCOFF = 0x140c;

/* LF_MEMBER sub-record layout.
       uint16_t leaf
       uint16_t attr       CV_fldattr_t
       uint32_t type       type index of member type
       numeric  offset     byte offset in struct (variable length)
       char[]   name       follows numeric offset.  */
inline constexpr auto LF_MEMBER_ATTR_OFFS = 2;
inline constexpr auto LF_MEMBER_TYPE_OFFS = 4;
/* Start of numeric + name.  */
inline constexpr auto LF_MEMBER_DATA_OFFS = 8;

/* LF_ENUMERATE sub-record layout.
       uint16_t leaf
       uint16_t attr       CV_fldattr_t
       numeric  value      enumerator value (variable length)
       char[]   name       follows numeric value.  */
inline constexpr auto LF_ENUMERATE_ATTR_OFFS = 2;
/* Start of numeric + name  */
inline constexpr auto LF_ENUMERATE_DATA_OFFS = 4;

/* LF_BCLASS sub-record layout.
       uint16_t leaf
       uint16_t attr         CV_fldattr_t
       uint32_t type         base class type index
       numeric  offset       byte offset of base (variable length).  */
inline constexpr auto LF_BCLASS_ATTR_OFFS = 2;
inline constexpr auto LF_BCLASS_TYPE_OFFS = 4;
/* Start of numeric offset  */
inline constexpr auto LF_BCLASS_DATA_OFFS = 8;

/* LF_VBCLASS / LF_IVBCLASS sub-record layout.
       uint16_t leaf
       uint16_t attr        CV_fldattr_t
       uint32_t type        base class type index
       uint32_t vbptr       virtual base pointer type index
       numeric  vbpoff      vbptr offset in object (variable length)
       numeric  vbte        vbtable entry offset (variable length).  */
inline constexpr auto LF_VBCLASS_ATTR_OFFS = 2;
inline constexpr auto LF_VBCLASS_TYPE_OFFS = 4;
inline constexpr auto LF_VBCLASS_VBPTR_OFFS = 8;
/* Start of numeric leaves  */
inline constexpr auto LF_VBCLASS_DATA_OFFS = 12;

/* LF_STMEMBER sub-record layout.
       uint16_t leaf
       uint16_t attr          CV_fldattr_t
       uint32_t type          type index
       char[]   name          null-terminated name.  */
inline constexpr auto LF_STMEMBER_ATTR_OFFS = 2;
inline constexpr auto LF_STMEMBER_TYPE_OFFS = 4;
inline constexpr auto LF_STMEMBER_NAME_OFFS = 8;

/* LF_NESTTYPE sub-record layout.
       uint16_t leaf
       uint16_t pad
       uint32_t type          nested type index
       char[]   name          null-terminated name.
   An alias uses the same layout and refers to its target's type index.
   LF_NESTTYPEEX replaces pad with CV_fldattr_t access attributes.  */
inline constexpr auto LF_NESTTYPE_TYPE_OFFS = 4;
inline constexpr auto LF_NESTTYPE_NAME_OFFS = 8;

/* LF_ONEMETHOD  sub-record layout.
       uint16_t leaf
       uint16_t attr          CV_fldattr_t
       uint32_t type          method type index
       [uint32_t vbaseoff]    vtable slot byte offset; present only if
			      this is the first declaration of a virtual
			      method (mprop INTRO or PUREINTRO in attr)
       char[]   name          follows optional vbaseoff.  */
inline constexpr auto LF_ONEMETHOD_ATTR_OFFS = 2;
inline constexpr auto LF_ONEMETHOD_TYPE_OFFS = 4;
/* vbaseoff + name start  */
inline constexpr auto LF_ONEMETHOD_DATA_OFFS = 8;

/* LF_METHOD sub-record layout.
       uint16_t leaf
       uint16_t count         number of overloads
       uint32_t mlist         LF_METHODLIST type index
       char[]   name          null-terminated name.  */
inline constexpr auto LF_METHOD_COUNT_OFFS = 2;
inline constexpr auto LF_METHOD_MLIST_OFFS = 4;
inline constexpr auto LF_METHOD_NAME_OFFS = 8;

/* LF_VFUNCTAB sub-record layout.
       uint16_t leaf
       uint16_t pad
       uint32_t type          vtable pointer type index.
   No object offset is encoded by this record.  */
inline constexpr auto LF_VFUNCTAB_TYPE_OFFS = 4;
inline constexpr auto LF_VFUNCTAB_SIZE = 8;

/* LF_VFUNCOFF sub-record layout.
       uint16_t leaf
       uint16_t pad
       uint32_t type          vtable pointer type index
       int32_t  offset        byte offset of that pointer in the object.  */
inline constexpr auto LF_VFUNCOFF_SIZE = 12;

/* LF_INDEX sub-record layout.
       uint16_t leaf
       uint16_t pad
       uint32_t type          continuation LF_FIELDLIST index.  */
inline constexpr auto LF_INDEX_TYPE_OFFS = 4;
inline constexpr auto LF_INDEX_SIZE = 8;

/* LF_METHODLIST entry layout (repeating within the record data).
       uint16_t attr          CV_fldattr_t
       uint16_t pad
       uint32_t type          method type index
       [uint32_t vbaseoff]    vtable slot byte offset; present only if
			      this is the first declaration of a virtual
			      method (mprop INTRO/PUREINTRO in attr).  */
inline constexpr auto LF_MLIST_ATTR_OFFS = 0;
inline constexpr auto LF_MLIST_TYPE_OFFS = 4;
/* Size w/o optional vbaseoff  */
inline constexpr auto LF_MLIST_ENTRY_SIZE = 8;

/* CV_prop_t property bits (shared by LF_CLASS/STRUCTURE/UNION/ENUM).  */
/* Declared inside another class.  */
inline constexpr auto CV_PROP_ISNESTED = 0x0008;
/* Forward reference.  */
inline constexpr auto CV_PROP_FWDREF = 0x0080;
/* Declared inside a function body.  */
inline constexpr auto CV_PROP_SCOPED = 0x0100;
/* A decorated unique name follows the display name.  */
inline constexpr auto CV_PROP_HASUNIQUENAME = 0x0200;

/* CV_fldattr_t access bits for field list sub-records.  */
inline constexpr auto CV_ACCESS_MASK = 0x0003;
inline constexpr auto CV_ACCESS_PRIVATE = 1;
inline constexpr auto CV_ACCESS_PROTECTED = 2;
inline constexpr auto CV_ACCESS_PUBLIC = 3;

/* LF_PROCEDURE payload (after RecordKind).
       uint32_t rvtype      return type index
       uint8_t  calltype    calling convention (TODO: store for ABI)
       uint8_t  funcattr    attributes (unused)
       uint16_t parmcount   number of parameters (unused)
       uint32_t arglist     type index of argument list  */
inline constexpr auto LF_PROC_RVTYPE_OFFS = 0;
inline constexpr auto LF_PROC_PARMCOUNT_OFFS = 6;
inline constexpr auto LF_PROC_ARGLIST_OFFS = 8;
/* Minimum record data size.  */
inline constexpr auto LF_PROC_SIZE = 12;

/* LF_MFUNCTION payload (after RecordKind).
       uint32_t rvtype       return type index
       uint32_t classtype    containing class type index
       uint32_t thistype     this pointer type index
       uint8_t  calltype     calling convention (TODO)
       uint8_t  funcattr     attributes (unused)
       uint16_t parmcount    number of parameters (unused)
       uint32_t arglist      type index of argument list
       int32_t  thisadjust   this adjuster (unused)  */
inline constexpr auto LF_MFUNC_RVTYPE_OFFS = 0;
inline constexpr auto LF_MFUNC_CLASSTYPE_OFFS = 4;
inline constexpr auto LF_MFUNC_THISTYPE_OFFS = 8;
inline constexpr auto LF_MFUNC_PARMCOUNT_OFFS = 14;
inline constexpr auto LF_MFUNC_ARGLIST_OFFS = 16;
/* Minimum record data size.  */
inline constexpr auto LF_MFUNC_SIZE = 24;

/* LF_CLASS / LF_STRUCTURE field offsets.
       uint16_t count          number of members
       uint16_t property       CV_prop_t flags (bit 7 = fwdref)
       uint32_t field_list     type index of LF_FIELDLIST
       uint32_t derived        type index of derived-from list
       uint32_t vshape         type index of vshape table
       numeric  size           structure byte size (variable length)
       char[]   name           null-terminated name follows size
       [char[] unique_name]    present when property bit 9 is set.  */
inline constexpr auto LF_STRUCT_COUNT_OFFS = 0;
inline constexpr auto LF_STRUCT_PROPERTY_OFFS = 2;
inline constexpr auto LF_STRUCT_FIELDLIST_OFFS = 4;
inline constexpr auto LF_STRUCT_DERIVED_OFFS = 8;
inline constexpr auto LF_STRUCT_VSHAPE_OFFS = 12;
/* Numeric leaf/name start.  */
inline constexpr auto LF_STRUCT_DATA_OFFS = 16;
inline constexpr auto LF_STRUCT_MIN_SIZE = 16;

/* LF_UNION field offsets.
       uint16_t count
       uint16_t property
       uint32_t field_list
       numeric  size            union byte size (variable length)
       char[]   name            null-terminated name follows size
       [char[] unique_name]     present when property bit 9 is set.  */
inline constexpr auto LF_UNION_COUNT_OFFS = 0;
inline constexpr auto LF_UNION_PROPERTY_OFFS = 2;
inline constexpr auto LF_UNION_FIELDLIST_OFFS = 4;
inline constexpr auto LF_UNION_DATA_OFFS = 8;
inline constexpr auto LF_UNION_MIN_SIZE = 8;

/* LF_ENUM field offsets.
       uint16_t count          number of enumerators
       uint16_t property       CV_prop_t flags
       uint32_t utype          underlying integer type index
       uint32_t field_list     type index of LF_FIELDLIST
       char[]   name           null-terminated name
       [char[] unique_name]    present when property bit 9 is set.  */
inline constexpr auto LF_ENUM_COUNT_OFFS = 0;
inline constexpr auto LF_ENUM_PROPERTY_OFFS = 2;
inline constexpr auto LF_ENUM_UTYPE_OFFS = 4;
inline constexpr auto LF_ENUM_FIELDLIST_OFFS = 8;
inline constexpr auto LF_ENUM_NAME_OFFS = 12;
inline constexpr auto LF_ENUM_MIN_SIZE = 12;

/* LF_MODIFIER sub-record layout.
       uint32_t type          modified type index
       uint16_t attr          CV_modifier_e flags.  */
inline constexpr auto LF_MOD_TYPE_OFFS = 0;
inline constexpr auto LF_MOD_ATTR_OFFS = 4;
/* Minimum record data size.  */
inline constexpr auto LF_MOD_SIZE = 6;

/* LF_POINTER sub-record layout.
       uint32_t utype         underlying type index
       uint32_t attr          packed CV_pointer_attr bitfield.  */
inline constexpr auto LF_POINTER_UTYPE_OFFS = 0;
inline constexpr auto LF_POINTER_ATTR_OFFS = 4;
/* Minimum record data size.  */
inline constexpr auto LF_POINTER_SIZE = 8;

/* LF_POINTER attribute bitfield (uint32_t at LF_POINTER_ATTR_OFFS).
   Bit layout (little-endian):
     bits  0- 4 : ptrtype     (CV_ptrtype_e)
     bits  5- 7 : ptrmode     (CV_ptrmode_e)
     bit      8 : isflat32
     bit      9 : isvolatile
     bit     10 : isconst
     bit     11 : isunaligned
     bit     12 : isrestrict
     bits 13-18 : size        (pointer size in bytes)
     bit     19 : ismocom
     bit     20 : islref
     bit     21 : isrref
     bits 22-31 : unused.  */
inline constexpr uint32_t LF_POINTER_ATTR_PTRTYPE_MASK = 0x1F;
inline constexpr uint32_t LF_POINTER_ATTR_PTRMODE_SHIFT = 5;
inline constexpr uint32_t LF_POINTER_ATTR_PTRMODE_MASK = 0x7;
inline constexpr uint32_t LF_POINTER_ATTR_ISVOLATILE = 1u << 9;
inline constexpr uint32_t LF_POINTER_ATTR_ISCONST = 1u << 10;
inline constexpr uint32_t LF_POINTER_ATTR_SIZE_SHIFT = 13;
inline constexpr uint32_t LF_POINTER_ATTR_SIZE_MASK = 0x3F;
inline constexpr uint32_t LF_POINTER_ATTR_ISLREF = 1u << 20;
inline constexpr uint32_t LF_POINTER_ATTR_ISRREF = 1u << 21;

/* LF_ARRAY sub-record layout.
       uint32_t elemtype      element type index
       uint32_t idxtype       indexing type index
       gdb_byte data[]        numeric leaf for total size in bytes,
			      followed by NUL-terminated name.  */
inline constexpr auto LF_ARRAY_ELEMTYPE_OFFS = 0;
inline constexpr auto LF_ARRAY_IDXTYPE_OFFS = 4;
inline constexpr auto LF_ARRAY_DATA_OFFS = 8;
inline constexpr auto LF_ARRAY_MIN_SIZE = 8;

/* LF_BITFIELD sub-record layout.
       uint32_t type          underlying type index
       uint8_t  length        number of bits
       uint8_t  position      starting bit position.  */
inline constexpr auto LF_BITFIELD_TYPE_OFFS = 0;
inline constexpr auto LF_BITFIELD_LENGTH_OFFS = 4;
inline constexpr auto LF_BITFIELD_POSITION_OFFS = 5;
/* Minimum record data size.  */
inline constexpr auto LF_BITFIELD_SIZE = 6;

/* LF_ARGLIST sub-record layout.
       uint32_t count         number of type indices
       uint32_t arg[count]    array of type indices.
   A trailing type index 0 marks an ellipsis, not a void parameter.  The
   count includes that marker; it excludes an implicit 'this' parameter.  */
inline constexpr auto LF_ARGLIST_COUNT_OFFS = 0;
inline constexpr auto LF_ARGLIST_ARGS_OFFS = 4;
inline constexpr auto LF_ARGLIST_MIN_SIZE = 4;

/* CV_modifier_e: Modifier attribute bits in LF_MODIFIER attr.  */
/* const modifier.  */
inline constexpr auto CV_MODIFIER_CONST = 0x01;
/* volatile modifier.  */
inline constexpr auto CV_MODIFIER_VOLATILE = 0x02;

/* LF_POINTER sub-record layout.
       uint32_t utype          type index of underlying type
       uint32_t attr           attributes bitfield
     Pointers to members (pmem / pmfunc) have two more fields:
       uint32_t pmclass        containing class type index
       uint16_t pmenum         member-pointer representation (not yet used,
			       ABI-dependent)  */
inline constexpr auto LF_PTR_UTYPE_OFFS = 0;
inline constexpr auto LF_PTR_ATTR_OFFS = 4;
/* Minimum record data size.  */
inline constexpr auto LF_PTR_MIN_SIZE = 8;
/* Containing-class type index for pmem / pmfunc pointers.  */
inline constexpr auto LF_PTR_PMCLASS_OFFS = 8;

/* CV_ptrmode_e: pointer/reference mode, bits 5-7 of LF_POINTER attr.  */
/* Ordinary pointer (*).  */
inline constexpr auto CV_PTRMODE_POINTER = 0;
/* Lvalue reference (&).  */
inline constexpr auto CV_PTRMODE_LVALUE_REF = 1;
/* Pointer to data member (T C::*).  */
inline constexpr auto CV_PTRMODE_PMEM = 2;
/* Pointer to member function (R (C::*)(...)).  */
inline constexpr auto CV_PTRMODE_PMFUNC = 3;
/* Rvalue reference (&&).  */
inline constexpr auto CV_PTRMODE_RVALUE_REF = 4;

/* Extract CV_ptrmode_e from LF_POINTER.  */

inline uint32_t
cv_ptr_mode (uint32_t attr)
{
  return (attr >> LF_POINTER_ATTR_PTRMODE_SHIFT)
	 & LF_POINTER_ATTR_PTRMODE_MASK;
}

/* Extract the pointer size in bytes from LF_POINTER.  */

inline uint32_t
cv_ptr_size (uint32_t attr)
{
  return (attr >> LF_POINTER_ATTR_SIZE_SHIFT) & LF_POINTER_ATTR_SIZE_MASK;
}

/* Return true when MODE is a pointer to member (pmem or pmfunc).  */

inline bool
cv_ptr_is_member (uint32_t mode)
{
  return mode == CV_PTRMODE_PMEM || mode == CV_PTRMODE_PMFUNC;
}

/* Human-readable name for a CV_ptrmode_e value (for dumps).  */

inline const char *
cv_ptr_mode_name (uint32_t mode)
{
  switch (mode)
    {
    case CV_PTRMODE_POINTER:
      return "ptr";
    case CV_PTRMODE_LVALUE_REF:
      return "lvref";
    case CV_PTRMODE_PMEM:
      return "pmem";
    case CV_PTRMODE_PMFUNC:
      return "pmfunc";
    case CV_PTRMODE_RVALUE_REF:
      return "rvref";
    default:
      return "?";
    }
}

/* Return a human-readable name for a CodeView symbol record type.  */
extern std::string pdb_sym_rec_type_name (uint16_t rectype);

/* Index raw TPI records (stream 2), allocate the type cache, then build
   tag-name and ownership maps.  Missing streams and detected format errors
   raise pdb_error.  Record indexing does not construct GDB types.  */

void pdb_read_tpi_stream (pdb_per_objfile *pdb);

/* Read and parse the IPI (id) stream (stream 4) into pdb->ipi.  The IPI uses
   the same record container as TPI and holds LF_FUNC_ID / LF_MFUNC_ID records.
   Returns false (non-fatal) when the stream is absent or malformed.  */
bool pdb_read_ipi_stream (pdb_per_objfile *pdb);

/* Resolve an IPI item id (from S_INLINESITE.inlinee) to the inlined
   function's NAME and the TPI index of its signature (SIG_TI).  NAME and
   SIG_TI may be null.  NAME includes the recorded class/namespace scope
   and owns its storage.  Returns false for an invalid record, an invalid
   scope, or a requested name without a terminator.  No GDB types are
   constructed.  */

bool pdb_ipi_lookup_inlinee (pdb_per_objfile *pdb, uint32_t item_id,
			     std::string *name, uint32_t *sig_ti);

/* Read a CodeView numeric leaf.
   Returns number of bytes consumed (0 on error).  */
uint32_t pdb_cv_read_numeric (const gdb_byte *data, uint32_t max_len,
			      uint64_t *value);

/* Extract a NUL-terminated string from a bounded buffer [P, END).
   Returns the string pointer if a NUL byte exists in the range,
   nullptr otherwise.  Used to safely read variable-length name fields
   from PDB records without risking out-of-bounds reads.  */
const char *pdb_extract_string (const gdb_byte *p, const gdb_byte *end);

/* Resolve a TPI index to a GDB type after TPI initialization.  Unsupported
   indices/leaves return an error placeholder.  A forward TI resolves to its
   definition when the TPI has one.  Only calls made while a field list is
   being read can return a struct or union whose members are pending.  */
type &pdb_tpi_resolve_type (pdb_per_objfile *pdb, uint32_t type_index);

/* Return " &" or " &&" for a method type built from an lvalue- or
   rvalue-ref-qualified LF_MFUNCTION, or "" otherwise.  */
const char *pdb_method_ref_qualifier (const pdb_per_objfile *pdb,
				      const type *method_type);

/* Return the raw LF_PROCEDURE/LF_MFUNCTION parameter count, adding one for
   LF_MFUNCTION only if its 'this' TI is nonzero.  Return 0 for an absent,
   truncated or non-function record; no parameter types are resolved.  */
int pdb_tpi_get_func_param_count (pdb_per_objfile *pdb, uint32_t type_idx);

/* Return the CV_prop_t word of the tagged-type record at TYPE_IDX, or 0
   when TYPE_IDX is not a tagged type or lacks this field.  It records a class
   (CV_PROP_ISNESTED) or a function body (CV_PROP_SCOPED) encloses it, which
   its qualified name only implies.  */
uint16_t pdb_tpi_tag_props (const pdb_tpi_context *tpi, uint32_t type_idx);

/* Return the name stored in the tagged-type record at TYPE_IDX, or nullptr
   when TYPE_IDX is not a tagged type or carries no name.  */
const char *pdb_tpi_tag_name (const pdb_tpi_context *tpi, uint32_t type_idx);

/* Return the class type index of the LF_MFUNCTION at TYPE_IDX, or 0 when
   TYPE_IDX is absent, truncated or not a member-function record.  This gives
   a procedure symbol its declaring-class TI independently of name parsing.  */

uint32_t pdb_tpi_mfunction_class (const pdb_tpi_context *tpi,
				  uint32_t type_idx);

/* Map each nested tag and each static data member to the tag that declares
   it: CHILD_TO_OWNER (pdb->tpi.nested_owner) takes a nested tag TI to its
   declaring tag TI, and pdb->tpi.static_member_owner takes "Tag::member" to
   the TI of Tag.  No GDB types are built.

   Namespace inference needs these to tell a class scope from a namespace:
   in "ns::Outer::Inner", only Inner's declaring tag says that Outer is a
   class.  CV_PROP_ISNESTED marks a nested tag without naming its parent, and
   a static member's data symbol names no class, so the owner comes from the
   field list of the tag that declares it.

   Each field list is given the first tag that refers to it, and that owner
   is carried through LF_INDEX continuations.  The lists are then scanned for
   LF_NESTTYPE and LF_STMEMBER.  A nested edge needs the child's tag to be
   spelled exactly "owner::nested", which rejects aliases to differently
   named types.  A list shared by several tags keeps only the first as
   owner; STATS counts the sharing and the rejected bindings.  */

void pdb_build_parent_map (pdb_per_objfile *pdb,
			   std::unordered_map<uint32_t, uint32_t>
			     &child_to_owner,
			   pdb_nesting_stats &stats);

/* Return true if TYPE_IDX refers to a compound type (struct/union/enum)
   that is a forward reference (incomplete type).  Returns false for
   simple types, non-compound records, or out-of-range indices.  */
bool pdb_tpi_type_is_fwdref (const pdb_tpi_context *tpi, uint32_t type_idx);

/* Build tagged_type_names, fwdref_definition and enumerator_names from the
   indexed TPI records.  Each canonical LF_CLASS/STRUCTURE/UNION/ENUM name
   selects one TI, preferring a definition over a forward record.  Names are
   display names, so distinct records with one spelling share an entry.
   Called by pdb_read_tpi_stream after indexing; builds no GDB types.  */
void pdb_tpi_build_tagged_name_cache (pdb_per_objfile *pdb);

/* Register named TPI compound types (struct/union/enum) as
   LOC_TYPEDEF / STRUCT_DOMAIN symbols in a <pdb-types> CU.  Skips types
   selected as forward records or already claimed.  Claims names before
   resolving types and also emits enumerator symbols.  */
void pdb_register_tpi_typedefs (pdb_per_objfile *pdb);

/* Build the GDB type for the tagged type named NAME and add it to
   a new <pdb-types> CU, including enumerator symbols for an enum.
   Return false if NAME is absent, selects a forward record or is already
   claimed.  The name is claimed before resolving the type.  */
bool pdb_build_tagged_type (pdb_per_objfile *pdb,
			    std::string_view name);

/* Build every tagged type whose name the C++ symbol matcher accepts for
   LOOKUP.  Backs completion, where the typed text is a prefix and
   pdb_build_tagged_type's exact match would build nothing.  */
void pdb_build_tagged_types_matching (pdb_per_objfile *pdb,
				      const lookup_name_info &lookup);

/* Build each enum declaring an enumerator whose qualified name the C++
   symbol matcher accepts for LOOKUP, so its LOC_CONST symbols reach
   <pdb-types>.  Returns false if nothing was built.  */
bool pdb_build_enum_for_enumerator (pdb_per_objfile *pdb,
				    const lookup_name_info &lookup);

/* Give the scopes of every qualified tagged-type name a namespace symbol
   in <pdb-types>.  The expression parser resolves a scope before the tag
   it qualifies, so the by-name path is too late to build these.  */
void pdb_register_tpi_namespaces (pdb_per_objfile *pdb);

/* Return true if QNAME names a tagged compound type
   (LF_CLASS/LF_STRUCTURE/LF_UNION/LF_ENUM) in TPI.  */
bool pdb_tpi_is_tagged_type_name (const pdb_tpi_context *tpi,
				  const char *qname);

/* Call F with each enclosing scope prefix of QNAME, outermost first.
   Only a "::" outside every bracket separates scopes: template arguments
   carry their own, and cl quotes a local scope as `name', as in
   `ns::f'::`2'::<lambda_1>.  */
void pdb_for_each_scope_prefix (const char *qname,
				gdb::function_view<void (std::string_view)> f);

/* Length of the scope component at NAME, up to the next "::" or the end.
   Templates, parameter lists and operator names are split by
   cp_find_first_component; a quoted scope is stepped over whole, which is
   the one rule the C++ splitter lacks.  */
size_t pdb_scope_component_len (const char *name);

/* Offset in QNAME of its last "::"-separated component, or 0 when QNAME
   is unqualified.  Splits the way split_name does, without building the
   component vector that callers on a per-symbol path cannot afford.  */
size_t pdb_last_component_offset (const char *qname);

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

/* Decode a symbol header at REC.  The caller must provide four readable
   header bytes; this function reads them before validating LEN.  END is
   one past the buffer.  Return std::nullopt if LEN is too small or the full
   record extends past END.  */

std::optional<pdb_sym_record_hdr>
pdb_parse_sym_record_hdr (const gdb_byte *rec, const gdb_byte *end);

/* Snapshot the default value of debug_file_directory so pdb-path.c
   can detect when the user overrides it.  Called once during
   INIT_GDB_FILE (pdb_read).  */
void pdb_path_init_default_debug_dir ();

/* Get module's files using C13 lines DEBUG_S_FILECHKSMS records.
   This function is called from an info command to print the file list for
   a module.  Note that pdb_read_module_files_file_info already provides this
   functionality but that one might be buggy even though LLVM and microsoft-pdb
   are using it; the File Info Stream provides a 16-bit record for the total
   number of files which was originally used to count the files.  Due to its
   small range it was later dropped and another record is now used - the number
   of files per module.  However, that one is 16-bits as well.  At least
   theoretically this should not make any difference thus we provide a more
   reliable information on the files in a module.  */
void pdb_read_module_files_c13 (pdb_per_objfile *pdb, pdb_module_info *mod);

/* Walk all DEBUG_S_LINES file blocks in a module's C13 data,
   invoking CALLBACK for each one.  */
void pdb_walk_c13_line_blocks (const pdb_per_objfile *pdb,
			       pdb_module_info *mod, gdb_byte *module_stream,
			       pdb_line_block_fn callback);

/* Record line-table entries for one inlined function instance into CU.
   INLINEE_ID is the IPI id from the S_INLINESITE record; CHUNKS are the
   decoded code chunks (each relative to BASES[chunk.base_index]) with per-chunk
   line deltas.  BASES[0] is the enclosing procedure's start.  Looks up the
   inlinee's base line and source file in MOD's DEBUG_S_INLINEELINES
   subsection, then emits a line at each chunk start, switching source files
   as needed, and appends each chunk's line to SPANS.  Skip unavailable bases
   (current callers supply only base 0); do nothing without InlineeLines
   information for INLINEE_ID.

   Where a chunk ends and the next chunk does not continue it, the caller's
   line resumes: the line of the innermost ENCLOSING site's span covering
   that address, else the C13 row of SYM_LINES in effect there.  Without
   either, the sequence ends.  */

void pdb_record_inline_lines
  (pdb_per_objfile *pdb, pdb_module_info *mod, buildsym_compunit *cu,
   uint32_t inlinee_id, const std::vector<pdb_code_origin> &bases,
   const std::vector<pdb_inline_chunk> &chunks,
   const std::vector<const std::vector<pdb_inline_line_span> *> &enclosing,
   const pdb_symbol_lines *sym_lines,
   std::vector<pdb_inline_line_span> *spans);

/* Decode an S_INLINESITE binary-annotation bytestream (ANNOT, LEN bytes) into
   the inlined body's code chunks.  Each chunk carries a [start, end) offset
   range relative to its base (base_index; ChangeCodeOffsetBase selects bases)
   and the source line it maps to, as a delta from the inlinee's base line.
   Stops on truncation or an unknown opcode, returning the chunks collected so
   far.  */
std::vector<pdb_inline_chunk>
pdb_decode_inline_annotations (const gdb_byte *annot, size_t len);

/* Map decoded inline CHUNKS to relocated PC ranges, resolving each chunk's
   offsets against BASES[chunk.base_index].  Chunks whose base is out of range,
   or that map to a null PC, are skipped.  */
pdb_range_pair_vec
pdb_inline_chunk_ranges (pdb_per_objfile *pdb,
			 const std::vector<pdb_inline_chunk> &chunks,
			 const std::vector<pdb_code_origin> &bases);

} /* namespace pdb */

/* GDB_PDB_PDB_INTERNAL_H */
#endif
