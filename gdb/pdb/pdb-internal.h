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
#include "gdbarch.h"
#include "gdbsupport/filestuff.h"

#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <optional>

struct buildsym_compunit;
struct objfile;

namespace pdb
{

/* Type aliases for common PDB vector types.  */
using pdb_scope_stack = std::vector<CORE_ADDR>;
using pdb_range_pair_vec = std::vector<std::pair<CORE_ADDR, CORE_ADDR>>;

/* Owned stream buffer returned by pdb_read_stream.  */
using gdb_byte_ptr = std::unique_ptr<gdb_byte[]>;

/* Little-endian integer readers.
   These are wrappers around BFD functions for consistent interface.  */

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

#define CSTR(data) ((const char *) (data))

/* Debug verbosity level: 0 = off, 1 = basic, 2 = verbose, 3 = trace  */
extern unsigned int pdb_read_debug;
extern unsigned int pdb_convert_debug;

/* Print a "pdb" debug statement if pdb_read_debug is >= 1.  */
#define pdb_dbg_printf(fmt, ...) \
  debug_prefixed_printf_cond (pdb_read_debug >= 1, "pdb", fmt, ##__VA_ARGS__)

#define pdb_dbg_printf_v(fmt, ...) \
  debug_prefixed_printf_cond (pdb_read_debug >= 2, "pdb", fmt, ##__VA_ARGS__)

#define pdb_dbg_printf_t(fmt, ...) \
  debug_prefixed_printf_cond (pdb_read_debug >= 3, "pdb", fmt, ##__VA_ARGS__)

/* Error — fatal issue that stops the operation.  */
#define pdb_error(fmt, ...) error (_ ("PDB Error: " fmt), ##__VA_ARGS__)

/* Warning — important one-time message that allows continuation.  */
#define pdb_warning(fmt, ...) warning (_ ("PDB Warning: " fmt), ##__VA_ARGS__)

/* Complaint — recoverable diagnostic issue (repeatable, suppressible).  */
#define pdb_complaint(fmt, ...) \
  complaint (_ ("PDB Complaint: " fmt), ##__VA_ARGS__)

/* MSF SuperBlock is the first block in the PDB file and contains some basic
   information like the block size, number of blocks, and most importantly
   the location of the stream directory, which is used to locate all the
   other streams in the file. The SuperBlock is 64 bytes long, but we only
   need a few fields from it.  Each stream is split across multiple blocks
   as recorded in this directory. The directory itself might be split into
   multiple blocks which is described by the block map in the MSF SuperBlock.
   https://llvm.org/docs/PDB/MsfFile.html
*/
inline constexpr std::string_view PDB_MSF_MAGIC
  = "Microsoft C/C++ MSF 7.00\r\n\x1A\x44\x53\x00\x00";
inline constexpr auto PDB_MSF_MAGIC_SIZE = 32;
inline constexpr auto PDB_MSF_MAGIC_OFFS = 0;

inline constexpr auto MSF_HEADER_SIZE = 64;

inline constexpr auto MSF_BLOCK_SIZE_OFFS = 32 /* Block size in bytes */;
inline constexpr auto MSF_NUM_BLOCKS_OFFS = 40 /* Number of blocks */;
inline constexpr auto MSF_NUM_DIRECTORY_BYTES_OFFS
  = 44 /* Number of directory bytes */;
inline constexpr auto MSF_BLOCK_MAP_ADDR_OFFS = 52 /* Block map address */;

/* PDB streams */
inline constexpr auto PDB_STREAM_PDB = 1;
inline constexpr auto PDB_STREAM_TPI = 2;
inline constexpr auto PDB_STREAM_DBI = 3;
inline constexpr auto PDB_STREAM_IPI = 4;
inline constexpr auto PDB_NO_STREAM
  = 0xFFFFFFFF /* Invalid stream - microsoft-pdb/msf.cpp */;

/* Size of the per-record header: uint16_t RecordLen + uint16_t RecordKind.  */
inline constexpr auto CV_REC_HDR_SIZE = 4;

/* The PDB info stream (stream 1) contains version/age/GUID followed by
   the "named stream map" which is a hash table that maps stream identifiers
   (e.g. "/names", "/LinkInfo") to MSF stream numbers. Except for a few fixed
   streams like PDB info, TPI, IPI and DBI, all other streams have arbitrary
   Id and need to be found by name.
   Here we only needs the /names stream as that is where the global filename
   table resides, used to resolve filenames in C13 line info. To find the
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
   Field PresentWordCount just tells how many words is in the vector: E.g. if
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
/* Used for checks */
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
/* Names stream size (4 bytes) */
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
      - Module streams. Each module stream contains the debug information
	for the corresponding module. Each stream contains 3 debug records:
	- Symbol records.
	  - C11 line info records. Ignored (old line info format).
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
#define DBI_HDR_MODULE_INFO_OFFS DBI_HDR_SIZE /* Substreams start here */

/* File Info substream — contains info on each module's files.
   Stored in DBI stream at offset DBI_HDR_SOURCE_INFO_OFFS.
   Header is 4 bytes and contains the number of modules, followed by
   arrays of indices, file counts, and file offsets for each module.
      Indices array: num modules × 2 bytes
      File count array: num modules × 2 bytes
      File offset array: num modules × 2 bytes
   Each module has its slice of the file offset arrays, however, the size of
   each slice is not explicitly stored anywhere. Instead, we have to read the
   file count for each module and sum them up to know where the next module's
   slice starts.  */
/* Total header size  */
inline constexpr auto FILE_INFO_HDR_SIZE = 4;
#define FILE_INFO_MOD_INDICES_OFFS FILE_INFO_HDR_SIZE
/* Number of modules (2 bytes).  */
inline constexpr auto FILE_INFO_HDR_NUM_MODULES_OFFS = 0;
/* Each module entry (2 bytes) */
inline constexpr auto FILE_INFO_ELEMENT_SIZE = 2;

/* Section Contribution offsets  */

/* Section identifier (2 bytes)  */
inline constexpr auto MODI_SC_ISECT_OFFS = 4;
/* Offset in the section (4 bytes) */
inline constexpr auto MODI_SC_OFFSET_OFFS = 8;
/* Size of the contribution (4 bytes) */
inline constexpr auto MODI_SC_SIZE_OFFS = 12;

/* Module Info Header (ModInfo in microsoft-pdb) specifes the sizes
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

/* C13 line info records. Layout:
   4: CV_SIGNATURE_C13 (optional)
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
	 4: offset     — code offset relative to CV_LineSection.offs
	 4: line_start — line number (low 24 bits) + flags  */

/* DEBUG_S subsection types (C13 format) */

inline constexpr auto C13_SUBSECT_HEADER_SIZE = 8;

/* CV Signatures */
inline constexpr auto CV_SIGNATURE_SIZE = 4;
inline constexpr auto CV_SIGNATURE_C6 = 0L;
inline constexpr auto CV_SIGNATURE_C7 = 1L;
inline constexpr auto CV_SIGNATURE_C11 = 2L;
inline constexpr auto CV_SIGNATURE_C13 = 4L;

/* C13 subsection types we support. */
inline constexpr auto DEBUG_S_LINES = 0xF2;
inline constexpr auto DEBUG_S_FILECHKSMS = 0xF4;

/* C13 File block */
struct CV_FileBlock
{
  uint32_t file_id;    /* File checksum index */
  uint32_t num_lines;  /* Number of lines */
  uint32_t block_size; /* Block size */
};

#define PDB_FILECHKSUM_HDR_SIZE sizeof (CV_FileBlock)

/* File checksum entry - used to access file names in the string table created
   from the /names stream.  */
struct CV_FileChecksum
{
  uint32_t file_name_offset; /* Offset in string table */
  uint8_t checksum_size;
  uint8_t checksum_type;
  /* Followed by checksum bytes */
};

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
    6. Windows registry SymbolSearchPath. Same docs as above.
       TODO: SRV, SYMSRV, CACHE entries need symbol server fetch via HTTP.  */
extern std::string pdb_find_pdb_file (const objfile *objfile,
				      const pdb_rsds_info &rsds);

/* Per-file info from DEBUG_S_FILECHKSMS.  */
struct pdb_file_info
{
  const char *filename;  /* Resolved from string table */
  uint8_t checksum_type; /* 0=none, 1=MD5, 2=SHA1, 3=SHA256 */
  uint8_t checksum_size;
  const gdb_byte *checksum; /* Points into file_checksums (nullptr if none) */
};

/* PDB Module information */
struct pdb_module_info
{
  uint32_t module_index; /* Index in pdb->modules array */
  char *module_name;     /* As recorded in DBI Module Info Header */
  char *obj_file_name;   /* As recorded in DBI Module Info Header */

  uint16_t stream_number; /* Stream number of the module */
  uint32_t sym_byte_size; /* Size of symbol records in the module stream */
  uint32_t c11_byte_size; /* Size of C11 records in the module stream */
  uint32_t c13_byte_size; /* Size of C13 records in the module stream */

  uint16_t flags;

  /* Section contribution from Module Info header — identifies the
     code range this module contributes to the PE image. */
  uint16_t sc_section; /* PE section index (0 = no code) */
  uint32_t sc_offset;  /* Offset within section */
  uint32_t sc_size;    /* Size of contribution in bytes */

  /* File checksums (copied onto objfile obstack from module stream) */
  gdb_byte *file_checksums;
  uint32_t file_checksums_size;

  /* Expansion state. When true, pdb_build_module has already been
     called and 'cu' holds the result */
  bool expanded;
  compunit_symtab *cu;

  /* Pointers to the files in Names Buffer (in pdb_per_objfile) that belong to
     this module. These don't have to be loaded lazily as we already had to
     load the Names buffer. The complexity of parsing the buffers is in finding
     the module offset array (file_name_offsets below) from where we easily
     assign the files pointers. And that module offset array parsing is done
     eagerly because of some special calculations there that are easier to do
     at once for all the modules then per module. In the result, the basic
     pointer arithmetic to get files pointers is even simpler and really the
     files pointers should be assigned eagerly. Do it when lazy loading is fully
     completed.  */
  uint16_t num_files;
  const char **files;

  /* Offset array provides the offsets into the Names Buffer for each file
     in this module. Populated during the initialization - this is because
     the location of each module's offset slice depends on the sizes of the
     slices before it. After files get populated eagerly (see comment above)
     this one can be dropped as it will be internal to a function.  */
  const gdb_byte *file_name_offsets;

  /* File list obtained from C13 records (for info pdb-files-c13 command)
     TODO: Consider removing this if the command is no longer necessary.
     But also, need to consider using this file list instead of the one
     obtained from the File Info substream; even though LLVM uses the File
     Info substream info, that one can store only 65535 files per module.
     This might sound sufficient but PDB has another variable that stores
     up to 65535 files in total, and was abandoned due to this limitation.
     The limitation doesn't seem to be sufficiently addressed.  */
  bool files_c13_read;
  pdb_file_info *files_c13;
  uint32_t num_files_c13;
};

/* Raw CodeView type record from TPI/IPI stream.  */
struct pdb_tpi_type
{
  uint16_t leaf;        /* CodeView record type (LF_*) */
  uint16_t length;      /* RecordLen field (length of data) */
  const gdb_byte *data; /* Pointer to record data (RecordKind + fields) */
  uint32_t data_len;    /* Size of data (length - 2 for RecordKind) */
};

/* TPI (Type Program Information) context.
   Parsed type records and caching.  */
struct pdb_tpi_context
{
  /* TPI stream types.  */
  pdb_tpi_type *types;     /* Array of parsed type records */
  uint32_t type_idx_begin; /* First type index in TPI  */
  uint32_t type_idx_end;   /* One past last type index */

  /* Type cache. Covers both simple types (0x0000-0x0FFF) and compound types
     (0x1000..type_idx_end).  */
  type **type_cache;

  /* Type used for unsupported types.  */
  type *undefined_type;
};

/* PDB per-objfile data */
struct pdb_per_objfile
{
  explicit pdb_per_objfile (::objfile *objfile)
    : objfile (objfile)
  {
  }

  /* Back link to objfile.  */
  ::objfile *objfile;

  /* Path of the loaded PDB file on disk.  */
  std::string pdb_file_path;

  /* Persistent file handle for PDB blocks reads.  */
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
  uint32_t num_streams;
  uint32_t *stream_sizes;
  uint32_t **stream_blocks;

  /* Cached stream data */
  gdb_byte **stream_data;

  /* Module info */
  uint32_t num_modules;
  pdb_module_info *modules;

  /* Section mapping */
  uint32_t num_sections;
  CORE_ADDR *section_addresses;

  /* Loading File Info Substream details:
     This substream is per PDB and not per module, thus there is not much sense
     loading it lazily - any command like break or info sources will try to
     expand that module which will require this substream to be loaded. So, we
     can just load it eagerly. Since the File Info substream is the larger
     part of the DBI stream, we keep the file info substream as a pointer into
     DBI stream which is kept alive.  */
  gdb_byte *file_info_names_buffer;
  uint32_t file_info_names_buffer_size;

  /* Symbol record stream number from DBI header.  */
  uint16_t sym_record_stream;
  uint16_t gsi_stream;   /* Global Symbol Information stream index */
  uint16_t psgsi_stream; /* Public Symbol Information stream index */

  /* Cached symbol record stream data.
     TODO: Analyze if we can free it after loading PSI and GSI:
     - For PSI, after building the min sym table we don't need this data.
     - For GSI, records there point to this stream. We could instead
       allocate what GSI needs but we can safely assume the symbol names
       take most of the memory, thus copying them over in order to save
       the rest of memory doesn't seem good.
     - Dump sym records - this info command would need to read the stream
       from the file again.
   */
  gdb_byte *sym_record_data;
  uint32_t sym_record_size;

  /* Global string table from /names stream. It is a simple array
     of null-terminated strings; names are accessed by offset in
     CV_FileChecksum which is how the records are stored in the PDB.  */
  gdb_byte *string_table;
  uint32_t string_table_size;

  /* TPI stream size (from stream directory).  The stream data itself
     is cached in stream_data[PDB_STREAM_TPI] via pdb_read_stream.  */
  uint32_t tpi_size;

  /* Parsed TPI & IPI streams. Check tpi.types to verify parsing succeeded.  */
  pdb_tpi_context tpi;
};

/* C13 Line subsection header - see microsoft-pdb cvinfo.h */
struct CV_LineSection
{
  uint32_t offs;    /* section offset */
  uint16_t seci;    /* section index */
  uint16_t flags;   /* flags */
  uint32_t code_sz; /* code size */
};

/* C13 Line entry - see microsoft-pdb cvinfo.h*/
struct CV_Line
{
  uint32_t offs;             /* Offset in section */
  uint32_t line_start : 24;  /* Starting line number */
  uint32_t delta_end : 7;    /* Ending line delta */
  uint32_t is_statement : 1; /* Is statement flag */
};

/* Base struct filled by pdb_walk_c13_line_blocks for each file block.  */
struct pdb_line_block_info
{
  const char *filename;
  CV_LineSection *line_sect;
  CV_Line *lines;
  uint32_t num_lines;
};

/* Callback type for pdb_walk_c13_line_blocks. The callback is used
   when processing C13 line info as we have multiple users of that data.
   One of the users is info pdb-files-c13 command which might not be necessary.
   If not, drop the callback as well. */
using pdb_line_block_fn = void (*) (void *cb_data);

/* Return the pdb_per_objfile associated with OBJFILE, or nullptr if
   no PDB data has been loaded for this objfile.  */
extern pdb_per_objfile *pdb_get_per_objfile (objfile *objfile);

/* Expand module into its own compunit_symtab. Returns the cached CU on
   subsequent calls. Returns nullptr if the module has no useful debug data.  */
extern compunit_symtab *pdb_build_module (pdb_per_objfile *pdb,
					  pdb_module_info *mod);

/* Flags for pdb_parse_symbols.  */
inline constexpr auto PDB_DUMP_SYM
  = 0x1 /* Dump records (GDB symbols not created).  */;

/* Parse CodeView symbol records from a module stream into GDB symbols.
   MODULE_STREAM is the raw module stream data (caller owns it).
   When flags includes PDB_DUMP_SYM, records are dumped to stdout
   instead of creating GDB symbols.
   If FUNC_RANGES is non-null, each function's[start, end) is appended
   so the caller can register discontiguous block ranges.  */
extern void pdb_parse_symbols (pdb_per_objfile *pdb,
			       const pdb_module_info *mod_info,
			       gdb_byte *module_stream, buildsym_compunit *cu,
			       uint32_t flags = 0,
			       pdb_range_pair_vec *func_ranges = nullptr);

/* Read the symbol record stream into pdb->sym_record_data.  */
extern void pdb_read_sym_record_stream (pdb_per_objfile *pdb);

/* Walk the SymRecordStream and create GDB symbols for S_GDATA32/S_LDATA32
   (global variables, skipping forward-reference types) and S_CONSTANT
   (enum members, constexpr values), adding them to CU.  */
extern void pdb_load_global_syms (pdb_per_objfile *pdb, buildsym_compunit *cu);

/* Create a <pdb-globals> CU, load global symbols into it via
   pdb_load_global_syms, and finalize the compunit.  */
extern void pdb_load_global_syms_cu (pdb_per_objfile *pdb);

/* Parse CodeView symbol records from the global symbol record stream
   (SymRecordStream from DBI header).  Contains S_PUB32 and other globals.
   When flags includes PDB_DUMP_SYM, records are dumped to stdout.  */
extern void pdb_parse_sym_record_stream (pdb_per_objfile *pdb,
					 uint32_t /*flags*/ = 0);

/* Dump GSI hash records — reads the GSI stream and resolves each hash
   record's offset into the SymRecordStream, dumping the referenced symbol.  */
extern void pdb_dump_gsi_stream (pdb_per_objfile *pdb);

/* A gap within a location entry (range where a variable is unavailable)  */
struct pdb_loc_gap
{
  CORE_ADDR start; /* Start of gap  */
  CORE_ADDR end;   /* End of gap  */
};

/* Location entry (one DEFRANGE record) in pdb_loclist_baton.
   Resolved at parse time. Stored as a linked list on the objfile obstack.
   Gap array follows the struct inline.  */
struct pdb_loc_entry
{
  pdb_loc_entry *next;
  CORE_ADDR start;    /* Start PC  */
  CORE_ADDR end;      /* End PC    */
  int gdb_regnum;     /* GDB register number (-1 = unsupported) */
  int32_t offset;     /* Byte offset from register */
  bool is_register;   /* Value is the register itself (not memory) */
  bool is_full_scope; /* Matches entire function, no range check */
  int num_gaps;       /* Number of gaps in the gaps[] array */
  pdb_loc_gap gaps[]; /* Inline gap array */
};

/* Per-symbol location baton.
   Contains a linked list of parsed location entries, each fully
   resolved at parse time.  At read time we walk the list and find
   the entry matching the current PC. Allocated on the objfile obstack.  */
struct pdb_loclist_baton
{
  /* Linked list of parsed location entries (nullptr if none).  */
  pdb_loc_entry *entries;

  /* Back-pointer needed by describe_location.  */
  pdb_per_objfile *pdb;
};

/* Return true if SYM has a PDB-computed location (pdb_loclist_funcs).  */
extern bool pdb_is_pdb_location (const symbol *sym);

/* Dump resolved variable location batons from a built compunit_symtab.
   Walks all blocks, finds symbols with pdb_loclist_baton, and prints
   each pdb_loc_entry (range, register, offset, gaps).
   If SYMBOL_FILTER is non-null, only prints the matching symbol.  */
extern void pdb_dump_locations (compunit_symtab *cust, gdbarch *gdbarch,
				const char *symbol_filter);

/* Parse and optionally dump a single symbol record from the SymRecordStream.
   If PDB_DUMP_SYM is set in FLAGS, calls the record's dump() method.  */
extern void pdb_dump_parse_record (pdb_per_objfile *pdb, gdb_byte *rec_data,
				   uint16_t rectype, uint16_t reclen,
				   uint32_t flags);

/* Dump PSGSI (public symbol) hash records — reads the PSGSI stream and
   resolves each hash record into the SymRecordStream.  */
extern void pdb_dump_psgsi_stream (pdb_per_objfile *pdb);

/* Build minimal symbols from S_PUB32 records in the PSI stream.
   Called during PDB init after sections are available.  */
extern void pdb_build_minsyms (pdb_per_objfile *pdb);

/* Initialize the GSI hash table from the GSI stream.  Parses the
   stream header and caches pointers to the hash records, bitmap,
   and bucket offsets for O(1) lookup.  No data is copied.  */
extern void pdb_init_gsi_table (pdb_per_objfile *pdb);

/* Register the PDB location list implementation with the
   symbol_impl table.  Called once from INIT_GDB_FILE.  */
extern void pdb_init_loclist (void);

/* Get module's files. Parses the DBI File Info substream and resolves
   file names into module->files[]. Populates module->num_files and
   module->files on first call; subsequent calls are no-ops.  */
extern void pdb_read_module_files (pdb_per_objfile *pdb, pdb_module_info *mod);

/* Read a module's debug stream and extract file checksums onto the
   objfile obstack.  Returns an owned buffer with the raw stream data;
   the caller keeps it alive as long as the data is needed.  Returns
   nullptr if the module has no stream.  */
extern gdb_byte_ptr pdb_read_module_stream (pdb_per_objfile *pdb,
					    pdb_module_info *mod);

/* Resolve the i-th filename for a module from the shared File Info
   substream. Returns nullptr if index is out of range or the substream
   is not loaded. Caller must call pdb_read_module_files first.  */
extern const char *pdb_module_file_name (pdb_per_objfile *pdb,
					 const pdb_module_info *mod,
					 uint32_t index);

/* Map a (section, offset) pair to a relocated PC. Returns 0 on
   failure (invalid section index or missing section addresses).  */
extern CORE_ADDR pdb_map_section_offset_to_pc (pdb_per_objfile *pdb,
					       uint16_t section,
					       uint32_t offset);

/* TODO: Symbol server fetcher */
std::string pdb_symserver_fetch (const std::string &entry,
				 const std::string &pdb_name,
				 const pdb_rsds_info &rsds);

/* Try to open PATH for reading. Returns true if the file exists.  */
bool pdb_try_open (const char *path);

/* Extract the directory portion of PATH (including trailing separator).
   Returns empty string if PATH has no directory component.  */
std::string pdb_dirname (const char *path);

/* Read a stream by index. Returns an owned buffer that the caller must
   keep alive as long as the data is needed. To cache a stream for the
   lifetime of the objfile, release the pointer into
   pdb->stream_data[stream_idx].  Returns nullptr if the stream doesn't
   exist.  */
extern gdb_byte_ptr pdb_read_stream (const pdb_per_objfile *pdb,
				     uint32_t stream_idx);

/* ---------------------------------------------------------------
   CodeView Simple Type Constants (cvinfo.h)
   Simple type indices < 0x1000 encode: kind (bits 0-7) + mode (bits 8-11)
   ---------------------------------------------------------------  */

#define CV_SIMPLE_KIND(ti) ((ti) & 0xFF) /* Extract kind from type index */
#define CV_SIMPLE_MODE(ti) \
  (((ti) >> 8) & 0x0F)                      /* Extract mode from type index */
#define CV_TI_IS_SIMPLE(ti) ((ti) < 0x1000) /* Check if type index is simple */

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
inline constexpr auto CV_SBYTE = 0x11;
inline constexpr auto CV_BYTE = 0x21;
/* Legacy MSVC \"basic\" integer codes that the existing CV_INT* set
   (0x72-0x77) does not cover.  MSVC emits T_LONG/T_QUAD for
   long / long long instead of T_INT4/T_INT8.  */
inline constexpr auto CV_LONG = 0x12 /* T_LONG  - long (4 bytes on Windows) */;
inline constexpr auto CV_ULONG = 0x22 /* T_ULONG - unsigned long */;
inline constexpr auto CV_QUAD = 0x13 /* T_QUAD  - long long / __int64 */;
inline constexpr auto CV_UQUAD = 0x23 /* T_UQUAD - unsigned long long */;
inline constexpr auto CV_INT16 = 0x72;
inline constexpr auto CV_UINT16 = 0x73;
inline constexpr auto CV_INT32 = 0x74;
inline constexpr auto CV_UINT32 = 0x75;
inline constexpr auto CV_INT64 = 0x76;
inline constexpr auto CV_UINT64 = 0x77;
inline constexpr auto CV_FLOAT32 = 0x40;
inline constexpr auto CV_FLOAT64 = 0x41;
inline constexpr auto CV_FLOAT80 = 0x42;
inline constexpr auto CV_BOOL8 = 0x30;

/* SimpleTypeMode values */
inline constexpr auto CV_TM_DIRECT = 0 /* No indirection */;
inline constexpr auto CV_TM_NPTR = 1 /* Near pointer */;
inline constexpr auto CV_TM_NPTR32 = 4 /* 32-bit near pointer */;
inline constexpr auto CV_TM_FPTR32 = 5 /* 32-bit far pointer */;
inline constexpr auto CV_TM_NPTR64 = 6 /* 64-bit near pointer */;
inline constexpr auto CV_TM_NPTR128 = 7 /* 128-bit near pointer */;

/* ---------------------------------------------------------------
   Global Symbol Streams
   ---------------------------------------------------------------

   The DBI header references three stream indices that together form
   the global symbol infrastructure:

   1. Symbol Records Stream -index at DBI_HDR_SYM_RECORD_STREAM_OFFS.
      Array of symbol records that GSI and PSI streams reference.
      Layout of each symbol record matches the layout of CodeView symbol
      records in module streams:
	  uint16_t reclen;   — length of data after this field
	  uint16_t rectype;  — CodeView record type
	  uint8_t  data[];   — record-type-specific payload

   2. Global Symbol Index (GSI stream) - Index at DBI_HDR_GSI_STREAM_OFFS.
      GSI is a hash table that maps the symbol name to the location in the
      module symbol records where the symbol is fully described. As such,
      GSI can be used as cooked index.
      Contains mostly S_PROCREF / S_LPROCREF / S_DATAREF records
      (which are cross-references pointing into module streams),
      plus S_UDT, S_CONSTANT, etc.  Layout:
	GSI Header (GSIHashHdr in microsoft-pdb gsi.h):
	  uint32_t VerSignature;  (0)  always 0xFFFFFFFF
	  uint32_t VerHdr;        (4)  always 0xeffe0000 + 19990810
	  uint32_t cbHr;          (8)  byte count of hash records
	  uint32_t cbBuckets;     (12) byte count of bucket data
	HashRecord[cbHr/8]:
	  uint32_t Off;   (0)  1-based byte offset into SymRecordStream
	  uint32_t CRef;  (4)  reference count (vestigial, always 1)
	Bucket data (cbBuckets bytes):
	  uint32_t bitmap[ceil(4097/32)] — one bit per bucket
	  uint32_t offsets[popcount(bitmap)] — compressed array,
	    one byte-offset-into-HR-array per present bucket

      Name lookup algorithm (pdb_gsi_lookup):
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
       pdb->sym_record_data / sym_record_size.
     - pdb_build_minsyms: walks PSGSI hash records, resolves each
       S_PUB32 in sym_record_data, creates minimal_symbol entries.
     - pdb_init_gsi_table: parses the GSI stream header and caches
       pointers to the hash records and bucket table.  This enables
       O(1) name lookup via pdb_gsi_lookup without copying any data.
       Lookup resolves S_PROCREF/S_DATAREF entries that point into
       module streams.  sym_record_data must stay alive because
       name pointers and record data reference it directly.  */
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
  uint32_t sig;          /* VerSignature (always 0xFFFFFFFF).  */
  uint32_t ver;          /* VerHdr (always 0xeffe0000 + 19990810).  */
  uint32_t hr_bytes;     /* Byte count of HashRecord entries.  */
  uint32_t bucket_bytes; /* Byte count of bucket data.  */
  gdb_byte *hr_data;     /* Start of HashRecord array.  */
  gdb_byte *bucket_data; /* Start of bucket data (after records).  */
  bool valid;            /* True if parsing succeeded.  */
};

/* Validate and parse a GSI hash region at DATA with DATA_SIZE bytes.
   Fills HDR on success and returns true.  The data pointer must
   point to the GSIHashHdr (i.e. past any PublicsStreamHeader).  */
extern pdb_gsi_hdr pdb_parse_gsi_hash (gdb_byte *data, uint32_t data_size);

/* GSI hash bucket constants (microsoft-pdb GSI1.h).  */
inline constexpr auto GSI_NUM_BUCKETS = 4097 /* IPHR_HASH + 1 */;
inline constexpr auto GSI_BITMAP_WORDS = 129 /* ceil(GSI_NUM_BUCKETS / 32) */;

/* ---------------------------------------------------------------
   CodeView Symbol Record Types
   ---------------------------------------------------------------  */
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
/* Register-relative var. */
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

/* TODO: Unsupported CodeView symbol types  */
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
     uint16_t reclen;   (length of data following reclen)
     uint16_t rectype;  (symbol type)
     ...  record-type-specific data ...
*/
inline constexpr auto PDB_RECORD_LEN_OFFS = 0;
inline constexpr auto PDB_RECORD_TYPE_OFFS = 2;
inline constexpr auto PDB_RECORD_DATA_OFFS = 4;
inline constexpr auto PDB_RECORD_HDR_SIZE = 4;

/* ---------------------------------------------------------------
   CodeView Leaf Type Record Types (LF_* values for TPI stream)
   ---------------------------------------------------------------  */

/* Virtual function table shape */
inline constexpr auto LF_VTSHAPE = 0x000a;
/* Label type */
inline constexpr auto LF_LABEL = 0x000e;
/* modifier (const, volatile)  */
inline constexpr auto LF_MODIFIER = 0x1001;
/* Pointer to another type */
inline constexpr auto LF_POINTER = 0x1002;
/* Function/procedure type */
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
inline constexpr auto LF_INTERFACE = 0x1508;
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
   Describe the members of a compound type (class, struct, union, enum). */
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
/* Overloaded method (ptr to methodlist). */
inline constexpr auto LF_METHOD = 0x150f;
/* Nested type definition  */
inline constexpr auto LF_NESTTYPE = 0x1510;
/* Non-overloaded method  */
inline constexpr auto LF_ONEMETHOD = 0x1511;
/* Virtual func. table pointer  */
inline constexpr auto LF_VFUNCTAB = 0x1409;
/* Continuation to another LF_FIELDLIST.  */
inline constexpr auto LF_INDEX = 0x1404;

/* LF_MEMBER sub-record layout.
       uint16_t leaf          (0)
       uint16_t attr          (2)  CV_fldattr_t
       uint32_t type          (4)  type index of member type
       numeric  offset        (8)  byte offset in struct (variable length)
       char[]   name               follows numeric offset.  */
inline constexpr auto LF_MEMBER_ATTR_OFFS = 2;
inline constexpr auto LF_MEMBER_TYPE_OFFS = 4;
/* Start of numeric + name.  */
inline constexpr auto LF_MEMBER_DATA_OFFS = 8;

/* LF_ENUMERATE sub-record layout.
       uint16_t leaf          (0)
       uint16_t attr          (2)  CV_fldattr_t
       numeric  value         (4)  enumerator value (variable length)
       char[]   name               follows numeric value.  */
inline constexpr auto LF_ENUMERATE_ATTR_OFFS = 2;
/* Start of numeric + name  */
inline constexpr auto LF_ENUMERATE_DATA_OFFS = 4;

/* LF_BCLASS sub-record layout.
       uint16_t leaf          (0)
       uint16_t attr          (2)  CV_fldattr_t
       uint32_t type          (4)  base class type index
       numeric  offset        (8)  byte offset of base (variable length).  */
inline constexpr auto LF_BCLASS_ATTR_OFFS = 2;
inline constexpr auto LF_BCLASS_TYPE_OFFS = 4;
/* Start of numeric offset  */
inline constexpr auto LF_BCLASS_DATA_OFFS = 8;

/* LF_VBCLASS / LF_IVBCLASS sub-record layout.
       uint16_t leaf          (0)
       uint16_t attr          (2)  CV_fldattr_t
       uint32_t type          (4)  base class type index
       uint32_t vbptr         (8)  virtual base pointer type index
       numeric  vbpoff       (12)  vbptr offset in object (variable length)
       numeric  vbte               vbtable entry offset (variable length).  */
inline constexpr auto LF_VBCLASS_ATTR_OFFS = 2;
inline constexpr auto LF_VBCLASS_TYPE_OFFS = 4;
inline constexpr auto LF_VBCLASS_VBPTR_OFFS = 8;
/* Start of numeric leaves  */
inline constexpr auto LF_VBCLASS_DATA_OFFS = 12;

/* LF_STMEMBER sub-record layout.
       uint16_t leaf          (0)
       uint16_t attr          (2)  CV_fldattr_t
       uint32_t type          (4)  type index
       char[]   name          (8)  null-terminated name.  */
inline constexpr auto LF_STMEMBER_ATTR_OFFS = 2;
inline constexpr auto LF_STMEMBER_TYPE_OFFS = 4;
inline constexpr auto LF_STMEMBER_NAME_OFFS = 8;

/* LF_NESTTYPE sub-record layout.
       uint16_t leaf          (0)
       uint16_t pad           (2)
       uint32_t type          (4)  nested type index
       char[]   name          (8)  null-terminated name.  */
inline constexpr auto LF_NESTTYPE_TYPE_OFFS = 4;
inline constexpr auto LF_NESTTYPE_NAME_OFFS = 8;

/* LF_ONEMETHOD  sub-record layout.
       uint16_t leaf          (0)
       uint16_t attr          (2)  CV_fldattr_t
       uint32_t type          (4)  method type index
       [uint32_t vbaseoff     (8)] vtable slot byte offset; present only if
				   this is the first declaration of a virtual
				   method (mprop INTRO or PUREINTRO in attr)
       char[]   name               follows optional vbaseoff.  */
inline constexpr auto LF_ONEMETHOD_ATTR_OFFS = 2;
inline constexpr auto LF_ONEMETHOD_TYPE_OFFS = 4;
/* vbaseoff + name start  */
inline constexpr auto LF_ONEMETHOD_DATA_OFFS = 8;

/* LF_METHOD sub-record layout.
       uint16_t leaf          (0)
       uint16_t count         (2)  number of overloads
       uint32_t mlist         (4)  LF_METHODLIST type index
       char[]   name          (8)  null-terminated name.  */
inline constexpr auto LF_METHOD_COUNT_OFFS = 2;
inline constexpr auto LF_METHOD_MLIST_OFFS = 4;
inline constexpr auto LF_METHOD_NAME_OFFS = 8;

/* LF_VFUNCTAB sub-record layout.
       uint16_t leaf          (0)
       uint16_t pad           (2)
       uint32_t type          (4).  */
inline constexpr auto LF_VFUNCTAB_SIZE = 8;

/* LF_INDEX sub-record layout.
       uint16_t leaf          (0)
       uint16_t pad           (2)
       uint32_t type          (4)  continuation type index.  */
inline constexpr auto LF_INDEX_TYPE_OFFS = 4;
inline constexpr auto LF_INDEX_SIZE = 8;

/* LF_METHODLIST entry layout (repeating within the record data).
       uint16_t attr          (0)  CV_fldattr_t
       uint16_t pad           (2)
       uint32_t type          (4)  method type index
       [uint32_t vbaseoff     (8)] vtable slot byte offset; present only if
				   this is the first declaration of a virtual
				   method (mprop INTRO/PUREINTRO in attr).  */
inline constexpr auto LF_MLIST_ATTR_OFFS = 0;
inline constexpr auto LF_MLIST_TYPE_OFFS = 4;
/* Size w/o optional vbaseoff  */
inline constexpr auto LF_MLIST_ENTRY_SIZE = 8;

/* CV_prop_t property bits (shared by LF_CLASS/STRUCTURE/UNION/ENUM).  */
inline constexpr auto CV_PROP_FWDREF = 0x0080 /* Forward reference  */;

/* CV_fldattr_t access bits for field list sub-records.  */
inline constexpr auto CV_ACCESS_MASK = 0x0003;
inline constexpr auto CV_ACCESS_PRIVATE = 1;
inline constexpr auto CV_ACCESS_PROTECTED = 2;
inline constexpr auto CV_ACCESS_PUBLIC = 3;

/* LF_PROCEDURE sub-record layout.
       uint32_t rvtype      (0)  return type index
       uint8_t  calltype    (4)  calling convention (TODO: store for ABI)
       uint8_t  funcattr    (5)  attributes (unused)
       uint16_t parmcount   (6)  number of parameters (unused)
       uint32_t arglist     (8)  type index of argument list  */
inline constexpr auto LF_PROC_RVTYPE_OFFS = 0;
inline constexpr auto LF_PROC_PARMCOUNT_OFFS = 6;
inline constexpr auto LF_PROC_ARGLIST_OFFS = 8;
inline constexpr auto LF_PROC_SIZE = 12 /* Minimum record data size.  */;

/* LF_MFUNCTION sub-record layout.
       uint32_t rvtype      (0)  return type index
       uint32_t classtype   (4)  containing class type index
       uint32_t thistype    (8)  this pointer type index
       uint8_t  calltype   (12)  calling convention (TODO)
       uint8_t  funcattr   (13)  attributes (unused)
       uint16_t parmcount  (14)  number of parameters (unused)
       uint32_t arglist    (16)  type index of argument list
       int32_t  thisadjust (20)  this adjuster (unused)  */
inline constexpr auto LF_MFUNC_RVTYPE_OFFS = 0;
inline constexpr auto LF_MFUNC_CLASSTYPE_OFFS = 4;
inline constexpr auto LF_MFUNC_THISTYPE_OFFS = 8;
inline constexpr auto LF_MFUNC_PARMCOUNT_OFFS = 14;
inline constexpr auto LF_MFUNC_ARGLIST_OFFS = 16;
inline constexpr auto LF_MFUNC_SIZE = 24 /* Minimum record data size.  */;

/* LF_CLASS / LF_STRUCTURE field offsets.
       uint16_t count          (0)  number of members
       uint16_t property       (2)  CV_prop_t flags (bit 7 = fwdref)
       uint32_t field_list     (4)  type index of LF_FIELDLIST
       uint32_t derived        (8)  type index of derived-from list
       uint32_t vshape        (12)  type index of vshape table
       numeric  size          (16)  structure byte size (variable length)
       char[]   name                null-terminated name follows size.  */
inline constexpr auto LF_STRUCT_COUNT_OFFS = 0;
inline constexpr auto LF_STRUCT_PROPERTY_OFFS = 2;
inline constexpr auto LF_STRUCT_FIELDLIST_OFFS = 4;
inline constexpr auto LF_STRUCT_DERIVED_OFFS = 8;
inline constexpr auto LF_STRUCT_VSHAPE_OFFS = 12;
inline constexpr auto LF_STRUCT_DATA_OFFS = 16 /* numeric leaf/name start  */;
inline constexpr auto LF_STRUCT_MIN_SIZE = 16;

/* LF_UNION field offsets.
       uint16_t count          (0)
       uint16_t property       (2)
       uint32_t field_list     (4)
       numeric  size           (8)  structure byte size (variable length)
       char[]   name                null-terminated name follows size.  */
inline constexpr auto LF_UNION_COUNT_OFFS = 0;
inline constexpr auto LF_UNION_PROPERTY_OFFS = 2;
inline constexpr auto LF_UNION_FIELDLIST_OFFS = 4;
inline constexpr auto LF_UNION_DATA_OFFS = 8;
inline constexpr auto LF_UNION_MIN_SIZE = 8;

/* LF_ENUM field offsets.
       uint16_t count          (0)  number of enumerators
       uint16_t property       (2)  CV_prop_t flags
       uint32_t utype          (4)  underlying integer type index
       uint32_t field_list     (8)  type index of LF_FIELDLIST
       char[]   name          (12)  null-terminated name.  */
inline constexpr auto LF_ENUM_COUNT_OFFS = 0;
inline constexpr auto LF_ENUM_PROPERTY_OFFS = 2;
inline constexpr auto LF_ENUM_UTYPE_OFFS = 4;
inline constexpr auto LF_ENUM_FIELDLIST_OFFS = 8;
inline constexpr auto LF_ENUM_NAME_OFFS = 12;
inline constexpr auto LF_ENUM_MIN_SIZE = 12;

/* LF_MODIFIER sub-record layout.
       uint32_t type          (0)  modified type index
       uint16_t attr          (4)  CV_modifier_e flags.  */
inline constexpr auto LF_MOD_TYPE_OFFS = 0;
inline constexpr auto LF_MOD_ATTR_OFFS = 4;
inline constexpr auto LF_MOD_SIZE = 6 /* Minimum record data size.  */;

/* LF_BITFIELD sub-record layout.
       uint32_t type          (0)  underlying type index
       uint8_t  length        (4)  number of bits
       uint8_t  position      (5)  starting bit position.  */
inline constexpr auto LF_BITFIELD_TYPE_OFFS = 0;
inline constexpr auto LF_BITFIELD_LENGTH_OFFS = 4;
inline constexpr auto LF_BITFIELD_POSITION_OFFS = 5;
/* Minimum record data size.  */
inline constexpr auto LF_BITFIELD_SIZE = 6;

/* LF_ARGLIST sub-record layout.
       uint32_t count         (0)  number of type indices
       uint32_t arg[count]    (4)  array of type indices.  */
inline constexpr auto LF_ARGLIST_COUNT_OFFS = 0;
inline constexpr auto LF_ARGLIST_ARGS_OFFS = 4;
inline constexpr auto LF_ARGLIST_MIN_SIZE = 4;

/* CV_modifier_e: Modifier attribute bits in LF_MODIFIER attr.  */
/* const modifier.  */
inline constexpr auto CV_MODIFIER_CONST = 0x01;
/* volatile modifier.  */
inline constexpr auto CV_MODIFIER_VOLATILE = 0x02;

/* LF_POINTER sub-record layout.
       uint32_t utype          (0)  type index of underlying type
       uint32_t attr           (4)  attributes bitfield.  */
inline constexpr auto LF_PTR_UTYPE_OFFS = 0;
inline constexpr auto LF_PTR_ATTR_OFFS = 4;
/* Minimum record data size.  */
inline constexpr auto LF_PTR_MIN_SIZE = 8;

/* LF_ARRAY sub-record layout.
       uint32_t elemtype       (0)  element type index
       uint32_t idxtype        (4)  index type index
       numeric  size           (8)  array byte size (variable length)
       char[]   name                null-terminated name follows size.  */
inline constexpr auto LF_ARRAY_ELEMTYPE_OFFS = 0;
inline constexpr auto LF_ARRAY_IDXTYPE_OFFS = 4;
/* Start of numeric leaf + name.  */
inline constexpr auto LF_ARRAY_DATA_OFFS = 8;
inline constexpr auto LF_ARRAY_MIN_SIZE = 8;

/* Pointer/reference mode encoding in LF_POINTER attr.  */
/* Lvalue reference (&).  */
inline constexpr auto CV_PTRMODE_LVALUE_REF = 1;
/* Pointer (*).  */
inline constexpr auto CV_PTRMODE_POINTER = 2;
/* Rvalue reference (&&).  */
inline constexpr auto CV_PTRMODE_RVALUE_REF = 3;

/* Return a human-readable name for a CodeView symbol record type.  */
extern std::string pdb_sym_rec_type_name (uint16_t rectype);

/* Read and parse TPI (stream 2) and IPI (stream 4) if present.
   Populates pdb->tpi->types[] with decoded type records.  */
bool pdb_read_tpi_stream (pdb_per_objfile *pdb);

/* Read a CodeView numeric leaf.
   Returns number of bytes consumed (0 on error).  */
uint32_t pdb_cv_read_numeric (const gdb_byte *data, uint32_t max_len,
			      uint64_t *value);

/* Resolve type index to GDB type.  */
type *pdb_tpi_resolve_type (pdb_per_objfile *pdb, uint32_t type_index);

/* Return the number of parameters for a function type index.
   For LF_MFUNCTION, includes the implicit 'this' parameter.
   Returns 0 if type_idx is not a function type.  */
int pdb_tpi_get_func_param_count (pdb_per_objfile *pdb, uint32_t type_idx);

/* Return true if TYPE_IDX refers to a compound type (struct/union/enum)
   that is a forward reference (incomplete type).  Returns false for
   simple types, non-compound records, or out-of-range indices.  */
bool pdb_tpi_type_is_fwdref (const pdb_tpi_context *tpi, uint32_t type_idx);

/* Register named TPI compound types (struct/union/enum) as
   LOC_TYPEDEF / STRUCT_DOMAIN symbols in a <pdb-types> CU.  */
void pdb_register_tpi_typedefs (pdb_per_objfile *pdb);

bool pdb_parse_sym_record_hdr (const gdb_byte *rec, const gdb_byte *end,
			       uint16_t &reclen, uint16_t &rectype,
			       size_t &rec_size);

/* Snapshot the default value of debug_file_directory so pdb-path.c
   can detect when the user overrides it.  Called once during
   INIT_GDB_FILE (pdb_read).  */
void pdb_path_init_default_debug_dir ();

/* Get module's files using C13 lines DEBUG_S_FILECHKSMS records.
   This function is called from an info command to print the file list for
   a module. Note that pdb_read_module_files_file_info already provides this
   functionality but that one might be buggy even though LLVM and microsoft-pdb
   are using it; the File Info Stream provides a 16-bit record for the total
   number of files which was originally used to count the files. Due to its
   small range it was later dropped and another record is now used - the number
   of files per module. However, that one is 16-bits as well. At least
   theoretically this should not make any difference thus we provide a more
   reliable information on the files in a module. */
void pdb_read_module_files_c13 (pdb_per_objfile *pdb, pdb_module_info *mod);

/* Walk all DEBUG_S_LINES file blocks in a module's C13 data,
   invoking CALLBACK for each one.  */
void pdb_walk_c13_line_blocks (const pdb_per_objfile *pdb,
			       pdb_module_info *mod, gdb_byte *module_stream,
			       pdb_line_block_fn callback, void *cb_data);

} // namespace pdb

#endif /* GDB_PDB_PDB_INTERNAL_H */
