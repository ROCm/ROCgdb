/* PDB debugging format support for GDB - Header file.

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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#ifndef GDB_PDB_PDB_H
#define GDB_PDB_PDB_H

#include "objfiles.h"

#include <string>
#include <vector>
#include <memory>

/* Type aliases for common PDB vector types.  */
using pdb_scope_stack = std::vector<std::pair<CORE_ADDR, struct symbol *>>;
using pdb_range_pair_vec = std::vector<std::pair<CORE_ADDR, CORE_ADDR>>;

/* Owned stream buffer returned by pdb_read_stream.  */
using gdb_byte_ptr = std::unique_ptr<gdb_byte[]>;

#define UINT32_SIZE	  sizeof (uint32_t)
#define UINT32_PTR_SIZE	  sizeof (uint32_t*)
#define SIZEOF_GDB_BYTE   sizeof (gdb_byte *)
#define INT32_CAST(a)	  *(int32_t*)((a))
#define UINT32_CAST(a)	  *(uint32_t*)((a))
#define INT16_CAST(a)	  *(int16_t*)((a))
#define UINT16_CAST(a)	  *(uint16_t*)((a))
#define INT8_CAST(a)	  *(int8_t*)((a))
#define UINT8_CAST(a)	  *(uint8_t*)((a))
#define UINT32_PTR(a)	  (uint32_t*)((a))
#define UINT16_PTR(a)	  (uint16_t*)((a))
#define UINT8_PTR(a)	  (uint8_t*)((a))
#define UINT64_CAST(a)    *(uint64_t*)((a))
#define CSTR(data)        ((const char *)(data))

/* Debug verbosity level: 0 = off, 1 = basic, 2 = verbose, 3 = trace  */
extern unsigned int pdb_read_debug;
extern unsigned int pdb_convert_debug;

/* Print a "pdb" debug statement if pdb_read_debug is >= 1.  */
#define pdb_dbg_printf(fmt, ...) \
  debug_prefixed_printf_cond (pdb_read_debug >= 1, "pdb", fmt, \
			      ##__VA_ARGS__)

#define pdb_dbg_printf_v(fmt, ...) \
  debug_prefixed_printf_cond (pdb_read_debug >= 2, "pdb", fmt, \
			      ##__VA_ARGS__)

#define pdb_dbg_printf_t(fmt, ...) \
  debug_prefixed_printf_cond (pdb_read_debug >= 3, "pdb", fmt, \
			      ##__VA_ARGS__)

/* Error — fatal issue that stops the operation.  */
#define pdb_error(fmt, ...) \
  error (_("PDB Error: " fmt), ##__VA_ARGS__)

/* Warning — important one-time message that allows continuation.  */
#define pdb_warning(fmt, ...) \
  warning (_("PDB Warning: " fmt), ##__VA_ARGS__)

/* Complaint — recoverable diagnostic issue (repeatable, suppressible).  */
#define pdb_complaint(fmt, ...) \
  complaint (_("PDB Complaint: " fmt), ##__VA_ARGS__)

/* MSF SuperBlock is the first block in the PDB file and contains some basic
   information like the block size, number of blocks, and most importantly
   the location of the stream directory, which is used to locate all the
   other streams in the file. The SuperBlock is 64 bytes long, but we only
   need a few fields from it.  Each stream is split across multiple blocks
   as recorded in this directory. The directory itself might be split into
   multiple blocks which is described by the block map in the MSF SuperBlock.
   https://llvm.org/docs/PDB/MsfFile.html
*/
#define PDB_MSF_MAGIC      "Microsoft C/C++ MSF 7.00\r\n\x1A\x44\x53\x00\x00"
#define PDB_MSF_MAGIC_SIZE  32
#define PDB_MSF_MAGIC_OFFS   0

#define MSF_HEADER_SIZE  64

#define MSF_BLOCK_SIZE_OFFS           32 /* Block size in bytes */
#define MSF_NUM_BLOCKS_OFFS           40 /* Number of blocks */
#define MSF_NUM_DIRECTORY_BYTES_OFFS  44 /* Number of directory bytes */
#define MSF_BLOCK_MAP_ADDR_OFFS       52 /* Block map address */

/* PDB streams */
#define PDB_STREAM_PDB  1
#define PDB_STREAM_TPI  2
#define PDB_STREAM_DBI  3
#define PDB_STREAM_IPI  4
#define PDB_NO_STREAM  0xFFFFFFFF /* Invalid stream - microsoft-pdb/msf.cpp */

/* Size of the per-record header: uint16_t RecordLen + uint16_t RecordKind.  */
#define CV_REC_HDR_SIZE  4

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
	- PresentWords [PresentWordCount]
	- deleted bit vector:
	- DeletedWordCount (4 bytes)
	- DeletedWords [DeletedWordCount]
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
#define INFO_STREAM_HASH_MAP_OFFS  28 /* Offset of the Named hash map  */
#define INFO_STREAM_MIN_SIZE       28 /* Used for checks */

#define NAMES_MAP_STRBUF_SIZE_OFFS     0 /* StringBuffer size */
#define NAMES_MAP_STRBUF_OFFS          4 /* StringBuffer */
#define NAMES_MAP_UINT32_SIZE          4 /* uint32_t size */

#define HASH_TABLE_SIZE_OFFS            0 /* Hash table size */
#define HASH_TABLE_CAPACITY_OFFS        4 /* Hash table capacity */
#define HASH_TABLE_PRESENT_CNT_OFFS     8 /* PresentWordCount (4 bytes) */
#define HASH_TABLE_PRESENT_WORD_OFFS   12 /* Hash table header size */
#define HASH_TABLE_BIT_WORD_SIZE       32 /* Bits per uint32_t word */
#define HASH_TABLE_BYTES_PER_WORD       4 /* Bytes per uint32_t word */

#define HASH_TABLE_ENTRY_SIZE 8 /* Hash entry size (key + value) */
#define HASH_TABLE_KEY_SIZE   4 /* Hash entry key size */
#define HASH_TABLE_VAL_SIZE   4 /* Hash entry value size */

#define NAMES_STREAM_SIGNATURE_OFFS  0 /* Signature ( 4 bytes ) */
#define NAMES_STREAM_DATA_SIZE_OFFS  8 /* Names stream size (4 bytes) */
#define NAMES_STREAM_SIGNATURE_VAL  0xEFFEEFFE /* Expected signature */
#define NAMES_STREAM_MIN_SIZE  12 /* Minimum size (12 bytes) */


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
#define DBI_HDR_SIGNATURE_OFFS         0 /* Version Signature (4 bytes) */
#define DBI_HDR_VERSION_OFFS           4 /* Version Header (4 bytes) */
#define DBI_HDR_AGE_OFFS               8 /* Age (4 bytes) */
#define DBI_HDR_GSI_STREAM_OFFS       12 /* Global Symbol Stream Index (2 bytes) */
#define DBI_HDR_BUILD_NUMBER_OFFS     14 /* Build Num. (2 bytes) */
#define DBI_HDR_PSGSI_STREAM_OFFS     16 /* Public Symbol Stream Index (2 bytes) */
#define DBI_HDR_PDB_DLL_VERSION_OFFS  18 /* Dll Ver. (2 bytes) */
#define DBI_HDR_SYM_RECORD_STREAM_OFFS 20 /* Symbol Record Stream (2 bytes) */
#define DBI_HDR_MODULE_SIZE_OFFS      24 /* Module Info Size (4 bytes) */
#define DBI_HDR_SECT_CONTRIB_OFFS     28 /* Section Contrib Size (4 bytes) */
#define DBI_HDR_SECT_MAP_OFFS         32 /* Section Map Size (4 bytes) */
#define DBI_HDR_SOURCE_INFO_OFFS      36 /* File Info Size (4 bytes) */

#define DBI_HDR_SIZE                  64 /* Total header size */

#define DBI_HDR_SIGNATURE_VAL  0xFFFFFFFF /* Expected signature value */
#define DBI_HDR_MODULE_INFO_OFFS  DBI_HDR_SIZE /* Substreams start here */

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
#define FILE_INFO_HDR_SIZE              4 /* Total header size  */
#define FILE_INFO_MOD_INDICES_OFFS     FILE_INFO_HDR_SIZE
/* Number of modules (2 bytes).  */
#define FILE_INFO_HDR_NUM_MODULES_OFFS  0
#define FILE_INFO_ELEMENT_SIZE          2 /* Each module entry (2 bytes) */

/* Section Contribution offsets  */
#define MODI_SC_ISECT_OFFS       4  /* Section identifier (2 bytes)  */
#define MODI_SC_OFFSET_OFFS      8  /* Offset in the section (4 bytes) */
#define MODI_SC_SIZE_OFFS       12  /* Size of the contribution (4 bytes) */

/* Module Info Header (ModInfo in microsoft-pdb) specifes the sizes
   of various debug data blocks in the corresponding module stream.
   Total size is 64 bytes plus the variable-length names.  */
#define MODI_FLAGS_OFFS           32  /* Flags (2 bytes) */
#define MODI_STREAM_NUM_OFFS      34  /* ModuleSymStream (2 bytes) */
#define MODI_SYM_BYTES_OFFS       36  /* Symbol stream size (4 bytes) */
#define MODI_C11_BYTES_OFFS       40  /* C11 Bytes Size (4 bytes) */
#define MODI_C13_BYTES_OFFS       44  /* C13 Bytes Size (4 bytes) */
#define MODI_SRC_FILE_COUNT_OFFS  48  /* Num. of Source files (2 bytes) */
#define MODI_FIXED_SIZE           64  /* Size of the fixed part of the header */

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

#define C13_SUBSECT_HEADER_SIZE    8

/* CV Signatures */
#define CV_SIGNATURE_SIZE 4
#define CV_SIGNATURE_C6   0L
#define CV_SIGNATURE_C7   1L
#define CV_SIGNATURE_C11  2L
#define CV_SIGNATURE_C13  4L

/* C13 subsection types we support. */
#define DEBUG_S_LINES      0xF2
#define DEBUG_S_FILECHKSMS 0xF4

/* C13 File block */
struct CV_FileBlock
{
  uint32_t file_id;       /* File checksum index */
  uint32_t num_lines;     /* Number of lines */
  uint32_t block_size;    /* Block size */
};

#define PDB_FILECHKSUM_HDR_SIZE  sizeof(struct CV_FileBlock)

/* File checksum entry - used to access file names in the string table created
   from the /names stream.  */
struct CV_FileChecksum
{
  uint32_t file_name_offset;  /* Offset in string table */
  uint8_t checksum_size;
  uint8_t checksum_type;
  /* Followed by checksum bytes */
};

/* RSDS data from the PE CodeView record.  */
struct pdb_rsds_info
{
   bool valid = false;
   gdb_byte guid[16] = {};
   uint32_t age = 0;
   std::string pdb_path;
};

/* Main entry point for reading PDB debug info for the objfile.
   Reads the PDB file and processes its debug information and associates
   it with the objfile. Returns true if PDB data was found and processed.
   Called from coff_symfile_read, analogous to dwarf2_initialize_objfile.  */
extern bool pdb_initialize_objfile (struct objfile *objfile);

/* Find the PDB file for the given objfile.
    1. --pdb-path command-line override. Applies to main executable only.
    2. PDB basename from EXE RSDS record in the EXE's directory.
    3. EXE path with extension replaced by .pdb.
    4. Full embedded path from RSDS record.
    5. Search path from _NT_SYMBOL_PATH / _NT_ALT_SYMBOL_PATH env. vars
       See https://learn.microsoft.com/en-us/windows/win32/debug/symbol-paths
       and microsoft-pdb locator.cpp on details how the paths are searched.
       TODO: SRV, SYMSRV, CACHE entries need symbol server fetch via HTTP.
    6. Windows registry (HKCU then HKLM). Same docs as above.
       TODO: Entries also can have  RV, SYMSRV, CACHE entries needing the symbol
       server fetch via HTTP.  */
extern std::string pdb_find_pdb_file (struct objfile *objfile);

/* Overrides the normal PDB search with this path.
   Set by the --pdb-path command-line argument.  */
extern const char *pdb_file_override;

/* Per-file info from DEBUG_S_FILECHKSMS.  */
struct pdb_file_info
{
  const char *filename;     /* Resolved from string table */
  uint8_t checksum_type;    /* 0=none, 1=MD5, 2=SHA1, 3=SHA256 */
  uint8_t checksum_size;
  const gdb_byte *checksum; /* Points into file_checksums (NULL if none) */
};

/* PDB Module information */
struct pdb_module_info
{
  uint32_t module_index;  /* Index in pdb->modules array */
  char *module_name;      /* As recorded in DBI Module Info Header */
  char *obj_file_name;    /* As recorded in DBI Module Info Header */

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
  struct compunit_symtab *cu;

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
  uint32_t *file_name_offsets;

  /* File list obtained from C13 records (for info pdb-files-c13 command)
     TODO: Consider removing this if the command is no longer necessary.
     But also, need to consider using this file list instead of the one
     obtained from the File Info substream; even though LLVM uses the File
     Info substream info, that one can store only 65535 files per module.
     This might sound sufficient but PDB has another variable that stores
     up to 65535 files in total, and was abandoned due to this limitation.
     The limitation doesn't seem to be sufficiently addressed by this change.  */
  bool files_c13_read;
  struct pdb_file_info *files_c13;
  uint32_t num_files_c13;
};

/* Raw CodeView type record from TPI/IPI stream.  */
struct pdb_tpi_type
{
  uint16_t leaf;               /* CodeView record type (LF_*) */
  uint16_t length;             /* RecordLen field (length of data) */
  const gdb_byte *data;        /* Pointer to record data (RecordKind + fields) */
  uint32_t data_len;           /* Size of data (length - 2 for RecordKind) */
};

/* TPI (Type Program Information) context.
   Parsed type records and caching.  */
struct pdb_tpi_context
{
  /* TPI stream types.  */
  struct pdb_tpi_type *types;  /* Array of parsed type records */
  uint32_t type_idx_begin;     /* First type index in TPI  */
  uint32_t type_idx_end;       /* One past last type index */

  /* Type cache. Covers both simple types (0x0000-0x0FFF) and compound types
     (0x1000..type_idx_end).  */
  struct type **type_cache;

  /* Type used for unsupported types.  */
  struct type *undefined_type;
};

/* PDB per-objfile data */
struct pdb_per_objfile
{
  /* Back link to objfile */
  struct objfile *objfile;

   /* Path of the loaded PDB file on disk.  */
   const char *pdb_file_path;

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
  struct pdb_module_info *modules;

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
  uint16_t gsi_stream;          /* Global Symbol Information stream index */
  uint16_t psgsi_stream;        /* Public Symbol Information stream index */

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

  /* Parsed TPI and IPI streams. Check tpi.types to verify parsing succeeded.  */
  struct pdb_tpi_context tpi;

  /* GSI hash table: O(1) name → module lookup for lazy expansion.
     Uses the GSI stream's bucket table directly — no data copied.  */
  struct pdb_gsi_table *gsi_table;
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
  struct CV_LineSection *line_sect;
  struct CV_Line *lines;
  uint32_t num_lines;
};

/* Callback type for pdb_walk_c13_line_blocks. The callback is used
   when processing C13 line info as we have multiple users of that data.
   One of the users is info pdb-files-c13 command which might not be necessary.
   If not, drop the callback as well. */
typedef void (*pdb_line_block_fn) (void *cb_data);

/* Register a loaded PDB instance for later use by info commands.
   It also sets the default PDB for info command, which is used when the
   command doesn't specify the PDB. The default PDB is the one associated
   with the OBJF_MAINLINE marked object file.  */
extern void
pdb_register_loaded_pdb (struct pdb_per_objfile *pdb);

/* Expand a single module into its own compunit_symtab. Returns the cached CU
   on subsequent calls. Returns NULL if the module has no useful debug data. */
extern struct compunit_symtab *pdb_build_module (struct pdb_per_objfile *pdb,
						 struct pdb_module_info *mod);

/* Flags for pdb_parse_symbols.  */
#define PDB_DUMP_SYM         0x1  /* Dump records (GDB symbols not created).  */

/* Parse CodeView symbol records from a module stream into GDB symbols.
   MODULE_STREAM is the raw module stream data (caller owns it).
   When flags includes PDB_DUMP_SYM, records are dumped to stdout
   instead of creating GDB symbols.
   If FUNC_RANGES is non-null, each function's [start, end) is appended
   so the caller can register discontiguous block ranges.  */
extern void pdb_parse_symbols (struct pdb_per_objfile *pdb,
			       struct pdb_module_info *module,
			       gdb_byte *module_stream,
			       struct buildsym_compunit *cu,
			       uint32_t flags = 0,
			       pdb_range_pair_vec *func_ranges = nullptr);

/* Read the symbol record stream into pdb->sym_record_data.  */
extern void pdb_read_sym_record_stream (struct pdb_per_objfile *pdb);

/* Walk the SymRecordStream and create GDB symbols for S_GDATA32/S_LDATA32
   global variable records, adding them to CU.  Called once per PDB load
   to populate globals that live outside any module stream.  */
extern void pdb_load_global_syms (struct pdb_per_objfile *pdb,
				  struct buildsym_compunit *cu);

/* Parse CodeView symbol records from the global symbol record stream
   (SymRecordStream from DBI header).  Contains S_PUB32 and other globals.
   When flags includes PDB_DUMP_SYM, records are dumped to stdout.  */
extern void pdb_parse_sym_record_stream (struct pdb_per_objfile *pdb,
					 uint32_t flags = 0);

/* Dump GSI hash records — reads the GSI stream and resolves each hash
   record's offset into the SymRecordStream, dumping the referenced symbol.  */
extern void pdb_dump_gsi_stream (struct pdb_per_objfile *pdb);

/* A gap within a location entry (range where a variable is unavailable)  */
struct pdb_loc_gap
{
  CORE_ADDR start;  /* Start of gap  */
  CORE_ADDR end;    /* End of gap  */
};

/* Location entry (one DEFRANGE record) in pdb_loclist_baton.
   Resolved at parse time. Stored as a linked list on the objfile obstack.
   Gap array follows the struct inline.  */
struct pdb_loc_entry
{
  struct pdb_loc_entry *next;
  CORE_ADDR start;       /* Start PC  */
  CORE_ADDR end;         /* End PC    */
  int gdb_regnum;        /* GDB register number (-1 = unsupported) */
  int32_t offset;        /* Byte offset from register */
  bool is_register;      /* Value is the register itself (not memory) */
  bool is_full_scope;    /* Matches entire function, no range check */
  int num_gaps;          /* Number of gaps in the gaps[] array */
  struct pdb_loc_gap gaps[];  /* Inline gap array */
};

/* Per-symbol location baton.
   Contains a linked list of parsed location entries, each fully
   resolved at parse time.  At read time we walk the list and find
   the entry matching the current PC. Allocated on the objfile obstack.  */
struct pdb_loclist_baton
{
  /* Linked list of parsed location entries (nullptr if none).  */
  struct pdb_loc_entry *entries;

  /* Back-pointer needed by describe_location.  */
  struct pdb_per_objfile *pdb;
};

/* Return true if SYM has a PDB-computed location (pdb_loclist_funcs).  */
extern bool pdb_is_pdb_location (struct symbol *sym);

/* Dump resolved variable location batons from a built compunit_symtab.
   Walks all blocks, finds symbols with pdb_loclist_baton, and prints
   each pdb_loc_entry (range, register, offset, gaps).
   If SYMBOL_FILTER is non-null, only prints the matching symbol.  */
extern void pdb_dump_locations (struct compunit_symtab *cust,
				struct gdbarch *gdbarch,
				const char *symbol_filter);

/* Parse and optionally dump a single symbol record from the SymRecordStream.
   If PDB_DUMP_SYM is set in FLAGS, calls the record's dump() method.  */
extern void pdb_dump_parse_record (struct pdb_per_objfile *pdb,
				   gdb_byte *rec_data, uint16_t rectype,
				   uint16_t reclen, uint32_t flags);

/* Dump PSGSI (public symbol) hash records — reads the PSGSI stream and
   resolves each hash record into the SymRecordStream.  */
extern void pdb_dump_psgsi_stream (struct pdb_per_objfile *pdb);

/* Build minimal symbols from S_PUB32 records in the PSI stream.
   Called during PDB init after sections are available.  */
extern void pdb_build_minsyms (struct pdb_per_objfile *pdb);

/* Initialize the GSI hash table from the GSI stream.  Parses the
   stream header and caches pointers to the hash records, bitmap,
   and bucket offsets for O(1) lookup.  No data is copied.  */
extern void pdb_init_gsi_table (struct pdb_per_objfile *pdb);

/* Look up a symbol by name in the GSI hash table.  Returns true and
   fills RESULT with the S_PROCREF/S_DATAREF entry if found.  Uses
   the GSI bucket table for O(1) lookup.  */
extern bool pdb_gsi_lookup (struct pdb_per_objfile *pdb,
			    const char *name,
			    struct pdb_gsi_entry *result);

/* Register the PDB location list implementation with the
   symbol_impl table.  Called once from INIT_GDB_FILE.  */
extern void pdb_init_loclist (void);

/* Get module's files. Parses the DBI File Info substream and resolves
   file names into module->files[]. Populates module->num_files and
   module->files on first call; subsequent calls are no-ops.  */
extern void pdb_read_module_files (struct pdb_per_objfile *pdb,
				   struct pdb_module_info *module);

/* Read a module's debug stream and extract file checksums onto the
   objfile obstack.  Returns an owned buffer with the raw stream data;
   the caller keeps it alive as long as the data is needed.  Returns
   nullptr if the module has no stream.  */
extern gdb_byte_ptr pdb_read_module_stream (struct pdb_per_objfile *pdb,
					    struct pdb_module_info *module);

/* Resolve the i-th filename for a module from the shared File Info
   substream. Returns NULL if index is out of range or the substream
   is not loaded. Caller must call pdb_read_module_files first.  */
extern const char *pdb_module_file_name (struct pdb_per_objfile *pdb,
					 struct pdb_module_info *module,
					 uint32_t index);

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
extern void pdb_read_module_files_c13 (struct pdb_per_objfile *pdb,
				       struct pdb_module_info *module);

/* Walk all DEBUG_S_LINES file blocks in a module's C13 data,
   invoking CALLBACK for each one.  */
extern void pdb_walk_c13_line_blocks (struct pdb_per_objfile *pdb,
				      struct pdb_module_info *module,
				      gdb_byte *module_stream,
				      pdb_line_block_fn callback,
				      void *cb_data);

/* Map a (section, offset) pair to a relocated PC. Returns 0 on
   failure (invalid section index or missing section addresses).  */
extern CORE_ADDR pdb_map_section_offset_to_pc (struct pdb_per_objfile *pdb,
					       uint16_t section,
					       uint32_t offset);

/* TODO: Symbol server fetcher */
std::string pdb_symserver_fetch(const std::string &entry,
				const std::string &pdb_name,
				const struct pdb_rsds_info &rsds);

/* Try to open PATH for reading. Returns true if the file exists.  */
bool pdb_try_open (const char *path);

/* Extract the directory portion of PATH (including trailing separator).
   Returns empty string if PATH has no directory component.  */
std::string pdb_dirname(const char *path);

/* Read a stream by index. Returns an owned buffer that the caller must
   keep alive as long as the data is needed. To cache a stream for the
   lifetime of the objfile, release the pointer into
   pdb->stream_data[stream_idx].  Returns nullptr if the stream doesn't
   exist.  */
extern gdb_byte_ptr pdb_read_stream (struct pdb_per_objfile *pdb,
				     uint32_t stream_idx);

/* ---------------------------------------------------------------
   CodeView Simple Type Constants (cvinfo.h)
   Simple type indices < 0x1000 encode: kind (bits 0-7) + mode (bits 8-11)
   ---------------------------------------------------------------  */

#define CV_SIMPLE_KIND(ti)  ((ti) & 0xFF)        /* Extract kind from type index */
#define CV_SIMPLE_MODE(ti)  (((ti) >> 8) & 0x0F) /* Extract mode from type index */
#define CV_TI_IS_SIMPLE(ti) ((ti) < 0x1000)      /* Check if type index is simple */

/* SimpleTypeKind values */
#define CV_NONE              0x00
#define CV_VOID              0x03
#define CV_HRESULT           0x08
#define CV_SIGNED_CHAR       0x10
#define CV_UNSIGNED_CHAR     0x20
#define CV_NARROW_CHAR       0x70
#define CV_WIDE_CHAR         0x71
#define CV_CHAR16            0x7a
#define CV_CHAR32            0x7b
#define CV_SBYTE             0x11
#define CV_BYTE              0x21
#define CV_INT16             0x72
#define CV_UINT16            0x73
#define CV_INT32             0x74
#define CV_UINT32            0x75
#define CV_INT64             0x76
#define CV_UINT64            0x77
#define CV_FLOAT32           0x40
#define CV_FLOAT64           0x41
#define CV_FLOAT80           0x42
#define CV_BOOL8             0x30

/* SimpleTypeMode values */
#define CV_TM_DIRECT     0  /* No indirection */
#define CV_TM_NPTR       1  /* Near pointer */
#define CV_TM_NPTR32     4  /* 32-bit near pointer */
#define CV_TM_FPTR32     5  /* 32-bit far pointer */
#define CV_TM_NPTR64     6  /* 64-bit near pointer */
#define CV_TM_NPTR128    7  /* 128-bit near pointer */

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

#define GSI_HASH_HDR_SIZE  16

#define PSGSI_HDR_SIZE              28
#define PSGSI_HDR_SYM_HASH_OFFS     0
#define PSGSI_HDR_ADDR_MAP_OFFS     4
//#define PSGSI_HDR_NUM_THUNKS_OFFS   8
//#define PSGSI_HDR_SIZE_OF_THUNK_OFFS 12
//#define PSGSI_HDR_ISECT_THUNK_TABLE_OFFS 16
//#define PSGSI_HDR_PADDING_OFFS       18
//#define PSGSI_HDR_OFF_THUNK_TABLE_OFFS 20
//#define PSGSI_HDR_NUM_SECTIONS_OFFS 24

#define GSI_HASH_HDR_SIG_VAL     0xFFFFFFFF
#define GSI_HASH_HDR_VER_VAL     (0xeffe0000 + 19990810)
#define GSI_HASH_RECORD_SIZE     8
#define GSI_HASH_RECORD_SYMOFFS_OFFS 0
//#define GSI_HASH_RECORD_CREF_OFFS   4

/* GSI hash header (GSIHashHdr in microsoft-pdb gsi.h).
   For PSGSI, it comes after PublicsStreamHeader main header.  */
struct pdb_gsi_hdr
{
  uint32_t sig;           /* VerSignature (always 0xFFFFFFFF).  */
  uint32_t ver;           /* VerHdr (always 0xeffe0000 + 19990810).  */
  uint32_t hr_bytes;      /* Byte count of HashRecord entries.  */
  uint32_t bucket_bytes;  /* Byte count of bucket data.  */
  gdb_byte *hr_data;	  /* Start of HashRecord array.  */
  gdb_byte *bucket_data;  /* Start of bucket data (after records).  */
  bool valid;             /* True if parsing succeeded.  */
};

/* Validate and parse a GSI hash region at DATA with DATA_SIZE bytes.
   Fills HDR on success and returns true.  The data pointer must
   point to the GSIHashHdr (i.e. past any PublicsStreamHeader).  */
extern struct pdb_gsi_hdr pdb_parse_gsi_hash (gdb_byte *data, uint32_t data_size);

/* GSI hash bucket constants (microsoft-pdb GSI1.h).  */
#define GSI_NUM_BUCKETS   4097   /* IPHR_HASH + 1 */
#define GSI_BITMAP_WORDS  129    /* ceil(GSI_NUM_BUCKETS / 32) */

/* Forward declarations — definitions are in pdb-cooked-index.c.  */
struct pdb_gsi_table;
struct pdb_gsi_entry;

/* ---------------------------------------------------------------
   CodeView Symbol Record Types
   ---------------------------------------------------------------  */
#define S_GPROC32       0x1110  /* Global procedure */
#define S_LPROC32       0x110f  /* Local procedure */
#define S_GPROC32_ID    0x1147  /* Global procedure with ID */
#define S_LPROC32_ID    0x1146  /* Local procedure with ID */
#define S_GDATA32       0x110d  /* Global data */
#define S_LDATA32       0x110c  /* Local data */
#define S_LOCAL         0x113e  /* Local variable */
#define S_LOCAL32       0x113e  /* Local variable (same as S_LOCAL) */
#define S_UDT           0x1108  /* User-defined type */
#define S_BLOCK32       0x1103  /* Block scope */
#define S_REGREL32      0x1111  /* Register-relative variable (LLVM uses this) */
#define S_PUB32         0x110e  /* Public symbol */
#define S_REGISTER      0x1106  /* Register variable */
#define S_CONSTANT      0x1107  /* Constant value */
#define S_LABEL32       0x1105  /* Code label */
#define S_END           0x0006  /* End of scope */
#define S_INLINESITE    0x114d  /* Inlined function site (opens scope) */
#define S_INLINESITE_END 0x114e /* End of inline site (closes scope) */
#define S_PROC_ID_END   0x114f  /* End of procedure (closes scope) */
#define S_THUNK32       0x1102  /* Thunk (opens scope) */
#define S_PROCREF       0x1125  /* Reference to procedure in module stream */
#define S_LPROCREF      0x1127  /* Reference to local procedure in module stream */

/* TODO: Unsupported CodeView symbol types  */
#define S_OBJNAME       0x1101  /* Object file name */
#define S_COMPILE2      0x1116  /* Compiler info v2 */
#define S_DATAREF       0x1126  /* Reference to data in module stream */
#define S_ANNOTATIONREF 0x1128  /* Reference to annotation */
#define S_COMPILE3      0x113c  /* Compiler info v3 */
#define S_ENVBLOCK      0x113d  /* Environment block (key/value pairs) */
#define S_UNAMESPACE    0x1124  /* Using namespace */
#define S_BPREL32       0x110b  /* BP-relative variable */
#define S_LTHREAD32     0x1112  /* Local thread-local storage */
#define S_GTHREAD32     0x1113  /* Global thread-local storage */
#define S_LMANDATA      0x111c  /* Managed local data */
#define S_GMANDATA      0x111d  /* Managed global data */
#define S_BUILDINFO     0x114c  /* Build information */
#define S_FRAMEPROC     0x1012  /* Frame procedure info */
#define S_CALLSITEINFO  0x1139  /* Call site information */
#define S_FILESTATIC    0x1153  /* File-scoped static */
#define S_EXPORT        0x1138  /* Exported symbol */
#define S_SECTION       0x1136  /* PE section info */
#define S_COFFGROUP     0x1137  /* COFF group info */
#define S_TRAMPOLINE    0x112c  /* Trampoline thunk */
#define S_FRAMECOOKIE   0x113a  /* Security cookie on stack frame */
#define S_HEAPALLOCSITE 0x115e  /* Heap allocation site */
#define S_CALLEES       0x115a  /* Callee list */
#define S_CALLERS       0x115b  /* Caller list */
#define S_POGODATA      0x115c  /* Profile-guided optimization data */
#define S_INLINESITE2   0x115d  /* Inline site v2 */
#define S_TOKENREF      0x1129  /* Managed token reference */
#define S_GMANPROC      0x112a  /* Managed global procedure */
#define S_LMANPROC      0x112b  /* Managed local procedure */
#define S_COBOLUDT      0x1109  /* COBOL UDT */
#define S_MANCONSTANT   0x112d  /* Managed constant */
#define S_SEPCODE       0x1132  /* Separated code */
#define S_DISCARDED     0x113b  /* Discarded symbol */
#define S_ANNOTATION    0x1019  /* Annotation */
#define S_DEFRANGE      0x113f  /* Def range */
#define S_DEFRANGE_SUBFIELD 0x1140  /* Def range subfield */
#define S_DEFRANGE_SUBFIELD_REGISTER 0x1143  /* Subfield in register */
#define S_LOCAL_2005    0x1133  /* Local variable (2005 format) */
#define S_DEFRANGE_2005 0x1134  /* Def range (2005 format) */
#define S_DEFRANGE2_2005 0x1135  /* Def range v2 (2005 format) */
#define S_ARMSWITCHTABLE 0x1159  /* ARM switch table / jump table */
#define S_MOD_TYPEREF   0x115f  /* Module type reference */
#define S_REF_MINIPDB   0x1160  /* Reference to mini PDB */
#define S_PDBMAP        0x1161  /* PDB path mapping */
#define S_LPROC32_DPC   0x1155  /* DPC local procedure */
#define S_LPROC32_DPC_ID 0x1156 /* DPC local procedure with ID */
#define S_INLINEES      0x1168  /* Inlinee list */
#define S_FASTLINK      0x1167  /* Fast link info */
#define S_HOTPATCHFUNC  0x1169  /* Hot-patch function */
#define S_FRAMEREG      0x1166  /* Frame register */
#define S_ATTR_FRAMEREL 0x112e  /* Attributed frame-relative */
#define S_ATTR_REGISTER 0x112f  /* Attributed register */
#define S_ATTR_REGREL   0x1130  /* Attributed register-relative */
#define S_ATTR_MANYREG  0x1131  /* Attributed many-register */
#define S_VFTABLE32     0x100c  /* Virtual function table */
#define S_WITH32        0x1104  /* WITH scope */
#define S_MANYREG       0x110a  /* Many-register variable */
#define S_MANYREG2      0x1117  /* Many-register variable v2 */
#define S_LOCALSLOT     0x111a  /* Local managed slot */
#define S_PARAMSLOT     0x111b  /* Parameter managed slot */

#define S_DEFRANGE_REGISTER           0x1141  /* Variable in register */
#define S_DEFRANGE_FRAMEPOINTER_REL   0x1142  /* FP-relative */
#define S_DEFRANGE_SUBFIELD_REGISTER  0x1143  /* Sub-field in register */
#define S_DEFRANGE_FRAMEPOINTER_REL_FULL_SCOPE 0x1144  /* FP-relative, valid entire function */
#define S_DEFRANGE_REGISTER_REL       0x1145  /* Register-relative */

/* Symbol record.  Layout:
     uint16_t reclen;   (length of data following reclen)
     uint16_t rectype;  (symbol type)
     ...  record-type-specific data ...
*/
#define PDB_RECORD_LEN_OFFS  0
#define PDB_RECORD_TYPE_OFFS 2
#define PDB_RECORD_DATA_OFFS 4
#define PDB_RECORD_HDR_SIZE  4

/* ---------------------------------------------------------------
   CodeView Leaf Type Record Types (LF_* values for TPI stream)
   ---------------------------------------------------------------  */

#define LF_MODIFIER    0x1001  /* Type modifier (const, volatile) wrapping base type */
#define LF_POINTER     0x1002  /* Pointer to another type */
#define LF_PROCEDURE   0x1008  /* Function/procedure type */
#define LF_MFUNCTION   0x1009  /* Member function type */
#define LF_ARGLIST     0x1201  /* Argument list for procedures */
#define LF_FIELDLIST   0x1203  /* Field list for structs/classes */
#define LF_BITFIELD    0x1205  /* Bit field */
#define LF_ARRAY       0x1503  /* Array type */
#define LF_CLASS       0x1504  /* Class type */
#define LF_STRUCTURE   0x1505  /* Structure type */
#define LF_UNION       0x1506  /* Union type */
#define LF_ENUM        0x1507  /* Enumeration type */
#define LF_INTERFACE   0x1508  /* Interface type */

/* Return a human-readable name for a CodeView symbol record type.  */
extern std::string pdb_sym_rec_type_name (uint16_t rectype);

/* Read and parse TPI (stream 2) and IPI (stream 4) if present.
   Populates pdb->tpi->types[] with decoded type records.  */
bool pdb_read_tpi_stream (struct pdb_per_objfile *pdb);

/* Read a CodeView numeric leaf.
   Returns number of bytes consumed (0 on error).  */
uint32_t pdb_cv_read_numeric (const gdb_byte *data, uint32_t max_len,
			      uint64_t *value);

/* Resolve type index to GDB type.
   Handles both simple (< 0x1000) and TPI-stream types.  */
struct type *pdb_tpi_resolve_type (struct pdb_per_objfile *pdb,
				   uint32_t type_index);

/* Return the number of parameters for a function type index.
   For LF_MFUNCTION, includes the implicit 'this' parameter.
   Returns 0 if type_idx is not a function type.  */
int pdb_tpi_get_func_param_count (struct pdb_per_objfile *pdb,
				  uint32_t type_idx);

bool
pdb_parse_sym_record_hdr (gdb_byte *rec, gdb_byte *end, uint16_t &reclen,
			  uint16_t &rectype, size_t &rec_size);

#endif /* GDB_PDB_PDB_H */
