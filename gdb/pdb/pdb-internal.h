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
#include "gdbsupport/filestuff.h"
#include "gdbsupport/byte-vector.h"
#include "gdbsupport/function-view.h"

#include <array>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <string_view>

struct objfile;

namespace pdb
{

struct pdb_per_objfile;

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

/* "set debug pdb-read": reader debug messages when non-zero.  */
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
inline constexpr auto PDB_STREAM_DBI = 3;
/* Invalid stream - microsoft-pdb/msf.cpp.  */
inline constexpr auto PDB_NO_STREAM = 0xFFFFFFFF;

/* The PDB info stream (stream 1) contains version/age/GUID followed by
   the "named stream map" which is a hash table that maps stream identifiers
   (e.g. "/names", "/LinkInfo") to MSF stream numbers.  Except for a few fixed
   streams like PDB info, TPI, IPI and DBI, all other streams have arbitrary
   Id and need to be found by name.

   PDB info stream layout:
     0:  Version
     4:  Signature
     8:  Age
     12: GUID
     28: Named stream map.  */

/* Offset of the named stream map.  */
inline constexpr auto INFO_STREAM_HASH_MAP_OFFS = 28;
/* Fixed Info header size.  */
inline constexpr auto INFO_STREAM_MIN_SIZE = 28;

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
/* Module Info Size (4 bytes).  */
inline constexpr auto DBI_HDR_MODULE_SIZE_OFFS = 24;

/* Total header size.  */
inline constexpr auto DBI_HDR_SIZE = 64;

/* Expected signature value.  */
inline constexpr auto DBI_HDR_SIGNATURE_VAL = 0xFFFFFFFF;
/* Substreams start here.  */
inline constexpr auto DBI_HDR_MODULE_INFO_OFFS = DBI_HDR_SIZE;

/* Section Contribution offsets within a Module Info header.  */

/* Section identifier (2 bytes).  */
inline constexpr auto MODI_SC_ISECT_OFFS = 4;
/* Offset in the section (4 bytes).  */
inline constexpr auto MODI_SC_OFFSET_OFFS = 8;
/* Size of the contribution (4 bytes).  */
inline constexpr auto MODI_SC_SIZE_OFFS = 12;

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

/* PDB Module information.  */
struct pdb_module_info
{
  /* Module name borrowed from pdb_per_objfile::dbi_data.  */
  const char *module_name;

  /* Module debug stream index; 0xFFFF means no stream.  */
  uint16_t stream_number;
  /* Symbol-area byte count, including the leading CodeView signature.  */
  uint32_t sym_byte_size;
  /* Size of C11 records in the module stream.  */
  uint32_t c11_byte_size;
  /* Size of C13 records in the module stream.  */
  uint32_t c13_byte_size;

  /* The section contribution in the module's Module Info record.
     1-based section index; 0 means none.  */
  uint16_t sc_section;
  /* Offset within section.  */
  uint32_t sc_offset;
  /* Size of contribution in bytes.  */
  uint32_t sc_size;
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
     pointers (module names) stay valid for the objfile's lifetime.  */
  /* PDB_STREAM_PDB.  */
  gdb::byte_vector info_data;
  /* PDB_STREAM_DBI.  */
  gdb::byte_vector dbi_data;
  /* PDB file mapping.  */
  pdb_file_mapping file_mapping;

  /* Module info.  */
  std::vector<pdb_module_info> modules;

  /* Age from the DBI header; 0 in old PDBs.  */
  uint32_t dbi_age = 0;

  /* Copy a stream by index out of the file mapping.  Returns an owned
     buffer.  To cache a stream for the lifetime of the objfile, move the
     result into one of the dedicated owning fields above (info_data,
     dbi_data).  Returns an empty vector if the stream doesn't exist.  */
  gdb::byte_vector read_stream (uint32_t stream_idx) const;
};

/* Return the pdb_per_objfile associated with OBJFILE, or nullptr if
   no PDB data has been loaded for this objfile.  */
extern pdb_per_objfile *get_pdb_per_objfile (objfile *objfile);

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
