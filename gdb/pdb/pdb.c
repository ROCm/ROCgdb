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

   This file implements reading of Microsoft PDB debug information format.
   PDB (Program Database) is the native debug format for Microsoft compilers.
   PDB is a multi-stream file container.  All the data is stored in various
   streams while the streams are composed of blocks which don't have to be
   consecutive.  References:
   - microsoft-pdb: https://github.com/microsoft/microsoft-pdb.
   - LLVM docs: https://llvm.org/docs/PDB/.  */

#include "pdb/pdb.h"

#include "objfiles.h"
#include "complaints.h"
#include "pdb/pdb-internal.h"
#include "cli/cli-cmds.h"
#include "gdbsupport/common-utils.h"
#include "gdbsupport/filestuff.h"
#include "gdbsupport/scope-exit.h"
#include "gdbsupport/selftest.h"

#include <string.h>
#include <algorithm>
#include <vector>
#include <optional>

#ifdef _WIN32
/* Define NOMINMAX so <windows.h> does not clobber std::min / std::max.  */
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

namespace pdb
{

/* PDB read debug statements.  */
unsigned int pdb_read_debug = 0;

/* One PDB loaded per object file.  */
static const registry<objfile>::key<pdb_per_objfile> pdb_objfile_data_key;

/* See pdb-internal.h.  */

pdb_per_objfile *
get_pdb_per_objfile (objfile *objfile)
{
  return pdb_objfile_data_key.get (objfile);
}

/* Get the pdb_per_objfile associated with OBJFILE, creating and registering
   a new one if none exists yet.  */

static pdb_per_objfile *
get_or_create_pdb_per_objfile (objfile *objfile)
{
  return &pdb_objfile_data_key.try_emplace (objfile, objfile);
}

/* Number of MSF blocks needed to hold SIZE bytes.  */

static inline uint32_t
pdb_blocks_for_size (uint32_t size, uint32_t block_size)
{
  gdb_assert (is_power_of_2 (block_size));
  return (uint32_t) (align_up ((ULONGEST) size, (int) block_size)
		     / block_size);
}

/* Read MSF SuperBlock header.  Throws via pdb_error on failure.
   https://llvm.org/docs/PDB/MsfFile.html#the-superblock  */

static void
pdb_read_msf_header (pdb_per_objfile *pdb)
{
  uint64_t file_size = pdb->file_mapping.size;
  if (file_size < PDB_MSF_MAGIC_SIZE
      || memcmp (pdb->file_mapping.base, PDB_MSF_MAGIC.data (),
		 PDB_MSF_MAGIC_SIZE) != 0)
    pdb_error ("Bad MSF signature");

  /* Read the SuperBlock from the file mapping and get the block size
     information in order to be able to read streams.  */
  if (file_size < MSF_HEADER_SIZE)
    pdb_error ("Failed to read MSF SuperBlock");
  const gdb_byte *header = pdb->file_mapping.base;

  pdb->block_size = read_u32 (header + MSF_BLOCK_SIZE_OFFS);
  pdb->num_blocks = read_u32 (header + MSF_NUM_BLOCKS_OFFS);
  pdb->num_dir_bytes = read_u32 (header + MSF_NUM_DIRECTORY_BYTES_OFFS);
  pdb->block_map_addr = read_u32 (header + MSF_BLOCK_MAP_ADDR_OFFS);

  /* Validate block size.  An MSF file has pages of 512 to 32768 bytes, as
     LLVM's msf::isValidBlockSize accepts: Microsoft link and lld-link write
     the larger ones for /PDBPAGESIZE.  */
  if (pdb->block_size < 512 || pdb->block_size > 32768
      || !is_power_of_2 (pdb->block_size))
    pdb_error ("Invalid PDB block size: %u", pdb->block_size);

  /* Compute the shift that turns a stream offset into a block number.
     The block size is a power of two, so a block number and an offset
     inside it are a shift and a mask of a stream offset.  */
  pdb->block_shift = 0;
  while ((pdb->block_size >> pdb->block_shift) > 1)
    pdb->block_shift++;

  /* Check that the file holds all its blocks, which keeps every block
     below num_blocks inside the mapped file.  */
  uint64_t expected_file_size = (uint64_t) pdb->num_blocks * pdb->block_size;
  if (expected_file_size > file_size)
    pdb_error ("PDB file too small: expected at least %" PRIu64 " bytes "
	       "for %u blocks of %u bytes each, but file is only %" PRIu64
	       " bytes",
	       expected_file_size, pdb->num_blocks, pdb->block_size,
	       file_size);
}

/* Read a uint32_t block number from BUF and validate it against
   pdb->num_blocks.  Throws via pdb_error if out of range; on return the
   returned block number is guaranteed to be < pdb->num_blocks.  CONTEXT
   is included in the error message to identify the call site.  */

static uint32_t
pdb_read_block_number (const pdb_per_objfile *pdb, const gdb_byte *buf,
		       const char *context)
{
  uint32_t block_idx = read_u32 (buf);
  if (block_idx >= pdb->num_blocks)
    pdb_error ("%s: block index %u out of range (%u blocks)", context,
	       block_idx, pdb->num_blocks);
  return block_idx;
}

/* Return LEN bytes of STREAM starting at its logical offset OFF.  A range
   inside one block is returned as a pointer into the file mapping and
   SCRATCH is left untouched; a range that crosses a block boundary is
   gathered into SCRATCH and scratch.data () is returned.  OFF + LEN must
   not exceed the stream's size, and the stream's block numbers must have
   been validated against pdb->num_blocks.  */

static const gdb_byte *
pdb_msf_bytes (const pdb_per_objfile *pdb, const msf_stream &stream,
	       uint32_t off, uint32_t len, gdb::byte_vector &scratch)
{
  const pdb_file_mapping &map = pdb->file_mapping;
  gdb_assert (map.ok ());
  gdb_assert ((uint64_t) off + len <= stream.size);

  uint32_t bs = pdb->block_size;

  /* Return the mapping's base for an empty read.  An empty read at the
     end of a stream that fills its last block names no block.  */
  if (len == 0)
    return map.base;

  uint32_t block_shift = pdb->block_shift;
  uint32_t block_mask = bs - 1;
  uint32_t b = off >> block_shift;
  uint32_t in_block = off & block_mask;

  gdb_assert (b < stream.block_nums.size ());

  uint32_t phys_block = stream.block_nums[b];
  gdb_assert (phys_block < pdb->num_blocks);

  if ((uint64_t) in_block + len <= bs)
    return map.base + (uint64_t) phys_block * bs + in_block;

  /* The range spans a block boundary.  Gather it block by block: each
     step starts at the current offset inside its block (cur_in) and
     copies up to that block's end, so a mid-block start and a span over
     more than two blocks are both handled.  */
  scratch.resize (len);
  uint32_t copied = 0;
  while (copied < len)
    {
      uint32_t cur_b = (off + copied) >> block_shift;
      uint32_t cur_in = (off + copied) & block_mask;
      gdb_assert (cur_b < stream.block_nums.size ());
      uint32_t cur_phys = stream.block_nums[cur_b];
      gdb_assert (cur_phys < pdb->num_blocks);
      uint32_t chunk = std::min (len - copied, bs - cur_in);
      memcpy (scratch.data () + copied,
	      map.base + (uint64_t) cur_phys * bs + cur_in, chunk);
      copied += chunk;
    }
  return scratch.data ();
}

/* Read stream directory in order to get the blocks for each stream.  Throws
   via pdb_error on failure.
   See https://llvm.org/docs/PDB/MsfFile.html#the-stream-directory.  */

static void
pdb_read_stream_directory (pdb_per_objfile *pdb)
{
  /* The stream directory specifies the number of streams, the size of each
     stream and the blocks that belong to each stream.  To access the directory
     we first need to assemble it as it might be split across multiple blocks.
     The block map in the MSF SuperBlock header is an array of uint32_t block
     numbers that belong to the directory.  We gather these blocks into one
     buffer, with pdb_msf_bytes, and walk it.  The number of directory bytes
     is in the MSF header.  Layout:
       struct StreamDirectory {
	 ulittle32_t NumStreams;
	 ulittle32_t StreamSizes[NumStreams];
	 ulittle32_t StreamBlocks[NumStreams][];
       };
     Directory blocks example:
       If the block map contains:
	 Entry 0 = block 12 (first block of directory)
	 Entry 1 = block 47 (second block of directory)
	 Entry 2 = block 89 (third block of directory)
	 Entry 3 = block 156 (fourth block of directory)
       If block_size = 4096 and entries_per_block = 1024:
	 Block 12 holds entries 0-1023
	 Block 47 holds entries 1024-2047
	 Block 89 holds entries 2048-3071
	 Block 156 holds entries 3072-4095   */

  uint32_t num_dir_blocks = pdb_blocks_for_size (pdb->num_dir_bytes,
						 pdb->block_size);

  uint32_t entries_per_block = pdb->block_size / sizeof (uint32_t);

  /* MSF 7.00: block_map_addr is the block number of a single stream
     directory block-map block.  That block holds up to entries_per_block
     uint32_t entries, each being a stream directory block number.  */
  if (num_dir_blocks > entries_per_block)
    pdb_error ("stream directory too large for MSF 7.00 single stream "
	       "directory block-map block (%u > %u)",
	       num_dir_blocks, entries_per_block);

  /* Bound the directory by the file's size, because its block numbers
     may repeat.  */
  if (num_dir_blocks > pdb->num_blocks)
    pdb_error ("stream directory of %u blocks exceeds the file's %u blocks",
	       num_dir_blocks, pdb->num_blocks);

  if (pdb->block_map_addr >= pdb->num_blocks)
    pdb_error ("Block map address out of range: %u (%u blocks)",
	       pdb->block_map_addr, pdb->num_blocks);

  /* Locate the single block-map block.  */
  uint64_t block_map_offset = (uint64_t) pdb->block_map_addr * pdb->block_size;
  const gdb_byte *block_map = pdb->file_mapping.base + block_map_offset;

  /* Collect (and validate) directory block numbers from the block map.  */
  msf_stream dir_stream;
  dir_stream.size = pdb->num_dir_bytes;
  dir_stream.block_nums.resize (num_dir_blocks);
  for (uint32_t i = 0; i < num_dir_blocks; i++)
    dir_stream.block_nums[i]
      = pdb_read_block_number (pdb, block_map + i * sizeof (uint32_t),
			       "stream directory");

  /* Gather the directory blocks into a contiguous StreamDirectory; a
     range that crosses blocks is gathered into DIR_SCRATCH.  */
  gdb::byte_vector dir_scratch;
  const gdb_byte *dir = pdb_msf_bytes (pdb, dir_stream, 0, dir_stream.size,
				       dir_scratch);

  /* Parse the stream directory to fill the streams vector in
     pdb_per_objfile.  A cursor walks through NumStreams, then
     StreamSizes[NumStreams], then StreamBlocks[NumStreams][].  */
  const gdb_byte *cursor = dir;
  const gdb_byte *dir_end = dir + dir_stream.size;

  /* Every directory field is a uint32_t; stop at the directory's end.  */
  auto next_u32 = [&] (const char *what)
    {
      if (dir_end - cursor < (ptrdiff_t) sizeof (uint32_t))
	pdb_error ("stream directory truncated in %s", what);
      const gdb_byte *field = cursor;
      cursor += sizeof (uint32_t);
      return field;
    };

  uint32_t num_streams = read_u32 (next_u32 ("stream count"));
  if (num_streams > (uint64_t) (dir_end - cursor) / sizeof (uint32_t))
    pdb_error ("stream directory too small for %u streams", num_streams);

  pdb->streams.resize (num_streams);
  for (uint32_t i = 0; i < num_streams; i++)
    pdb->streams[i].size = read_u32 (next_u32 ("stream sizes"));

  for (uint32_t i = 0; i < num_streams; i++)
    {
      uint32_t stream_size = pdb->streams[i].size;
      if (stream_size == PDB_NO_STREAM)
	{
	  pdb->streams[i].size = 0;
	  continue;
	}

      uint32_t num_blocks = pdb_blocks_for_size (stream_size, pdb->block_size);
      if (num_blocks > (uint64_t) (dir_end - cursor) / sizeof (uint32_t))
	pdb_error ("stream directory too small for stream %u", i);

      /* Bound a stream by the file's size, because block lists are not
	 checked for repeated blocks.  */
      if (num_blocks > pdb->num_blocks)
	pdb_error ("stream %u size %u exceeds the file's %u blocks", i,
		   stream_size, pdb->num_blocks);
      pdb->streams[i].block_nums.reserve (num_blocks);
      for (uint32_t j = 0; j < num_blocks; j++)
	pdb->streams[i].block_nums.push_back
	  (pdb_read_block_number (pdb, next_u32 ("stream block list"),
				  "stream block list"));
    }
}

/* See pdb-internal.h.  */

gdb::byte_vector
pdb_per_objfile::read_stream (uint32_t stream_idx) const
{
  if (stream_idx >= streams.size ())
    return {};

  const msf_stream &stream = streams[stream_idx];
  if (stream.size == 0)
    return {};

  gdb::byte_vector buf;
  const gdb_byte *data = pdb_msf_bytes (this, stream, 0, stream.size, buf);
  if (data != buf.data ())
    buf.assign (data, data + stream.size);
  return buf;
}

/* Unmap the file, if it is mapped.  */

pdb_file_mapping::~pdb_file_mapping ()
{
  if (base == nullptr)
    return;
#ifdef _WIN32
  UnmapViewOfFile ((void *) base);
#else
  munmap ((void *) base, size);
#endif
}

/* Open the PDB file at PDB->pdb_file_path and map all of it read-only
   into PDB->file_mapping, with the size its handle reports.  Only a
   regular file is accepted: a FIFO or a device could block or have no
   size.  The handles are closed once the file is mapped, as the mapping
   keeps the file alive.  Returns false when the file cannot be opened or
   is not a regular file; throws when it is too small to be a PDB or
   cannot be mapped.  */

static bool
pdb_map_file (pdb_per_objfile *pdb)
{
  const char *path = pdb->pdb_file_path.c_str ();

#ifdef _WIN32
  /* Let a linker rewrite, rename or delete the PDB while GDB has it
     mapped.  */
  HANDLE fh = CreateFileA (path, GENERIC_READ,
			   FILE_SHARE_READ | FILE_SHARE_WRITE
			   | FILE_SHARE_DELETE,
			   nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
			   nullptr);
  if (fh == INVALID_HANDLE_VALUE)
    {
      pdb_load_diagnostic (pdb->objfile, "cannot open PDB '%s': error %lu",
			   path, (unsigned long) GetLastError ());
      return false;
    }
  SCOPE_EXIT { CloseHandle (fh); };

  LARGE_INTEGER file_size;
  if (GetFileType (fh) != FILE_TYPE_DISK || !GetFileSizeEx (fh, &file_size))
    {
      pdb_load_diagnostic (pdb->objfile,
			   "PDB '%s' is not a regular file", path);
      return false;
    }
  uint64_t size = file_size.QuadPart;
  if (size < PDB_MSF_MAGIC_SIZE)
    pdb_error ("Bad MSF signature");
  if (size > SIZE_MAX)
    pdb_error ("PDB file too large to map");

  /* Give the size explicitly, so that mapping fails if the file shrank
     since it was measured.  The view keeps the mapping object alive.  */
  HANDLE mh = CreateFileMappingA (fh, nullptr, PAGE_READONLY,
				  (DWORD) (size >> 32), (DWORD) size, nullptr);
  if (mh == nullptr)
    pdb_error ("Failed to map PDB file");
  void *base = MapViewOfFile (mh, FILE_MAP_READ, 0, 0, size);
  CloseHandle (mh);
  if (base == nullptr)
    pdb_error ("Failed to map PDB file");
#else
  scoped_fd fd = gdb_open_cloexec (path, O_RDONLY | O_NONBLOCK, 0);
  if (fd.get () < 0)
    {
      pdb_load_diagnostic (pdb->objfile, "cannot open PDB '%s': %s", path,
			   safe_strerror (errno));
      return false;
    }

  struct stat st;
  if (fstat (fd.get (), &st) != 0 || !S_ISREG (st.st_mode))
    {
      pdb_load_diagnostic (pdb->objfile,
			   "PDB '%s' is not a regular file", path);
      return false;
    }
  uint64_t size = st.st_size;
  if (size < PDB_MSF_MAGIC_SIZE)
    pdb_error ("Bad MSF signature");
  if (size > SIZE_MAX)
    pdb_error ("PDB file too large to map");

  /* On a POSIX host another process can truncate the file while it is
     mapped, and a later read past the new end raises SIGBUS.  Windows
     refuses to truncate a mapped file.  */
  void *base = mmap (nullptr, size, PROT_READ, MAP_PRIVATE, fd.get (), 0);
  if (base == MAP_FAILED)
    pdb_error ("Failed to map PDB file");
#endif

  pdb->file_mapping.base = (const gdb_byte *) base;
  pdb->file_mapping.size = size;
  return true;
}

/* See pdb-internal.h.  */

const gdb_byte *
pdb_stream_bytes (const pdb_per_objfile *pdb, uint32_t stream_idx,
		  uint32_t off, uint32_t len, gdb::byte_vector &scratch)
{
  gdb_assert (stream_idx < pdb->streams.size ());

  const msf_stream &stream = pdb->streams[stream_idx];
  if ((uint64_t) off + len > stream.size)
    pdb_error ("stream %u: read of %u bytes at offset %u exceeds its %u"
	       " bytes", stream_idx, len, off, stream.size);

  return pdb_msf_bytes (pdb, stream, off, len, scratch);
}

/* See pdb-internal.h.  */

std::string
pdb_format_guid (const gdb_byte *g)
{
  return string_printf ("%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X",
			read_u32 (g), read_u16 (g + 4), read_u16 (g + 6), g[8],
			g[9], g[10], g[11], g[12], g[13], g[14], g[15]);
}

/* A pdb_info_stream holds the fields of the PDB Info stream.  */
struct pdb_info_stream
{
  uint32_t version;
  uint32_t signature;
  uint32_t age;
  /* 16 bytes.  */
  const gdb_byte *guid;
  /* Named stream map start.  */
  const gdb_byte *map;
  /* Named stream map size.  */
  uint32_t map_size;
};

/* Decode the fixed header (version, signature, age, GUID) of the PDB Info
   stream bytes DATA.  map/map_size are set to the bytes that follow the
   fixed header.  The returned struct holds pointers into DATA, not
   copies.  */

static pdb_info_stream
pdb_decode_info_stream (const gdb::byte_vector &data)
{
  size_t size = data.size ();
  if (size < INFO_STREAM_MIN_SIZE)
    pdb_error ("PDB info stream too small (%zu bytes)", size);

  pdb_info_stream info;
  info.version = read_u32 (&data[0]);
  info.signature = read_u32 (&data[4]);
  info.age = read_u32 (&data[8]);
  info.guid = &data[12];
  info.map = data.data () + INFO_STREAM_HASH_MAP_OFFS;
  info.map_size = size - INFO_STREAM_HASH_MAP_OFFS;
  return info;
}

/* Read the PDB Info stream (stream 1), caching its bytes on
   pdb->info_data, and decode it.  */

static pdb_info_stream
pdb_read_info_stream (pdb_per_objfile *pdb)
{
  pdb->info_data = pdb->read_stream (PDB_STREAM_PDB);
  if (pdb->info_data.empty ())
    pdb_error ("Failed to read PDB info stream (stream 1)");

  return pdb_decode_info_stream (pdb->info_data);
}

/* Validate that the PDB matches the EXE's CodeView RSDS record, as
   microsoft-pdb's PDB::OpenValidate does: the PDB info stream has the
   same GUID and an age at least the EXE's, as writing to a PDB (for
   instance to source-index it) raises its age; and the DBI stream, once
   read (DBI_AGE nonzero), has the EXE's age.  Returns true on match, false
   on mismatch.  On false, the caller MUST drop this PDB file: symbol
   offsets, line tables and type indices are all keyed to the exact build
   that produced the EXE, so a mismatched PDB cannot be safely used.
   Mirrors the build-id check GDB performs for separate debug files (see
   build_id_verify).

   TODO: Implement logic that will skip wrong candidates and keep searching
   other locations for the PDB with matching GUID/age.  */

static bool
pdb_validate_guid (pdb_per_objfile *pdb, const pdb_rsds_info &rsds,
		   const pdb_info_stream &info, uint32_t dbi_age)
{
  if (memcmp (info.guid, rsds.guid.data (), 16) == 0
      && info.age >= rsds.age
      && (dbi_age == 0 || dbi_age == rsds.age))
    return true;

  /* The DBI age is not known until the DBI stream is read.  */
  std::string dbi_age_text;
  if (dbi_age != 0)
    dbi_age_text = string_printf (" (DBI age %u)", dbi_age);

  pdb_load_diagnostic (pdb->objfile,
		       "PDB '%s' has a different GUID/age, file skipped:\n"
		       "  EXE expects GUID=%s age=%u\n"
		       "  PDB provides GUID=%s age=%u%s",
		       pdb->pdb_file_path.c_str (),
		       pdb_format_guid (rsds.guid.data ()).c_str (), rsds.age,
		       pdb_format_guid (info.guid).c_str (), info.age,
		       dbi_age_text.c_str ());
  return false;
}

/* Find the end of a null-terminated string at P within [P, END).  Returns a
   pointer past the null terminator.  Calls pdb_error with MSG if no null
   byte is found within bounds.  */

static const gdb_byte *
pdb_skip_cstring (const gdb_byte *p, const gdb_byte *end, const char *msg)
{
  auto nul = (const gdb_byte *) memchr (p, 0, end - p);
  if (nul == nullptr)
    pdb_error ("Truncated %s in module info", msg);

  return nul + 1;
}

/* Read DBI (Debug Info) stream.  Reads headers and Module Info Records in order
   to obtain the sizes and stream numbers of the debug information for each
   module (object file).  Throws via pdb_error on failure.  */

static void
pdb_read_dbi_stream (pdb_per_objfile *pdb)
{
  /* Read DBI stream into the owned buffer; module names point into it.  */
  pdb->dbi_data = pdb->read_stream (PDB_STREAM_DBI);
  if (pdb->dbi_data.empty ())
    pdb_error ("Failed to read DBI stream");
  if (pdb->dbi_data.size () < DBI_HDR_MODULE_INFO_OFFS)
    pdb_error ("DBI stream too small for its header (%zu bytes)",
	       pdb->dbi_data.size ());

  /* Parse DBI header.  */

  const gdb_byte *dbi_hdr = pdb->dbi_data.data ();
  uint32_t signature = read_u32 (dbi_hdr + DBI_HDR_SIGNATURE_OFFS);

  if (signature != DBI_HDR_SIGNATURE_VAL)
    pdb_error ("Invalid DBI signature: 0x%x", signature);

  /* Read sizes from DBI header.  */

  uint32_t mod_info_size = read_u32 (dbi_hdr + DBI_HDR_MODULE_SIZE_OFFS);
  if (mod_info_size > pdb->dbi_data.size () - DBI_HDR_MODULE_INFO_OFFS)
    pdb_error ("DBI module info (%u bytes) overflows the stream",
	       mod_info_size);
  pdb->dbi_age = read_u32 (dbi_hdr + DBI_HDR_AGE_OFFS);

  /* Module info starts right after the DBI header.  */
  const gdb_byte *mod_info = dbi_hdr + DBI_HDR_MODULE_INFO_OFFS;
  const gdb_byte *mod_info_end = mod_info + mod_info_size;

  if (mod_info_size == 0)
    {
      /* A PDB without modules is valid; it has no debug information.  */
      pdb_dbg_printf ("%s: module info is empty",
		      pdb->pdb_file_path.c_str ());
      return;
    }

  /* Each Module Info record is its fixed part, then the module and object
     file names, padded to 4 bytes.  A tail too short for a fixed part and
     a name is not a record.  */
  const gdb_byte *p = mod_info;
  while (mod_info_end - p > MODI_FIXED_SIZE)
    {
      pdb_module_info &module = pdb->modules.emplace_back ();

      /* MODI fixed header fields (0x40 bytes per LLVM/MS spec).  */
      module.stream_number = read_u16 (p + MODI_STREAM_NUM_OFFS);
      module.sym_byte_size = read_u32 (p + MODI_SYM_BYTES_OFFS);
      module.c11_byte_size = read_u32 (p + MODI_C11_BYTES_OFFS);
      module.c13_byte_size = read_u32 (p + MODI_C13_BYTES_OFFS);

      /* Section contribution.  */
      module.sc_section = read_u16 (p + MODI_SC_ISECT_OFFS);
      module.sc_offset = read_u32 (p + MODI_SC_OFFSET_OFFS);
      module.sc_size = read_u32 (p + MODI_SC_SIZE_OFFS);

      /* Module name starts at offset 0x40.  */
      p += MODI_FIXED_SIZE;

      module.module_name = (const char *) p;
      p = pdb_skip_cstring (p, mod_info_end, "module name");

      /* Skip the object file name.  */
      p = pdb_skip_cstring (p, mod_info_end, "obj file name");

      /* Round P up to the next record, but not past the end of the
	 module info.  Records are padded to 4 bytes.  The buffer is
	 4-aligned, so rounding the address rounds the offset.  */
      p = std::min (align_up (p, 4), mod_info_end);
    }
}

/* Load the PDB file for the given pdb context: find it and map it
   read-only into pdb->file_mapping, with its path in pdb->pdb_file_path.
   The mapping remains for the lifetime of the pdb_per_objfile.  Returns
   false when the file is not found or cannot be opened.  */

static bool
pdb_load_pdb_file (pdb_per_objfile *pdb, const pdb_rsds_info &rsds)
{
  /* Find the PDB file on disk.  */
  std::string pdb_path = pdb_find_pdb_file (pdb->objfile, rsds);
  if (pdb_path.empty ())
    return false;

  pdb->pdb_file_path = std::move (pdb_path);
  if (!pdb_map_file (pdb))
    return false;

  pdb_dbg_printf ("PDB file mapped: %s (%" PRIu64 " bytes)",
		  pdb->pdb_file_path.c_str (), pdb->file_mapping.size);
  return true;
}

/* Read the PDB that RSDS names for OBJFILE; pdb_initialize_objfile is the
   reader's entry point.  PDB parsing starts by reading the MSF SuperBlock,
   which contains basic info about the file and location of the stream
   directory.  Each stream spans multiple blocks that are not necessarily
   contiguous and the layout of every single stream is described in the
   stream directory.
   Then we continue with the DBI stream, which describes the modules.  From
   it we get the list of modules, with the stream number and the sizes of
   each module's debug information.  */

static pdb_per_objfile *
pdb_read_pdb_file (objfile *objfile, const pdb_rsds_info &rsds)
{
  gdb_assert (objfile->obfd != nullptr);

  /* Initialize PDB reader: create and register per_objfile.  */
  pdb_per_objfile *pdb = get_or_create_pdb_per_objfile (objfile);

  /* Find, open and map the PDB.  */
  if (!pdb_load_pdb_file (pdb, rsds))
    /* pdb_find_pdb_file reports a PDB that is not found, and pdb_map_file
       one that cannot be opened.  */
    return nullptr;

  /* Read MSF structure.  */
  pdb_read_msf_header (pdb);
  pdb_read_stream_directory (pdb);

  /* Read & decode the PDB Info stream, which holds the GUID and age.  */
  pdb_info_stream info = pdb_read_info_stream (pdb);

  /* Validate PDB GUID/age against the EXE's RSDS record.  A mismatch means
     this PDB belongs to a different build, so loading is aborted.  The
     DBI age is checked once the DBI stream is read.  */
  if (!pdb_validate_guid (pdb, rsds, info, 0))
    return nullptr;

  /* Read DBI stream (contains modules and line info).  */
  pdb_read_dbi_stream (pdb);
  if (!pdb_validate_guid (pdb, rsds, info, pdb->dbi_age))
    return nullptr;

  if (info_verbose)
    gdb_printf ("\nUsing PDB file: %s\n", pdb->pdb_file_path.c_str ());
  return pdb;
}
} /* namespace pdb */

/* See pdb.h.  */

bool
pdb_initialize_objfile (objfile *objfile)
{
  using namespace pdb;

  /* Called for any COFF file; pdb_read_rsds_info finds no record in one
     that is not PE.  */
  gdb_assert (bfd_get_flavour (objfile->obfd.get ())
	      == bfd_target_coff_flavour);

  /* Check for RSDS (CodeView/PDB7) record.  */
  auto rsds = pdb_read_rsds_info (objfile->obfd.get ());
  if (!rsds)
    return false;

  pdb_per_objfile *pdb = nullptr;
  try
    {
      pdb = pdb_read_pdb_file (objfile, *rsds);
    }
  catch (const gdb_exception_error &ex)
    {
      /* A malformed PDB is dropped like a mismatched one.  The error is
	 about the file's contents, so name the file.  */
      const pdb_per_objfile *failed = get_pdb_per_objfile (objfile);
      std::string message = ex.what ();
      if (startswith (message, "PDB Error: "))
	message.erase (0, strlen ("PDB Error: "));
      if (failed != nullptr && !failed->pdb_file_path.empty ())
	message = failed->pdb_file_path + ": " + message;
      if ((objfile->flags & OBJF_MAINLINE) != 0)
	warning ("%s", message.c_str ());
      else
	complaint ("%s", message.c_str ());
    }
  catch (const gdb_exception &)
    {
      /* Drop the half-built reader state when interrupted.  */
      pdb_objfile_data_key.clear (objfile);
      throw;
    }

  if (pdb == nullptr)
    {
      /* Drop the half-built reader state.  pdb_read_pdb_file registers the
	 pdb_per_objfile and records the candidate path before it can tell
	 whether the file is usable, so a rejected PDB would otherwise stay
	 attached to the objfile and be reported as loaded by
	 'maint info pdb-loaded-files', naming the very file that was
	 refused.  */
      pdb_objfile_data_key.clear (objfile);
      return false;
    }

  return true;
}

#if GDB_SELF_TEST

namespace pdb::selftests
{

/* See pdb-internal.h.  */

void
put_u32 (gdb::byte_vector &bytes, size_t offset, uint32_t v)
{
  if (bytes.size () < offset + 4)
    bytes.resize (offset + 4);
  bytes[offset] = v & 0xff;
  bytes[offset + 1] = (v >> 8) & 0xff;
  bytes[offset + 2] = (v >> 16) & 0xff;
  bytes[offset + 3] = (v >> 24) & 0xff;
}

/* See pdb-internal.h.  */

std::string
error_of (gdb::function_view<void ()> body)
{
  try
    {
      body ();
    }
  catch (const gdb_exception_error &ex)
    {
      std::string message = ex.what ();
      if (startswith (message, "PDB Error: "))
	message.erase (0, strlen ("PDB Error: "));
      return message;
    }
  return "";
}

/* See pdb-internal.h.  */

void
test_pdb::build ()
{
  uint32_t next_block = 3;
  std::vector<std::vector<uint32_t>> stream_blocks;
  for (const gdb::byte_vector &s : streams)
    {
      std::vector<uint32_t> blocks;
      for (size_t off = 0; off < s.size (); off += block_size)
	blocks.push_back (next_block++);
      stream_blocks.push_back (std::move (blocks));
    }

  /* Reserve two blocks for the directory, so that set_directory can grow
     it.  */
  directory_block = next_block;
  next_block += 2;
  uint32_t block_map = next_block++;

  bytes.assign ((size_t) next_block * block_size, 0);
  memcpy (bytes.data (), PDB_MSF_MAGIC.data (), PDB_MSF_MAGIC_SIZE);
  put_u32 (bytes, MSF_BLOCK_SIZE_OFFS, block_size);
  put_u32 (bytes, MSF_BLOCK_SIZE_OFFS + 4, 1);
  put_u32 (bytes, MSF_NUM_BLOCKS_OFFS, next_block);
  put_u32 (bytes, MSF_BLOCK_MAP_ADDR_OFFS, block_map);
  put_u32 (bytes, (size_t) block_map * block_size, directory_block);
  put_u32 (bytes, (size_t) block_map * block_size + 4, directory_block + 1);

  for (size_t i = 0; i < streams.size (); i++)
    for (size_t b = 0; b < stream_blocks[i].size (); b++)
      {
	size_t off = b * block_size;
	size_t len = std::min<size_t> (block_size, streams[i].size () - off);
	memcpy (bytes.data () + (size_t) stream_blocks[i][b] * block_size,
		streams[i].data () + off, len);
      }

  std::vector<uint32_t> words { (uint32_t) streams.size () };
  for (const gdb::byte_vector &s : streams)
    words.push_back (s.size ());
  for (const std::vector<uint32_t> &blocks : stream_blocks)
    words.insert (words.end (), blocks.begin (), blocks.end ());
  set_directory (words);
}

/* See pdb-internal.h.  */

void
test_pdb::set_directory (const std::vector<uint32_t> &words)
{
  gdb_assert (words.size () * 4 <= 2 * block_size);
  for (size_t i = 0; i < words.size (); i++)
    put_u32 (bytes, (size_t) directory_block * block_size + i * 4, words[i]);
  put_u32 (bytes, MSF_NUM_DIRECTORY_BYTES_OFFS, words.size () * 4);
}

/* See pdb-internal.h.  */

test_pdb_reader::test_pdb_reader (const test_pdb &file)
  : pdb (nullptr)
{
  pdb.file_mapping.base = file.bytes.data ();
  pdb.file_mapping.size = file.bytes.size ();
}

test_pdb_reader::~test_pdb_reader ()
{
  /* The bytes belong to the test_pdb: nothing to unmap.  */
  pdb.file_mapping.base = nullptr;
}

/* See pdb-internal.h.  */

void
test_pdb_reader::read_msf ()
{
  pdb_read_msf_header (&pdb);
  pdb_read_stream_directory (&pdb);
}

/* Return a test_pdb of BLOCK_SIZE blocks with three streams, of 1, 2 and
   3 blocks.  Each stream is 100 bytes short of filling its last block.  */

static test_pdb
three_streams (uint32_t block_size = 4096)
{
  test_pdb file;
  file.block_size = block_size;
  file.streams.resize (3);
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < (i + 1) * block_size - 100; j++)
      file.streams[i].push_back ((gdb_byte) (i * 7 + j));
  file.build ();
  return file;
}

/* Check the SuperBlock's magic and block size, and the number of blocks
   the file holds.  */

static void
test_msf_superblock ()
{
  test_pdb file = three_streams ();
  {
    test_pdb_reader r (file);
    SELF_CHECK (error_of ([&] { r.read_msf (); }).empty ());
    SELF_CHECK (r.pdb.block_size == 4096);
    SELF_CHECK (r.pdb.streams.size () == 3);
  }

  /* Another magic.  */
  test_pdb bad = file;
  bad.bytes[0] = 'X';
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); }).find ("Bad MSF signature")
		!= std::string::npos);
  }

  /* A file too small for the magic.  */
  bad = file;
  bad.bytes.resize (PDB_MSF_MAGIC_SIZE - 1);
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); }).find ("Bad MSF signature")
		!= std::string::npos);
  }

  /* A file of 48 bytes holds the magic but not the SuperBlock.  */
  bad = file;
  bad.bytes.resize (48);
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); })
		== "Failed to read MSF SuperBlock");
  }

  for (uint32_t block_size : { 0u, 100u, 256u, 3000u, 65536u, 0x80000000u })
    {
      bad = file;
      put_u32 (bad.bytes, MSF_BLOCK_SIZE_OFFS, block_size);
      test_pdb_reader r (bad);
      SELF_CHECK (error_of ([&] { r.read_msf (); })
		  .find ("Invalid PDB block size") != std::string::npos);
    }

  /* Check more blocks than the file holds, also when the size wraps 32
     bits.  */
  for (uint32_t blocks : { 1000u, 0x100001u, 0xffffffffu })
    {
      bad = file;
      put_u32 (bad.bytes, MSF_NUM_BLOCKS_OFFS, blocks);
      test_pdb_reader r (bad);
      SELF_CHECK (error_of ([&] { r.read_msf (); }).find ("PDB file too small")
		  != std::string::npos);
    }

  /* Other block sizes are read, up to those /PDBPAGESIZE gives.  */
  for (uint32_t block_size : { 512u, 1024u, 2048u, 8192u, 16384u, 32768u })
    {
      test_pdb other = three_streams (block_size);
      test_pdb_reader r (other);
      SELF_CHECK (error_of ([&] { r.read_msf (); }).empty ());
      SELF_CHECK (r.pdb.streams[2].size == 3 * block_size - 100);
    }
}

/* Check where the stream directory is, the stream count, and a stream's
   blocks.  */

static void
test_msf_directory ()
{
  test_pdb file = three_streams ();
  uint32_t num_blocks = file.bytes.size () / file.block_size;

  test_pdb bad = file;
  put_u32 (bad.bytes, MSF_BLOCK_MAP_ADDR_OFFS, num_blocks);
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); })
		.find ("Block map address out of range") != std::string::npos);
  }

  /* More streams than the directory has sizes for.  */
  bad = file;
  bad.set_directory ({ 0x40000000, 1, 2 });
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); })
		.find ("too small for 1073741824 streams")
		!= std::string::npos);
  }

  /* A directory may list one block many times, so it can give a stream
     more blocks than the file has.  */
  bad = file;
  std::vector<uint32_t> words { 1, (num_blocks + 1) * file.block_size };
  for (uint32_t i = 0; i < num_blocks + 1; i++)
    words.push_back (3);
  bad.set_directory (words);
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); })
		.find (string_printf ("exceeds the file's %u blocks",
				      num_blocks)) != std::string::npos);
  }

  /* A block number past the file.  */
  bad = file;
  bad.set_directory ({ 1, 10, num_blocks });
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); }).find ("out of range")
		!= std::string::npos);
  }

  /* A directory that stops inside the block list.  */
  bad = file;
  bad.set_directory ({ 1, 3 * file.block_size });
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); })
		.find ("directory too small for stream 0")
		!= std::string::npos);
  }

  /* Check a directory of more blocks than one block of the block map
     lists.  */
  bad = file;
  put_u32 (bad.bytes, MSF_NUM_DIRECTORY_BYTES_OFFS,
	   (file.block_size / 4 + 1) * file.block_size);
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); })
		.find ("stream directory too large for MSF 7.00")
		!= std::string::npos);
  }

  /* Check a directory of more blocks than the file has.  The block map
     can list them by repeating one block.  */
  bad = file;
  put_u32 (bad.bytes, MSF_NUM_DIRECTORY_BYTES_OFFS,
	   (num_blocks + 1) * file.block_size);
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); })
		.find (string_printf ("exceeds the file's %u blocks",
				      num_blocks)) != std::string::npos);
  }

  /* A directory too short for its stream count.  */
  bad = file;
  put_u32 (bad.bytes, MSF_NUM_DIRECTORY_BYTES_OFFS, 2);
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); })
		.find ("directory truncated in stream count")
		!= std::string::npos);
  }

  /* A stream of size PDB_NO_STREAM is empty and has no blocks listed: the
     blocks of the stream after it are read from their place.  */
  bad = file;
  bad.set_directory ({ 3, 100, PDB_NO_STREAM, 100, 3, 4 });
  {
    test_pdb_reader r (bad);
    SELF_CHECK (error_of ([&] { r.read_msf (); }).empty ());
    SELF_CHECK (r.pdb.streams.size () == 3);
    SELF_CHECK (r.pdb.streams[1].size == 0
		&& r.pdb.streams[1].block_nums.empty ());
    SELF_CHECK (r.pdb.streams[2].size == 100
		&& r.pdb.streams[2].block_nums
		   == std::vector<uint32_t> ({ 4 }));
    SELF_CHECK (r.pdb.read_stream (1).empty ());
    SELF_CHECK (r.pdb.read_stream (2).size () == 100);
  }
}

/* Check reading stream bytes.  A range inside one block is the mapping
   itself, and a range across blocks is gathered.  */

static void
test_msf_stream_bytes ()
{
  test_pdb file = three_streams ();
  test_pdb_reader r (file);
  r.read_msf ();

  const gdb::byte_vector &s = file.streams[2];
  gdb::byte_vector scratch;
  for (uint32_t off : { 0u, 100u, 4090u, 4096u, 8000u })
    for (uint32_t len : { 1u, 6u, 200u, 5000u })
      {
	if (off + len > s.size ())
	  continue;
	const gdb_byte *p = pdb_stream_bytes (&r.pdb, 2, off, len, scratch);
	SELF_CHECK (memcmp (p, s.data () + off, len) == 0);
	bool one_block = off / 4096 == (off + len - 1) / 4096;
	SELF_CHECK ((p == scratch.data ()) != one_block);
      }

  /* Check that an empty read at the end of a stream that fills its last
     block names no block.  */
  test_pdb full;
  full.streams.push_back (gdb::byte_vector (4096, 1));
  full.build ();
  test_pdb_reader rf (full);
  rf.read_msf ();
  SELF_CHECK (error_of ([&] { pdb_stream_bytes (&rf.pdb, 0, 4096, 0,
						scratch); }).empty ());

  /* Past the end.  */
  SELF_CHECK (error_of ([&] { pdb_stream_bytes (&r.pdb, 2, s.size () - 3, 4,
						scratch); })
	      .find ("exceeds its") != std::string::npos);
  SELF_CHECK (error_of ([&] { pdb_stream_bytes (&r.pdb, 2, 0xffffffff, 2,
						scratch); })
	      .find ("exceeds its") != std::string::npos);

  /* The whole stream.  */
  SELF_CHECK (r.pdb.read_stream (1) == file.streams[1]);
}

/* Check that a GUID prints as the Windows tools print it.  Its first three
   fields print little-endian, and the rest prints byte by byte.  */

static void
test_format_guid ()
{
  gdb_byte guid[16];
  for (int i = 0; i < 16; i++)
    guid[i] = i == 0 ? 0xab : i;
  SELF_CHECK (pdb_format_guid (guid)
	      == "030201AB-0504-0706-0809-0A0B0C0D0E0F");
}

/* Return the fixed part of a PDB Info stream of GUID and AGE, followed by
   an empty named stream map.  */

static gdb::byte_vector
info_header (const std::array<gdb_byte, 16> &guid, uint32_t age)
{
  gdb::byte_vector info;
  put_u32 (info, 0, 20000404);
  put_u32 (info, 4, 0x12345678);
  put_u32 (info, 8, age);
  info.insert (info.end (), guid.begin (), guid.end ());
  for (int i = 0; i < 5; i++)
    put_u32 (info, info.size (), 0);
  return info;
}

/* Check the fields of the PDB Info stream, and a stream that is missing
   or too small for them.  */

static void
test_info_stream ()
{
  std::array<gdb_byte, 16> guid;
  for (int i = 0; i < 16; i++)
    guid[i] = 0x10 + i;

  test_pdb file;
  file.streams.resize (2);
  file.streams[1] = info_header (guid, 3);
  file.build ();
  {
    test_pdb_reader r (file);
    r.read_msf ();
    pdb_info_stream info = pdb_read_info_stream (&r.pdb);
    SELF_CHECK (info.version == 20000404);
    SELF_CHECK (info.signature == 0x12345678);
    SELF_CHECK (info.age == 3);
    SELF_CHECK (memcmp (info.guid, guid.data (), 16) == 0);
    SELF_CHECK (info.map_size == 20);
  }

  /* An Info stream that ends right after its fixed header is valid; its
     named stream map is empty.  */
  file.streams[1].resize (INFO_STREAM_MIN_SIZE);
  file.build ();
  {
    test_pdb_reader r (file);
    r.read_msf ();
    pdb_info_stream info = pdb_read_info_stream (&r.pdb);
    SELF_CHECK (info.age == 3);
    SELF_CHECK (info.map_size == 0);
  }

  file.streams[1].resize (INFO_STREAM_MIN_SIZE - 1);
  file.build ();
  {
    test_pdb_reader r (file);
    r.read_msf ();
    SELF_CHECK (error_of ([&] { pdb_read_info_stream (&r.pdb); })
		== "PDB info stream too small (27 bytes)");
  }

  file.streams[1].clear ();
  file.build ();
  {
    test_pdb_reader r (file);
    r.read_msf ();
    SELF_CHECK (error_of ([&] { pdb_read_info_stream (&r.pdb); })
		== "Failed to read PDB info stream (stream 1)");
  }
}

/* Check that a PDB belongs to the executable whose RSDS record has its
   GUID and an age no higher than the Info stream's.  Writing to a PDB
   after the link raises that age.  The DBI stream keeps the link's age,
   which must be the executable's when it is known (not 0).  */

static void
test_validate_guid ()
{
  pdb_per_objfile pdb (nullptr);
  pdb_rsds_info rsds;
  for (int i = 0; i < 16; i++)
    rsds.guid[i] = i;
  rsds.age = 5;

  std::array<gdb_byte, 16> guid = rsds.guid;
  pdb_info_stream info;
  info.guid = guid.data ();
  info.age = 5;

  SELF_CHECK (pdb_validate_guid (&pdb, rsds, info, 0));
  SELF_CHECK (pdb_validate_guid (&pdb, rsds, info, 5));
  SELF_CHECK (!pdb_validate_guid (&pdb, rsds, info, 4));
  SELF_CHECK (!pdb_validate_guid (&pdb, rsds, info, 6));

  info.age = 6;
  SELF_CHECK (pdb_validate_guid (&pdb, rsds, info, 5));
  info.age = 4;
  SELF_CHECK (!pdb_validate_guid (&pdb, rsds, info, 0));

  info.age = 5;
  for (int i : { 0, 7, 15 })
    {
      guid = rsds.guid;
      guid[i] ^= 1;
      SELF_CHECK (!pdb_validate_guid (&pdb, rsds, info, 0));
    }
}

/* Return a DBI module record.  It holds the fixed part, then the module
   and object file names, padded to 4 bytes.  */

static gdb::byte_vector
module_record (uint16_t stream, const char *module, const char *obj)
{
  gdb::byte_vector rec (MODI_FIXED_SIZE, 0);
  rec[MODI_STREAM_NUM_OFFS] = stream & 0xff;
  rec[MODI_STREAM_NUM_OFFS + 1] = stream >> 8;
  put_u32 (rec, MODI_SYM_BYTES_OFFS, 100);
  put_u32 (rec, MODI_C11_BYTES_OFFS, 0);
  put_u32 (rec, MODI_C13_BYTES_OFFS, 200);
  rec[MODI_SC_ISECT_OFFS] = 1;
  put_u32 (rec, MODI_SC_OFFSET_OFFS, 0x20);
  put_u32 (rec, MODI_SC_SIZE_OFFS, 0x30);
  rec.insert (rec.end (), module, module + strlen (module) + 1);
  rec.insert (rec.end (), obj, obj + strlen (obj) + 1);
  rec.resize (align_up (rec.size (), 4));
  return rec;
}

/* Check the DBI header and module records, whose sizes are file data.  */

static void
test_dbi_header ()
{
  gdb::byte_vector mods = module_record (10, "a.obj", "a.obj");
  gdb::byte_vector b = module_record (0xffff, "* Linker *", "");
  mods.insert (mods.end (), b.begin (), b.end ());

  auto dbi_with = [] (const gdb::byte_vector &records)
    {
      gdb::byte_vector dbi (DBI_HDR_SIZE, 0);
      put_u32 (dbi, DBI_HDR_SIGNATURE_OFFS, DBI_HDR_SIGNATURE_VAL);
      put_u32 (dbi, DBI_HDR_AGE_OFFS, 9);
      put_u32 (dbi, DBI_HDR_MODULE_SIZE_OFFS, records.size ());
      dbi.insert (dbi.end (), records.begin (), records.end ());
      return dbi;
    };

  /* Read DBI as the DBI stream; return the error, if any.  Each module
     read has the fields module_record wrote; its name goes to NAMES and
     its stream number to STREAMS, when given.  */
  auto read = [] (const gdb::byte_vector &dbi,
		  std::vector<std::string> *names,
		  std::vector<uint16_t> *streams = nullptr)
    {
      test_pdb file;
      file.streams.resize (4);
      file.streams[PDB_STREAM_DBI] = dbi;
      file.build ();
      test_pdb_reader r (file);
      r.read_msf ();
      std::string err = error_of ([&] { pdb_read_dbi_stream (&r.pdb); });
      if (err.empty ())
	{
	  SELF_CHECK (r.pdb.dbi_age == 9);
	  for (const pdb_module_info &m : r.pdb.modules)
	    {
	      SELF_CHECK (m.sym_byte_size == 100);
	      SELF_CHECK (m.c11_byte_size == 0);
	      SELF_CHECK (m.c13_byte_size == 200);
	      SELF_CHECK (m.sc_section == 1);
	      SELF_CHECK (m.sc_offset == 0x20);
	      SELF_CHECK (m.sc_size == 0x30);
	      names->push_back (m.module_name);
	      if (streams != nullptr)
		streams->push_back (m.stream_number);
	    }
	}
      return err;
    };

  std::vector<std::string> names;
  std::vector<uint16_t> streams;
  SELF_CHECK (read (dbi_with (mods), &names, &streams).empty ());
  SELF_CHECK (names == std::vector<std::string> ({ "a.obj", "* Linker *" }));
  SELF_CHECK (streams == std::vector<uint16_t> ({ 10, 0xffff }));

  /* No modules at all.  */
  names.clear ();
  SELF_CHECK (read (dbi_with ({}), &names).empty () && names.empty ());

  /* A module record shorter than its fixed part is not read.  */
  names.clear ();
  gdb::byte_vector short_tail = mods;
  short_tail.resize (short_tail.size () + 32, 0);
  SELF_CHECK (read (dbi_with (short_tail), &names).empty ()
	      && names.size () == 2);

  /* The last record's padding is not in the module info: its names end
     right after their terminators, at an offset that is not a multiple of
     4.  */
  gdb::byte_vector unpadded = module_record (10, "abc", "");
  unpadded.resize (MODI_FIXED_SIZE + strlen ("abc") + 1 + 1);
  names.clear ();
  SELF_CHECK (read (dbi_with (unpadded), &names).empty ()
	      && names == std::vector<std::string> ({ "abc" }));

  /* Check names without their terminators before the module records
     end.  */
  gdb::byte_vector cut = mods;
  cut.resize (MODI_FIXED_SIZE + 3);
  SELF_CHECK (read (dbi_with (cut), &names)
	      == "Truncated module name in module info");
  cut = mods;
  cut.resize (MODI_FIXED_SIZE + 8);
  SELF_CHECK (read (dbi_with (cut), &names)
	      == "Truncated obj file name in module info");

  /* Module records past the stream.  */
  gdb::byte_vector dbi = dbi_with (mods);
  put_u32 (dbi, DBI_HDR_MODULE_SIZE_OFFS, mods.size () + 1);
  SELF_CHECK (read (dbi, &names).find ("overflows the stream")
	      != std::string::npos);
  put_u32 (dbi, DBI_HDR_MODULE_SIZE_OFFS, 0xffffffff);
  SELF_CHECK (read (dbi, &names).find ("overflows the stream")
	      != std::string::npos);

  dbi = dbi_with (mods);
  put_u32 (dbi, DBI_HDR_SIGNATURE_OFFS, 0);
  SELF_CHECK (read (dbi, &names) == "Invalid DBI signature: 0x0");

  dbi.resize (DBI_HDR_SIZE - 1);
  SELF_CHECK (read (dbi, &names)
	      == "DBI stream too small for its header (63 bytes)");
  SELF_CHECK (read ({}, &names) == "Failed to read DBI stream");
}

} /* namespace pdb::selftests */

#endif /* GDB_SELF_TEST */

INIT_GDB_FILE (pdb_read)
{
  using namespace pdb;

  add_setshow_zuinteger_cmd ("pdb-read", no_class, &pdb_read_debug, _("\
Set debugging of the PDB reader."), _("\
Show debugging of the PDB reader."), _("\
When enabled (non-zero), debugging messages are printed during PDB\n\
reading."),
			     nullptr, nullptr, &setdebuglist, &showdebuglist);

#if GDB_SELF_TEST
  ::selftests::register_test ("pdb-msf-superblock",
			      pdb::selftests::test_msf_superblock);
  ::selftests::register_test ("pdb-msf-directory",
			      pdb::selftests::test_msf_directory);
  ::selftests::register_test ("pdb-msf-stream-bytes",
			      pdb::selftests::test_msf_stream_bytes);
  ::selftests::register_test ("pdb-format-guid",
			      pdb::selftests::test_format_guid);
  ::selftests::register_test ("pdb-info-stream",
			      pdb::selftests::test_info_stream);
  ::selftests::register_test ("pdb-validate-guid",
			      pdb::selftests::test_validate_guid);
  ::selftests::register_test ("pdb-dbi-header",
			      pdb::selftests::test_dbi_header);
#endif
}
