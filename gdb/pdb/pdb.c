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
#include "buildsym.h"
#include "complaints.h"
#include "block.h"
#include "symfile.h"
#include "osabi.h"
#include "gdbarch.h"
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
#include <chrono>

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

/* Quick functions debug statements.  */
#define pdb_qf_printf(fmt, ...) \
  debug_prefixed_printf_cond (pdb_read_debug >= 2, "pdb qf", fmt, ##__VA_ARGS__)

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
   fixed header so the caller can hand them to pdb_parse_named_stream_map.
   The returned struct holds pointers into DATA, not copies.  */

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

/* One entry in the Named Stream Map, which maps a stream name to a stream
   index.  NAME points into the cached PDB info stream StringBuffer, so it stays
   valid for the lifetime of the pdb_per_objfile.  */
struct pdb_named_stream
{
  const char *name;
  uint32_t stream_idx;
};

/* The Named Stream Map decoded from the PDB info stream.  An empty `entries`
   vector means that the map was missing or corrupt.  Callers can treat that the
   same as "name not found" via lookup().  */
struct pdb_named_stream_map
{
  std::vector<pdb_named_stream> entries;

  /* Return the stream index registered for NAME, or PDB_NO_STREAM if
     NAME is not present in the map.  */
  uint32_t lookup (const char *name) const
  {
    for (const auto &e : entries)
      if (strcmp (e.name, name) == 0)
	return e.stream_idx;
    return PDB_NO_STREAM;
  }
};

/* One present bucket of a serialized PDB hash table.  It holds the raw uint32
   key and value, which the caller interprets.  */
struct pdb_hash_bucket
{
  uint32_t key;
  uint32_t value;
};

/* Decode the serialized hash table beginning at HT, bounded by END.  On
   corruption, emit a warning and return std::nullopt.  The on-disk layout
   (microsoft-pdb HashTable) is:
     uint32_t Size;
     uint32_t Capacity;
     uint32_t PresentWordCount;  uint32_t Present[PresentWordCount];
     uint32_t DeletedWordCount;  uint32_t Deleted[DeletedWordCount];
     { uint32_t Key; uint32_t Value; } Buckets[popcount (Present)];
   One bucket is stored per set bit in the Present bit-vector.  */

static std::optional<std::vector<pdb_hash_bucket>>
pdb_decode_hash_table (const gdb_byte *ht, const gdb_byte *end)
{
  if (end - ht < HASH_TABLE_PRESENT_WORD_OFFS)
    {
      pdb_warning ("PDB hash table too small for header");
      return {};
    }

  uint32_t size = read_u32 (ht + HASH_TABLE_SIZE_OFFS);
  uint32_t capacity = read_u32 (ht + HASH_TABLE_CAPACITY_OFFS);
  uint32_t present_words = read_u32 (ht + HASH_TABLE_PRESENT_CNT_OFFS);
  const gdb_byte *present_bits = ht + HASH_TABLE_PRESENT_WORD_OFFS;

  /* Compare the counts with the words left before forming any pointer from
     them, because the counts are file data.  The present bit vector and the
     DeletedWordCount field must fit.  */
  size_t words_left = (end - present_bits) / HASH_TABLE_BYTES_PER_WORD;
  if (present_words >= words_left)
    {
      pdb_warning ("PDB hash table too small for header "
		   "and present bit vector");
      return {};
    }

  const gdb_byte *deleted_cnt_ptr
    = present_bits + (size_t) present_words * HASH_TABLE_BYTES_PER_WORD;
  uint32_t deleted_words = read_u32 (deleted_cnt_ptr);
  if (deleted_words > words_left - present_words - 1)
    {
      pdb_warning ("PDB hash table too small for deleted bit vector");
      return {};
    }

  const gdb_byte *table
    = (deleted_cnt_ptr
       + ((size_t) deleted_words + 1) * HASH_TABLE_BYTES_PER_WORD);

  /* Only a set bit of the present vector has a bucket, so neither SIZE
     nor a CAPACITY past the vector's bits can add one.  */
  std::vector<pdb_hash_bucket> buckets;
  buckets.reserve (std::min<size_t> (size,
				     (end - table) / HASH_TABLE_ENTRY_SIZE));
  uint64_t num_bits
    = std::min<uint64_t> (capacity,
			  (uint64_t) present_words * HASH_TABLE_BIT_WORD_SIZE);

  for (uint64_t i = 0; i < num_bits; i++)
    {
      uint32_t word_idx = i / HASH_TABLE_BIT_WORD_SIZE;
      uint32_t bit_idx = i % HASH_TABLE_BIT_WORD_SIZE;
      if (!(read_u32 (present_bits + word_idx * HASH_TABLE_BYTES_PER_WORD)
	    & (1u << bit_idx)))
	continue;

      /* The bucket array holds only present entries and the on-disk table
	 may end right after the last one.  */
      if (end - table < HASH_TABLE_ENTRY_SIZE)
	{
	  pdb_warning ("PDB hash table truncated past entry %" PRIu64, i);
	  break;
	}

      uint32_t key = read_u32 (table + 0);
      uint32_t value = read_u32 (table + HASH_TABLE_KEY_SIZE);
      table += HASH_TABLE_ENTRY_SIZE;

      buckets.push_back ({ key, value });
    }

  return buckets;
}

/* Parse the Named Stream Map embedded in the PDB info stream and return
   it as a flat list of (name, stream_idx) pairs.  See LLVM "PDB Info
   Stream" docs for the on-disk layout.  On corruption, emits a warning
   and returns an empty map.  */

static pdb_named_stream_map
pdb_parse_named_stream_map (const pdb_info_stream &info)
{
  pdb_named_stream_map result;

  const gdb_byte *map = info.map;
  uint32_t map_size = info.map_size;
  const gdb_byte *map_end = map + map_size;

  /* Check that the map's header is whole: 4 bytes each for StringBuffer
     Size, HashTable Size/Capacity, PresentWordCount and DeletedWordCount,
     20 bytes total.  */
  if (map_size < NAMES_MAP_STRBUF_SIZE_OFFS + 5 * NAMES_MAP_UINT32_SIZE)
    {
      /* TODO: distinguish an absent named stream map (map_size == 0) from
	 a corrupt one.  */
      pdb_warning ("PDB info stream too small for named stream map header");
      return result;
    }

  /* Get pointer to StringBuffer and its size.  */
  const gdb_byte *strbuf = map + NAMES_MAP_STRBUF_OFFS;
  uint32_t strbuf_size = read_u32 (map + NAMES_MAP_STRBUF_SIZE_OFFS);
  if (strbuf_size > map_size - NAMES_MAP_STRBUF_OFFS)
    {
      pdb_warning ("PDB named stream map string buffer overflows its stream");
      return result;
    }
  const gdb_byte *strbuf_end = strbuf + strbuf_size;

  /* The serialized hash table follows the StringBuffer; each bucket key is
     an offset into that buffer and each value a stream index.  */
  auto buckets = pdb_decode_hash_table (strbuf_end, map_end);
  if (!buckets)
    return result;

  result.entries.reserve (buckets->size ());
  for (const auto &b : *buckets)
    {
      if (b.key >= strbuf_size)
	continue;

      const char *name = pdb_extract_string (strbuf + b.key, strbuf_end);
      if (name == nullptr)
	continue;
      result.entries.push_back ({ name, b.value });
    }

  return result;
}

/* Read the /names stream and fill the global string table.
   INFO is the already-decoded PDB info stream header; we reuse it for
   the named-stream map.  */

static bool
pdb_read_names_stream (pdb_per_objfile *pdb, const pdb_info_stream &info)
{
  pdb->string_table = nullptr;
  pdb->string_table_size = 0;

  /* Decode the Named Stream Map and look up "/names".  */
  auto nsm = pdb_parse_named_stream_map (info);
  uint32_t names_stream = nsm.lookup ("/names");

  if (names_stream == PDB_NO_STREAM || names_stream >= pdb->streams.size ())
    {
      pdb_load_diagnostic (pdb->objfile, "/names stream not found");
      return false;
    }

  /* Read the /names stream and fill the global string table.  The data moves
     into pdb->names_data, and string_table points into it for the objfile
     lifetime.  */

  pdb->names_data = pdb->read_stream (names_stream);
  if (pdb->names_data.empty ())
    {
      pdb_load_diagnostic (pdb->objfile, "No data in /names stream");
      return false;
    }

  gdb_byte *names_p = pdb->names_data.data ();
  size_t names_size = pdb->names_data.size ();

  if (names_size < NAMES_STREAM_MIN_SIZE)
    {
      pdb_load_diagnostic (pdb->objfile,
			   "/names stream too small (%zu bytes)", names_size);
      return false;
    }

  uint32_t names_sig = read_u32 (names_p + NAMES_STREAM_SIGNATURE_OFFS);
  uint32_t data_sz = read_u32 (names_p + NAMES_STREAM_DATA_SIZE_OFFS);

  if (names_sig != NAMES_STREAM_SIGNATURE_VAL)
    pdb_error ("unexpected /names signature");

  if (data_sz > names_size - NAMES_STREAM_MIN_SIZE)
    pdb_error ("/names string data extends beyond stream");

  pdb->string_table = names_p + NAMES_STREAM_MIN_SIZE;
  pdb->string_table_size = data_sz;

  return true;
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

/* Read the code contributions of the Section Contribution substream,
   which follows the MOD_INFO_SIZE bytes of Module Info, into
   pdb->code_contribs.  A module header names only one contribution; a
   module compiled with a section per function (/Gy) has one per function.
   An absent or malformed substream leaves the list empty.  */

static void
pdb_read_section_contribs (pdb_per_objfile *pdb, uint32_t mod_info_size)
{
  const gdb_byte *dbi = pdb->dbi_data.data ();
  uint64_t offset = (uint64_t) DBI_HDR_SIZE + mod_info_size;
  uint32_t size = read_u32 (dbi + DBI_HDR_SECT_CONTRIB_OFFS);
  if (size < 4 || offset > pdb->dbi_data.size ()
      || size > pdb->dbi_data.size () - offset)
    return;

  const gdb_byte *p = dbi + offset;
  uint32_t version = read_u32 (p);
  size_t entry_size;
  if (version == DBI_SC_VER_60)
    entry_size = DBI_SC_ENTRY_SIZE;
  else if (version == DBI_SC_VER_2)
    entry_size = DBI_SC2_ENTRY_SIZE;
  else
    {
      pdb_load_diagnostic (pdb->objfile,
			   "unknown section contribution version 0x%x",
			   version);
      return;
    }

  size_t count = (size - 4) / entry_size;
  p += 4;
  for (size_t i = 0; i < count; i++, p += entry_size)
    {
      uint32_t characteristics = read_u32 (p + DBI_SC_CHARACTERISTICS_OFFS);
      uint32_t contrib_size = read_u32 (p + DBI_SC_SIZE_OFFS);
      uint16_t imod = read_u16 (p + DBI_SC_IMOD_OFFS);
      if (contrib_size == 0 || imod >= pdb->modules.size ()
	  || (characteristics & (PDB_SCN_CODE | PDB_SCN_EXECUTE)) == 0)
	continue;

      pdb->code_contribs.push_back ({ read_u16 (p + DBI_SC_ISECT_OFFS), imod,
				      read_u32 (p + DBI_SC_OFFSET_OFFS),
				      contrib_size });
    }

  pdb_dbg_printf ("DBI code section contributions: %zu",
		  pdb->code_contribs.size ());
}

/* Read DBI (Debug Info) stream.  Reads headers and Module Info Records in order
   to obtain the sizes and stream numbers of the debug information for each
   module (object file).  Throws via pdb_error on failure.  */

static void
pdb_read_dbi_stream (pdb_per_objfile *pdb)
{
  /* Read DBI stream into the owned buffer; module names and
     file_info_names_buffer point into it.  */
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
  pdb->psgsi_stream = read_u16 (dbi_hdr + DBI_HDR_PSGSI_STREAM_OFFS);
  pdb->sym_record_stream = read_u16 (dbi_hdr + DBI_HDR_SYM_RECORD_STREAM_OFFS);

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

  pdb_read_section_contribs (pdb, mod_info_size);
}

/* Read section addresses from BFD.  */

static bool
pdb_read_section_addresses (pdb_per_objfile *pdb)
{
  if (pdb->objfile->obfd == nullptr)
    {
      pdb_load_diagnostic (pdb->objfile,
			   "No BFD available for section mapping");
      return false;
    }

  for (asection *sect : gdb_bfd_sections (pdb->objfile->obfd))
    {
      pdb->section_addresses.push_back (bfd_section_vma (sect));
      pdb->section_sizes.push_back
	(pdb_section_extent (pdb->objfile->obfd.get (), sect));
    }

  if (pdb->section_addresses.empty ())
    {
      pdb_load_diagnostic (pdb->objfile, "No sections found in BFD");
      return false;
    }

  return true;
}

/* See pdb-internal.h.  */

CORE_ADDR
pdb_per_objfile::map_section_offset_to_pc (uint16_t section,
					   uint64_t offset) const
{
  if (!section_valid (section))
    {
      pdb_complaint ("Invalid section number %u (max %zu)", section,
		     section_addresses.size ());
      return 0;
    }

  if (offset > UINT32_MAX)
    {
      pdb_complaint ("section %u offset 0x%" PRIx64 " out of range", section,
		     offset);
      return 0;
    }

  /* Get section base address (VMA from PE header).  */
  CORE_ADDR section_base = section_addresses[section - 1];

  /* Apply relocation offset (difference between linked and loaded address).  */
  CORE_ADDR reloc_offset = objfile->section_offsets[section - 1];

  /* Calculate final address: section VMA + relocation + offset.  */
  CORE_ADDR result = section_base + reloc_offset + offset;

  return result;
}

/* See pdb-internal.h.  */

CORE_ADDR
pdb_per_objfile::map_section_offset_unrelocated (uint16_t section,
						 uint32_t offset)
{
  if (!section_valid (section))
    {
      pdb_complaint ("Invalid section number %u (max %zu)", section,
		     section_addresses.size ());
      return 0;
    }

  return section_addresses[section - 1] + offset;
}

/* See pdb-internal.h.  */

pdb_module_info *
pdb_module_at_pc (pdb_per_objfile *pdb, CORE_ADDR pc)
{
  for (const auto &c : pdb->code_contribs)
    {
      if (c.module_index >= pdb->modules.size ()
	  || !pdb->range_in_section (c.section, c.offset, c.size))
	continue;
      CORE_ADDR start = pdb->map_section_offset_to_pc (c.section, c.offset);
      if (pc >= start && pc - start < c.size)
	return &pdb->modules[c.module_index];
    }
  return nullptr;
}

/* Call CALLBACK (KIND, DATA, SIZE) for each C13 subsection of MODULE, whose
   stream bytes are MODULE_STREAM.  The C13 data follows the symbols and the
   C11 lines and may start with the CV signature; each subsection is a kind,
   a size and SIZE bytes of data padded to 4 bytes.  Throws via pdb_error
   when a subsection overflows the C13 data.  */

template<typename Callback>
static void
pdb_for_each_c13_subsection (const pdb_module_info *module,
			     gdb_byte *module_stream, Callback &&callback)
{
  gdb_byte *c13 = module_stream + module->sym_byte_size
		  + module->c11_byte_size;
  gdb_byte *p = c13;
  const gdb_byte *c13_end = p + module->c13_byte_size;

  if (p + CV_SIGNATURE_SIZE <= c13_end
      && read_u32 (p) == CV_SIGNATURE_C13)
    p += CV_SIGNATURE_SIZE;

  while (p + C13_SUBSECT_HEADER_SIZE <= c13_end)
    {
      uint32_t kind = read_u32 (p);
      uint32_t size = read_u32 (p + 4);
      p += C13_SUBSECT_HEADER_SIZE;

      if (size > (size_t) (c13_end - p))
	pdb_error ("C13 subsection type 0x%x size %u overflows stream", kind,
		   size);

      callback (kind, p, size);
      /* Subsections are padded to 4 bytes from the start of the C13
	 data.  */
      p = c13 + align_up ((size_t) (p - c13) + size, 4);
    }
}

/* Read a module stream and extract C13 file checksums.  Returns an owned
   buffer containing the stream data; the caller keeps it alive as long as
   the data is needed (for symbol parsing, line walking, etc.).
   File checksums are copied onto the objfile obstack so they survive
   after the buffer is freed.  */

gdb::byte_vector
pdb_read_module_stream (pdb_per_objfile *pdb, pdb_module_info *module)
{
  if (module->stream_number == MODI_NO_STREAM)
    return {};

  auto buf = pdb->read_stream (module->stream_number);
  if (buf.empty ())
    return {};

  uint32_t stream_size = pdb->streams[module->stream_number].size;

  /* Validate that sym + c11 + c13 sizes fit within the stream.  */
  uint64_t fixed_part_size = (uint64_t) module->sym_byte_size
			     + (uint64_t) module->c11_byte_size;

  if (fixed_part_size > stream_size)
    pdb_error ("Module %u: sym+c11 sizes exceed stream size",
	       module->stream_number);

  if (module->c13_byte_size > stream_size - fixed_part_size)
    pdb_error ("Module %u: c13 size exceeds stream size",
	       module->stream_number);

  /* Copy SIZE bytes at DATA onto the objfile obstack into *COPY, unless
     they are there already, so that they outlive the stream buffer.  */
  auto keep = [pdb] (const gdb_byte **copy, uint32_t *copy_size,
		     const gdb_byte *data, uint32_t size)
    {
      if (*copy != nullptr)
	return;
      gdb_byte *mem
	= (gdb_byte *) obstack_alloc (&pdb->objfile->objfile_obstack, size);
      memcpy (mem, data, size);
      *copy = mem;
      *copy_size = size;
    };

  /* DEBUG_S_LINES is read when the module is expanded.  */
  pdb_for_each_c13_subsection
    (module, buf.data (),
     [&] (uint32_t kind, const gdb_byte *data, uint32_t size)
       {
	 if (kind == DEBUG_S_FILECHKSMS)
	   keep (&module->file_checksums, &module->file_checksums_size, data,
		 size);
       });

  return buf;
}

/* Get string from the global string table (/names stream) at given offset.  */

static const char *
pdb_string_table_get (const gdb_byte *string_table, uint32_t table_size,
		      uint32_t offset)
{
  if (string_table == nullptr || offset >= table_size)
    return nullptr;

  return pdb_extract_string (string_table + offset, string_table + table_size);
}

/* Resolve a DEBUG_S_FILECHKSMS file id to a source filename via the /names
   string table.  Returns nullptr if MODULE has no checksums, FILE_ID is
   out of range, or the name is not in the string table.  */

static const char *
pdb_get_filename_from_file_id (const pdb_per_objfile *pdb,
			       pdb_module_info *module, uint32_t file_id)
{
  if (module->file_checksums == nullptr)
    return nullptr;

  /* Access filename through the global string table from /names stream.  */
  if (pdb->string_table == nullptr)
    return nullptr;

  const gdb_byte *p = module->file_checksums;

  /* Check that FILE_ID, a byte offset into the DEBUG_S_FILECHKSMS subsection,
     leaves the checksum record header there whole.  A block whose file cannot
     be found keeps its lines, under an unknown file.  */
  if (file_id >= module->file_checksums_size
      || module->file_checksums_size - file_id < CV_FileChecksum::HDR_SIZE)
    {
      pdb_complaint ("file_id %u beyond file checksums size %u", file_id,
		     module->file_checksums_size);
      return nullptr;
    }

  CV_FileChecksum checksum (p + file_id);

  return pdb_string_table_get (pdb->string_table, pdb->string_table_size,
			       checksum.file_name_offs);
}

/* Walk all DEBUG_S_LINES subsections in a module's C13 data.  For each file
   block, invokes CALLBACK with the resolved filename and the raw line array
   view.  MODULE_STREAM is the raw module stream data.  */

static void
pdb_walk_c13_line_blocks (const pdb_per_objfile *pdb, pdb_module_info *module,
			  gdb_byte *module_stream, pdb_line_block_fn callback)
{
  /* Callers must verify there is C13 data before calling.  */
  gdb_assert (module->c13_byte_size > 0);

  /* See pdb-internal.h for the C13 layout.  */
  pdb_for_each_c13_subsection
    (module, module_stream,
     [&] (uint32_t kind, gdb_byte *data, uint32_t size)
       {
	 if (kind != DEBUG_S_LINES)
	   return;

	 gdb_byte *sp = data;
	 const gdb_byte *sp_end = data + size;

	 /* Ensure there's enough space for the header.  */
	 if (sp + CV_LineSection::SIZE > sp_end)
	   pdb_error ("DEBUG_S_LINES subsection too small for header"
		      " (%u bytes)", size);

	 CV_LineSection ls (sp);
	 sp += CV_LineSection::SIZE;

	 /* Each line has a CV_Line and, with CV_LINES_HAVE_COLUMNS, a 4-byte
	    column record.  */
	 uint64_t entry_size
	   = CV_Line::SIZE + ((ls.flags & CV_LINES_HAVE_COLUMNS) != 0 ? 4 : 0);

	 /* Iterate file blocks within this subsection.  */
	 while (sp + CV_FileBlock::SIZE <= sp_end)
	   {
	     CV_FileBlock fb (sp);
	     uint32_t file_id = fb.file_id;
	     uint32_t num_lines = fb.num_lines;
	     uint32_t block_size = fb.block_size;

	     if (block_size < CV_FileBlock::SIZE
		 || block_size > (size_t) (sp_end - sp))
	       break;

	     gdb_byte *line_data = sp + CV_FileBlock::SIZE;
	     if (num_lines * entry_size > block_size - CV_FileBlock::SIZE)
	       break;

	     /* Resolve filename from file checksums + string table.  */
	     auto fname = pdb_get_filename_from_file_id (pdb, module, file_id);
	     if (fname == nullptr)
	       fname = "<unknown>";

	     callback (fname, ls, line_data, num_lines);

	     sp += block_size;
	   }
       });
}

/* Callback body for pdb_walk_c13_line_blocks: queue one file's line rows
   for CU.  Converts the file path for display, opens a subfile, then
   resolves each CV_Line to a PC via the section map and queues a row for
   it; pdb_emit_lines records them unrelocated.  Lines that don't map
   to any PC (pc == 0) are skipped.

   Each DEBUG_S_LINES subsection covers one contiguous code range
   (CV_LineSection.code_sz bytes from sect:offs).  We close it with an
   end-of-sequence marker (line 0) at its end PC so the last line does not
   extend into whatever code follows the range.  */

static void
pdb_record_line_block (pdb_per_objfile *pdb, buildsym_compunit *cu,
		       const char *filename, CV_LineSection line_sect,
		       const gdb_byte *lines, uint32_t num_lines,
		       pdb_symbol_lines *sym_lines)
{
  uint16_t sect = line_sect.seci;
  uint32_t sect_offs = line_sect.offs;

  /* Convert MSYS2/Unix paths to Windows paths for source display.  */
  std::string path = pdb_convert_path (filename);

  cu->start_subfile (path.c_str (), path.c_str ());
  subfile *current_subfile = cu->get_current_subfile ();

  for (uint32_t i = 0; i < num_lines; i++)
    {
      CV_Line line (lines + i * CV_Line::SIZE);
      uint32_t offs = line.offs;
      uint32_t line_num = line.line_start;

      /* A row outside its section maps to no code of this module.  */
      uint64_t row_offs = (uint64_t) sect_offs + offs;
      if (row_offs > UINT32_MAX
	  || !pdb->range_in_section (sect, (uint32_t) row_offs, 0))
	continue;

      CORE_ADDR pc = pdb->map_section_offset_to_pc (sect, row_offs);
      if (pc == 0)
	continue;

      /* End the previous line at a row of 0xfeefee or 0xf00f00.  These values
	 mark code that belongs to no source line, such as compiler-generated
	 code.  */
      if (line_num == 0xfeefee || line_num == 0xf00f00)
	{
	  sym_lines->rows.push_back ({ pc, 0, current_subfile, LEF_IS_STMT });
	  continue;
	}

      /* Every row is a statement.  The CV_Line fStatement bit carries no
	 per-row information from either producer: Microsoft cl clears no
	 row's bit and clang-cl sets no row's bit, at /Od and /O2 alike.
	 GDB places line breakpoints and ends steps only at statement rows,
	 so taking clang-cl's bit at its word would remove them all.  */
      linetable_entry_flags flags = LEF_IS_STMT;

      sym_lines->rows.push_back ({ pc, (int) line_num, current_subfile,
				   flags });

      /* Remember the file the line at PC counts within, for a symbol that
	 starts at PC.  First entry wins.  */
      sym_lines->at_pc.emplace (pc, current_subfile);
    }

  /* Close the contribution with an end-of-sequence marker at its end PC, so
     the last line stops here instead of extending into the following code.  */
  if (line_sect.code_sz != 0)
    {
      uint64_t end_offset = (uint64_t) sect_offs + line_sect.code_sz;
      CORE_ADDR end_pc = 0;
      if (end_offset <= UINT32_MAX
	  && pdb->range_in_section (sect, (uint32_t) end_offset, 0))
	end_pc = pdb->map_section_offset_to_pc (sect, end_offset);
      if (end_pc != 0)
	sym_lines->rows.push_back ({ end_pc, 0, current_subfile, LEF_IS_STMT });
    }
}

/* Hand the module's collected rows to buildsym in address order, closing a
   file's run wherever the next code belongs to a different file.

   CodeView gives each file its own line blocks, and their address ranges
   interleave when one function's code comes from several files (#line
   directives, inlining).  Each file's table alone would then show a line
   running over the other file's code, and GDB takes a line's extent from the
   next entry in the same table, so `next` would step straight past the
   other file's statement.

   A DWARF line program is already one address-ordered stream, so
   lnp_state_machine just watches for the file to change and closes the
   previous run (gdb/dwarf2/line-program.c).  Sorting is what CodeView needs
   extra; the loop below is that same rule.  */

static void
pdb_emit_lines (pdb_per_objfile *pdb, buildsym_compunit *cu,
		pdb_symbol_lines *sym_lines)
{
  auto &rows = sym_lines->rows;

  /* Put an end marker first among the rows at one PC.  It ends the block below,
     and buildsym drops the rows recorded at an end marker's PC.  */
  std::stable_sort (rows.begin (), rows.end (),
		    [] (const pdb_symbol_lines::row &a,
			const pdb_symbol_lines::row &b)
		    {
		      if (a.pc != b.pc)
			return a.pc < b.pc;
		      return a.line == 0 && b.line != 0;
		    });

  CORE_ADDR text_off = pdb->objfile->text_section_offset ();
  subfile *last_subfile = nullptr;
  int last_line = 0;

  for (const auto &r : rows)
    {
      unrelocated_addr addr { r.pc - text_off };

      if (r.line == 0)
	{
	  cu->record_line (r.subfile, 0, addr, LEF_IS_STMT);
	  last_subfile = nullptr;
	  continue;
	}

      if (last_subfile != nullptr && last_subfile != r.subfile)
	cu->record_line (last_subfile, 0, addr, LEF_IS_STMT);

      /* clang-cl writes one CV_Line per source line by default, and one
	 per expression with -gcolumn-info, so a line can have several
	 consecutive rows.  Only the first of them is a statement, as in
	 clang's DWARF.  */
      linetable_entry_flags flags = r.flags;
      if (last_subfile == r.subfile && last_line == r.line)
	flags &= ~LEF_IS_STMT;

      cu->record_line (r.subfile, r.line, addr, flags);
      last_subfile = r.subfile;
      last_line = r.line;
    }
}

/* Give each symbol queued in SYM_LINES its symtab.  Call once the
   compunit has ended, so its subfiles have symtabs, and before it is
   destroyed.  */

static void
pdb_apply_symbol_lines (const pdb_symbol_lines &sym_lines)
{
  for (const auto &p : sym_lines.to_apply)
    if (p.subfile != nullptr && p.subfile->symtab != nullptr)
      p.sym->set_symtab (p.subfile->symtab);
}

static void pdb_read_module_files (pdb_per_objfile *pdb,
				   pdb_module_info *mod);

/* Build MODULE's compunit, for pdb_build_module.  Throws via pdb_error on
   malformed module data.  */

static compunit_symtab *
pdb_build_module_or_throw (pdb_per_objfile *pdb, pdb_module_info *module)
{
  /* Make symbol lookups issued from inside symtab construction (e.g.
     from check_typedef) bail out.  Without the flag they would re-enter
     the reader.  */
  scoped_restore decrementer = increment_reading_symtab ();

  /* Read module stream, owned locally and freed at scope exit.  */
  auto stream_buf = pdb_read_module_stream (pdb, module);

  /* Cache file list by reading File Info substream.  */
  pdb_read_module_files (pdb, module);

  /* A module with no code contribution (linker-generated, resource or
     import modules) has no address range.  A module without a symbol
     stream (an object built without debug info) gets no compunit, as
     DWARF gives such an object no CU.  */
  bool has_code = pdb->section_valid (module->sc_section);
  if (stream_buf.empty ())
    return nullptr;

  /* Compute module address range based on section contribution from DBI.
     A module compiled with a section per function (/Gy) has more code
     than this; pdb_module_at_pc finds a PC in any of it.  */
  CORE_ADDR low_pc = 0;
  CORE_ADDR high_pc = 0;
  if (has_code)
    {
      low_pc = pdb->map_section_offset_to_pc (module->sc_section,
					      module->sc_offset);
      high_pc = low_pc + module->sc_size;
      if (high_pc <= low_pc)
	high_pc = low_pc + 1;
    }

  /* The compunit is named after the module.  */
  std::string cu_name = module->module_name;

  enum language lang = pdb_module_language (pdb, module);

  /* Start the compunit at 0, because its start is known only once the
     procedures are read.  Address 0 keeps its global and static blocks
     ahead of the procedure blocks in the blockvector.  */
  buildsym_compunit cu (pdb->objfile, cu_name.c_str (),
			"",
			lang, 0);
  cu.record_debugformat ("CodeView");

  /* Record line tables (only if C13 data exists).  */
  pdb_symbol_lines sym_lines;
  if (module->c13_byte_size > 0)
    {
      pdb_walk_c13_line_blocks
	(pdb, module, stream_buf.data (),
	 [pdb, &cu, &sym_lines] (const char *filename,
				 CV_LineSection line_sect,
				 const gdb_byte *lines,
				 uint32_t num_lines)
	   {
	     pdb_record_line_block (pdb, &cu, filename, line_sect, lines,
				    num_lines, &sym_lines);
	   });

      pdb_emit_lines (pdb, &cu, &sym_lines);
    }

  /* Parse CodeView symbol records and collect function ranges.  */
  pdb_range_pair_vec func_ranges;
  pdb_parse_symbols (pdb, module, stream_buf.data (), &cu, &func_ranges,
		       lang, &sym_lines);

  /* A module without a code contribution gets no compunit: it declares
     nothing but procedures, and those are functions the linker folded
     into an identical copy in another module, whose records give that
     copy's address, which the other module's compunit covers.  */
  if (!has_code)
    return nullptr;
  bool declares_nothing = func_ranges.empty ();

  /* A module that declares nothing and has neither procedure records nor
     line data gets no compunit.  The linker's "Import:<dll>" modules are
     like this: each holds the import thunks, stubs that jump to the DLL's
     functions, and describes each thunk only with an S_THUNK32 record,
     from which no symbol is made.  A compunit for such a module would
     cover the thunks' addresses, and "info functions" does not list a
     minimal symbol whose address a compunit covers.  DWARF has no CU for
     them either.  */
  if (declares_nothing && func_ranges.empty () && module->c13_byte_size == 0)
    return nullptr;

  /* A module with procedures spans them, as a DWARF CU's ranges cover its
     code.  The DBI section contribution can lie above them or cover only
     the first, and a span reaching past a single procedure would claim
     other modules' PCs, since one range gets no address map.  */
  if (!func_ranges.empty ())
    {
      low_pc = func_ranges[0].first;
      high_pc = func_ranges[0].second;
    }
  for (const auto &range : func_ranges)
    {
      low_pc = std::min (low_pc, range.first);
      high_pc = std::max (high_pc, range.second);
    }
  if (high_pc <= low_pc)
    high_pc = low_pc + 1;

  /* Give the main subfile the module's language, because the compunit takes its
     language from the main subfile.  As start_subfile does for C subfiles, a C
     module whose source is a C++ or f2c file name takes that language, as
     DWARF's CU name gives.  */
  enum language main_lang = lang;
  if (lang == language_c)
    {
      for (uint16_t i = 0; module->files != nullptr && i < module->num_files;
	   i++)
	{
	  enum language l = deduce_language_from_filename (module->files[i]);
	  if (is_cplus_dialect (l) || l == language_fortran)
	    {
	      main_lang = l;
	      break;
	    }
	}
    }
  cu.start_subfile (cu_name.c_str (), cu_name.c_str ());
  subfile *main_subfile = cu.get_current_subfile ();
  main_subfile->language = main_lang;

  /* Register function ranges on the static block so the CU-level scope
     covers all functions even when they are scattered at non-contiguous
     addresses.  */
  block *static_block = cu.end_compunit_symtab_get_static_block (high_pc,
								 false, true);

  std::vector<blockrange> rangevec;
  rangevec.reserve (func_ranges.size ());
  for (auto &range : func_ranges)
    rangevec.emplace_back (range.first, range.second);
  if (!rangevec.empty ())
    static_block->set_ranges (make_blockranges (pdb->objfile, rangevec));

  module->cu = cu.end_compunit_symtab_from_static_block (static_block, false);

  if (module->cu != nullptr)
    {
      blockvector *bv = module->cu->blockvector ();
      bv->global_block ()->set_start (low_pc);
      bv->static_block ()->set_start (low_pc);
    }

  /* Set the primary source file's language to the module's, from its
     S_COMPILE2/S_COMPILE3 record, instead of the one GDB took from the
     file name, as DWARF does with DW_AT_language: a ".c" file compiled
     as C++ (-TP) is C++.  A C module whose file name gave a language
     keeps it: GDB turns the C files of a compunit that also has a C++ or
     Fortran file name (as f2c output does) into C++ or Fortran.  */
  if (module->cu != nullptr
      && !(lang == language_c
	   && module->cu->primary_filetab ()->language () != language_unknown))
    module->cu->primary_filetab ()->set_language (lang);

  /* The subfiles now have symtabs, so each queued function can be filed
     under the source file its code actually came from.

     Its line is deliberately not set.  CodeView records no declaration
     line, and the only candidate, the first line-table entry at the entry
     PC, is the first executable line rather than the declaration.  Giving
     a function a line later than its declaration stops GDB advancing
     "break FILE:LINE" from a line with no code to the next one that has
     some.  */
  pdb_apply_symbol_lines (sym_lines);

  return module->cu;
}

/* Expand module into its own compunit_symtab.  Returns the cached CU on
   subsequent calls.  Returns nullptr if the module has no useful debug
   data, or after warning that its data is malformed.  */

static compunit_symtab *
pdb_build_module (pdb_per_objfile *pdb, pdb_module_info *module)
{
  if (module->expanded)
    return module->cu;

  module->expanded = true;

  /* A malformed module loses only its own symbols, and only once.  It stays
     expanded, without a compunit.  */
  try
    {
      return pdb_build_module_or_throw (pdb, module);
    }
  catch (const gdb_exception_error &ex)
    {
      pdb_load_diagnostic (pdb->objfile, "module %s skipped: %s",
			   module->module_name, ex.what ());
      return nullptr;
    }
}

/* Expand all modules.  Called during initial PDB load.  */

static void
pdb_expand_all_modules (pdb_per_objfile *pdb)
{
  using clock = std::chrono::steady_clock;
  auto start = clock::now ();

  unsigned int expanded = 0;
  for (auto &module : pdb->modules)
    {
      if (module.expanded)
	continue;
      pdb_build_module (pdb, &module);
      expanded++;
    }

  double ms = std::chrono::duration<double, std::milli>
    (clock::now () - start).count ();
  pdb_dbg_printf ("expanded all %u/%zu modules in %.2f ms", expanded,
		  pdb->modules.size (), ms);
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

/* Process the File Info substream from the DBI stream.
   Create the Names Buffer and precompute module pointers into the Names Buffer.
   Layout:
     uint16_t  NumModules;
     uint16_t  NumSourceFiles;
     uint16_t  ModIndices[NumModules];                skipped
     uint16_t  ModFileCounts[NumModules];             file count per module
     uint32_t  FileNameOffsets[Sum(ModFileCounts)];   offsets into NamesBuffer
     char      NamesBuffer[];                         filename strings.  */

static void
pdb_read_file_info_substream (pdb_per_objfile *pdb)
{
  if (pdb->dbi_data.empty ())
    pdb_error ("Can't read file info: DBI stream not loaded");

  gdb_byte *dbi = pdb->dbi_data.data ();

  uint32_t mod_info_size = read_u32 (dbi + DBI_HDR_MODULE_SIZE_OFFS);
  uint32_t sect_contrib_size = read_u32 (dbi + DBI_HDR_SECT_CONTRIB_OFFS);
  uint32_t sect_map_size = read_u32 (dbi + DBI_HDR_SECT_MAP_OFFS);
  uint32_t source_info_size = read_u32 (dbi + DBI_HDR_SOURCE_INFO_OFFS);

  /* Sum the substream sizes without wrapping, because they are file data.  */
  uint64_t offset = ((uint64_t) DBI_HDR_SIZE + mod_info_size
		     + sect_contrib_size + sect_map_size);

  if (offset + source_info_size > pdb->dbi_data.size ())
    pdb_error ("File Info substream out of bounds");

  /* Without a File Info substream the modules name no files.  */
  if (source_info_size == 0)
    return;

  if (source_info_size < 4)
    pdb_error ("File Info substream too small");

  gdb_byte *fi = dbi + offset;
  const gdb_byte *fi_end = fi + source_info_size;

  /* Check NumModules against the low 16 bits of the module count.  NumModules
     is 16 bits wide, so a PDB of 65536 or more modules keeps only the low bits
     of the count, while the arrays have an entry per module.  */
  uint16_t fi_num_mods = read_u16 (fi + FILE_INFO_HDR_NUM_MODULES_OFFS);
  size_t num_mods = pdb->modules.size ();
  if (fi_num_mods != (uint16_t) num_mods)
    pdb_error ("File Info module count (%u), expected %zu", fi_num_mods,
	       num_mods);

  /* The ModIndices and ModFileCounts arrays must fit within the
     substream.  */
  uint64_t counts_offs = (uint64_t) FILE_INFO_MOD_INDICES_OFFS
			 + num_mods * FILE_INFO_ELEMENT_SIZE;
  if (counts_offs + num_mods * FILE_INFO_ELEMENT_SIZE > source_info_size)
    pdb_error ("File info substream too small");

  gdb_byte *mod_file_counts = fi + counts_offs;

  /* FileNameOffsets array starts after ModFileCounts.  */
  gdb_byte *offsets_base = mod_file_counts
			   + num_mods * FILE_INFO_ELEMENT_SIZE;

  /* Record the file name offsets of all the modules.  The FileNameOffsets array
     does not mark module boundaries, so the loop finds them from
     ModFileCounts.  */
  uint64_t sum = 0;
  for (size_t i = 0; i < num_mods; i++)
    sum += read_u16 (mod_file_counts + i * FILE_INFO_ELEMENT_SIZE);

  /* Names Buffer follows FileNameOffsets and contains null-terminated
     filename strings.  A module naming no source file leaves the buffer
     empty, so it may start exactly at the end of the substream.  */
  if (sum > (uint64_t) (fi_end - offsets_base) / 4)
    pdb_error ("File info names buffer out of bounds");

  sum = 0;
  for (size_t i = 0; i < num_mods; i++)
    {
      pdb->modules[i].file_name_offsets = offsets_base + sum * 4;
      uint16_t fc = read_u16 (mod_file_counts + i * FILE_INFO_ELEMENT_SIZE);
      pdb->modules[i].num_files = fc;
      /* Skip past this module's offsets.  */
      sum += fc;
    }

  gdb_byte *names_buf = offsets_base + sum * 4;
  pdb->file_info_names_buffer = names_buf;
  pdb->file_info_names_buffer_size = static_cast<uint32_t> (fi_end
							    - names_buf);
}

/* Index the TPI records of PDB.  A malformed TPI costs the types, not
   the PDB.  The reader reports it and reads the PDB with no type record,
   its functions without types, and its line tables and public symbols as
   they are.  The built-in types are still cached.  */

static void
pdb_read_tpi_or_none (pdb_per_objfile *pdb)
{
  try
    {
      pdb_read_tpi_stream (pdb);
    }
  catch (const gdb_exception_error &ex)
    {
      std::string message = ex.what ();
      if (startswith (message, "PDB Error: "))
	message.erase (0, strlen ("PDB Error: "));
      pdb_load_diagnostic (pdb->objfile, "%s: %s; types are not read",
			   pdb->pdb_file_path.c_str (), message.c_str ());
      pdb->tpi = pdb_tpi_context ();
      pdb->tpi.type_cache = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack,
					    0x1000, type *);
    }
}

/* Read the PDB that RSDS names for OBJFILE; pdb_initialize_objfile is the
   reader's entry point.  PDB parsing starts by reading the MSF SuperBlock,
   which contains basic info about the file and location of the stream
   directory.  Each stream spans multiple blocks that are not necessarily
   contiguous and the layout of every single stream is described in the
   stream directory.
   Next, we parse the /names stream which contains the global string table.
   Then we continue with the DBI stream, which describes the modules.  From
   it we get the list of modules, with the stream number and the sizes of
   each module's debug information.  */

static pdb_per_objfile *
pdb_read_pdb_file (objfile *objfile, const pdb_rsds_info &rsds)
{
  /* Read only the minimal symbols with OBJF_READNEVER.  They are still
     needed because lld writes no COFF symbol table, so the publics are the
     image's only symbol names, as .symtab is for an ELF file.  */
  const bool readnever = (objfile->flags & OBJF_READNEVER) != 0;

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

  /* Read & decode the PDB Info stream once; both GUID validation and the
     /names lookup consume it.  */
  pdb_info_stream info = pdb_read_info_stream (pdb);

  /* Validate PDB GUID/age against the EXE's RSDS record.  A mismatch means
     this PDB belongs to a different build, so loading is aborted.  The
     DBI age is checked once the DBI stream is read.  */
  if (!pdb_validate_guid (pdb, rsds, info, 0))
    return nullptr;

  /* Read the /names stream for global string table.  A /names stream
     that is missing or too small is reported there; the PDB is still
     read, without file names.  */
  pdb_read_names_stream (pdb, info);

  /* Read DBI stream (contains modules and line info).  */
  pdb_read_dbi_stream (pdb);
  if (!pdb_validate_guid (pdb, rsds, info, pdb->dbi_age))
    return nullptr;

  pdb_read_file_info_substream (pdb);

  /* Index the TPI records; a type is built the first time a symbol needs
     it.  */
  if (!readnever)
    pdb_read_tpi_or_none (pdb);

  /* Read section addresses from BFD to map a section and offset to a
     PC.  */
  if (!pdb_read_section_addresses (pdb))
    return nullptr;

  /* Read and cache the symbol record stream.  */
  pdb_read_sym_record_stream (pdb);

  /* Build minimal symbols from S_PUB32 records in the public symbol stream.  */
  pdb_build_minsyms (pdb);

  if (info_verbose)
    gdb_printf ("\nUsing PDB file: %s\n", pdb->pdb_file_path.c_str ());
  return pdb;
}

/* The PDB reader expands all modules when it loads the PDB.  Below are
   minimal quick functions to support this use model.  */
struct pdb_readnow_functions : public quick_symbol_functions
{
  bool has_symbols (objfile *) override
  {
    return true;
  }

  bool has_unexpanded_symtabs (objfile *) override
  {
    return false;
  }

  symtab *find_last_source_symtab (objfile *) override
  {
    return nullptr;
  }

  void forget_cached_source_info (objfile *) override
  {
    /* PDB has no internal file name cache.  Symtab fullname cleanup is handled
       by GDB generically.  */
  }

  enum language lookup_global_symbol_language (objfile *, const char *,
					       domain_search_flags,
					       bool *symbol_found_p) override
  {
    *symbol_found_p = false;
    return language_unknown;
  }

  void print_stats (objfile *, bool) override
  {
  }

  void dump (objfile *) override
  {
  }

  void expand_all_symtabs (objfile *) override
  {
  }

  iteration_status
  search (objfile *objfile, search_symtabs_file_matcher file_matcher,
	  const lookup_name_info *,
	  search_symtabs_symbol_matcher,
	  compunit_symtab_iteration_callback compunit_callback,
	  block_search_flags, domain_search_flags,
	  search_symtabs_lang_matcher = nullptr) override
  {
    if (compunit_callback == nullptr)
      return iteration_status::keep_going;

    for (compunit_symtab &cu : objfile->compunits ())
      {
	/* If a file matcher is supplied, check whether any filetab
	   in this compunit matches.  */
	if (file_matcher != nullptr)
	  {
	    bool match = false;
	    for (const symtab *s : cu.filetabs ())
	      {
		if (file_matcher (s->filename (), false)
		    || file_matcher (lbasename (s->filename ()), true))
		  {
		    match = true;
		    break;
		  }
	      }

	    if (!match)
	      continue;
	  }

	if (compunit_callback (&cu) == iteration_status::stop)
	  return iteration_status::stop;
      }

    return iteration_status::keep_going;
  }

  compunit_symtab *find_pc_sect_compunit_symtab (objfile *objfile,
						 bound_minimal_symbol,
						 CORE_ADDR pc,
						 obj_section *,
						 int) override
  {
    /* Iterate PDB modules directly to find matching PC.  */
    pdb_per_objfile *pdb = pdb_objfile_data_key.get (objfile);
    if (pdb == nullptr)
      return nullptr;

    /* A module's code contributions, of which its compunit's range is
       only the first.  */
    if (pdb_module_info *module = pdb_module_at_pc (pdb, pc);
	module != nullptr && module->cu != nullptr)
      {
	pdb_qf_printf ("find_pc_sect: module %zu matched by contribution",
		       (size_t) (module - pdb->modules.data ()));
	return module->cu;
      }

    for (uint32_t i = 0; i < pdb->modules.size (); i++)
      {
	pdb_module_info *module = &pdb->modules[i];

	/* Skip modules without a compunit.  */
	if (module->cu == nullptr)
	  continue;

	const blockvector *bv = module->cu->blockvector ();

	if (bv->contains (pc))
	  {
	    pdb_qf_printf ("find_pc_sect: module %u matched", i);
	    return module->cu;
	  }
      }

    pdb_qf_printf ("find_pc_sect: no match");
    return nullptr;
  }

  symbol *find_symbol_by_address (objfile *objfile, CORE_ADDR address) override
  {
    /* Readnow has expanded every module, so the built compunits hold the
       answer.  */
    for (compunit_symtab &cu : objfile->compunits ())
      {
	if (symbol *sym = cu.symbol_at_address (address))
	  return sym;
      }
    return nullptr;
  }

  void map_symbol_filenames (objfile *, symbol_filename_listener,
			     bool) override
  {
  }
};

/* Get module's file names from its precomputed file_name_offsets[] pointers
   into the global Names Buffer.  */

static void
pdb_read_module_files (pdb_per_objfile *pdb, pdb_module_info *module)
{
  /* Already populated.  */
  if (module->files != nullptr)
    return;

  if (module->file_name_offsets == nullptr
      || pdb->file_info_names_buffer == nullptr)
    return;

  uint16_t file_count = module->num_files;
  if (file_count == 0)
    return;

  const gdb_byte *my_offsets = module->file_name_offsets;
  const gdb_byte *nb = pdb->file_info_names_buffer;
  uint32_t nb_size = pdb->file_info_names_buffer_size;

  module->files = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack, file_count,
				  const char *);
  for (uint16_t i = 0; i < file_count; i++)
    {
      uint32_t name_offset = read_u32 (my_offsets + i * sizeof (uint32_t));
      if (name_offset < nb_size)
	module->files[i] = pdb_extract_string (nb + name_offset, nb + nb_size);
      else
	module->files[i] = nullptr;
    }
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

  if ((objfile->flags & OBJF_READNEVER) != 0)
    return true;

  /* Expand all modules eagerly.  */
  pdb_expand_all_modules (pdb);
  objfile->qf.emplace_front (new pdb_readnow_functions);

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

/* Return the bytes of a PDB Info stream whose named stream map holds NAMES,
   each name mapped to the stream number at its position plus 10.  CAPACITY is
   the hash table's capacity, and every name is in a present bucket.  */

static gdb::byte_vector
info_stream (const std::vector<std::string> &names, uint32_t capacity = 8)
{
  gdb::byte_vector info (INFO_STREAM_HASH_MAP_OFFS, 0);
  put_u32 (info, 0, 20000404);

  std::string strbuf;
  std::vector<uint32_t> keys;
  for (const std::string &name : names)
    {
      keys.push_back (strbuf.size ());
      strbuf += name;
      strbuf += '\0';
    }
  put_u32 (info, info.size (), strbuf.size ());
  info.insert (info.end (), strbuf.begin (), strbuf.end ());

  put_u32 (info, info.size (), names.size ());
  put_u32 (info, info.size (), capacity);
  uint32_t words = (capacity + 31) / 32;
  put_u32 (info, info.size (), words);
  for (uint32_t w = 0; w < words; w++)
    {
      uint32_t bits = 0;
      for (uint32_t b = 0; b < 32; b++)
	if (w * 32 + b < names.size ())
	  bits |= 1u << b;
      put_u32 (info, info.size (), bits);
    }
  put_u32 (info, info.size (), 0);
  for (size_t i = 0; i < names.size (); i++)
    {
      put_u32 (info, info.size (), keys[i]);
      put_u32 (info, info.size (), i + 10);
    }
  return info;
}

/* Return the offset of the hash table in an info_stream, past the string buffer
   of NAMES.  */

static size_t
hash_table_offset (const std::vector<std::string> &names)
{
  size_t strbuf = 0;
  for (const std::string &name : names)
    strbuf += name.size () + 1;
  return INFO_STREAM_HASH_MAP_OFFS + 4 + strbuf;
}

/* Test the named stream map of the PDB Info stream.  Build a valid map and maps
   with corrupt counts, offsets and names, and check that the reader checks each
   count, in 64 bits, before it uses it, because the counts are file data.  */

static void
test_named_stream_map ()
{
  const std::vector<std::string> names { "/names", "/LinkInfo",
					 "/src/headerblock" };
  size_t ht = hash_table_offset (names);

  auto parse = [] (const gdb::byte_vector &info)
    {
      return pdb_parse_named_stream_map (pdb_decode_info_stream (info));
    };

  gdb::byte_vector info = info_stream (names);
  pdb_named_stream_map map = parse (info);
  SELF_CHECK (map.entries.size () == 3);
  SELF_CHECK (map.lookup ("/names") == 10);
  SELF_CHECK (map.lookup ("/src/headerblock") == 12);
  SELF_CHECK (map.lookup ("/missing") == PDB_NO_STREAM);

  /* PresentWordCount * 4 wraps to 4, so a 32-bit check would pass.  */
  gdb::byte_vector bad = info;
  put_u32 (bad, ht + 8, 0x40000001);
  SELF_CHECK (parse (bad).entries.empty ());

  /* Check a DeletedWordCount past the stream.  */
  bad = info;
  put_u32 (bad, ht + 16, 0x40000000);
  SELF_CHECK (parse (bad).entries.empty ());

  /* No bucket exists past the present bits, so neither a huge Size nor a
     huge Capacity is acted on.  */
  for (size_t field : { 0, 4 })
    {
      bad = info;
      put_u32 (bad, ht + field, 0xffffffff);
      SELF_CHECK (parse (bad).lookup ("/names") == 10);
    }

  /* The table may end right after its last present bucket.  */
  bad = info_stream (names, 64);
  SELF_CHECK (parse (bad).entries.size () == 3);

  /* Check a table with one bucket fewer than the present bits promise.  */
  bad = info;
  bad.resize (bad.size () - 8);
  SELF_CHECK (parse (bad).entries.size () == 2);

  /* Check a string buffer past the stream.  */
  bad = info;
  put_u32 (bad, INFO_STREAM_HASH_MAP_OFFS, 0x7fffffff);
  SELF_CHECK (parse (bad).entries.empty ());

  /* Check a key past the string buffer, and a name without its terminator.  */
  bad = info;
  put_u32 (bad, bad.size () - 16, 1000);
  SELF_CHECK (parse (bad).lookup ("/LinkInfo") == PDB_NO_STREAM);
  SELF_CHECK (parse (bad).lookup ("/names") == 10);
  bad = info_stream ({ "/names" });
  bad[INFO_STREAM_HASH_MAP_OFFS + 4 + 6] = 'X';
  SELF_CHECK (parse (bad).entries.empty ());

  /* Check a stream too small for the map's header.  */
  bad.resize (INFO_STREAM_HASH_MAP_OFFS + 8);
  SELF_CHECK (parse (bad).entries.empty ());

  /* Check an info stream smaller than its fixed header.  */
  bad.resize (INFO_STREAM_MIN_SIZE - 1);
  SELF_CHECK (error_of ([&] { pdb_decode_info_stream (bad); })
	      .find ("too small") != std::string::npos);
}

/* Test reading the /names stream, whose data size is file data.  Build a valid
   stream, streams whose data size wraps or passes the stream, one with a bad
   signature and one too small for its header, and check how each is read.  */

static void
test_names_stream ()
{
  gdb::byte_vector names;
  put_u32 (names, NAMES_STREAM_SIGNATURE_OFFS, NAMES_STREAM_SIGNATURE_VAL);
  put_u32 (names, 4, 1);
  const char data[] = "\0a.c\0b.h";
  put_u32 (names, NAMES_STREAM_DATA_SIZE_OFFS, sizeof data);
  names.insert (names.end (), data, data + sizeof data);

  auto read = [&] (const gdb::byte_vector &stream, bool *found)
    {
      test_pdb file;
      file.streams.resize (11);
      file.streams[1] = info_stream ({ "/names" });
      file.streams[10] = stream;
      file.build ();
      test_pdb_reader r (file);
      r.read_msf ();
      pdb_info_stream info = pdb_read_info_stream (&r.pdb);
      std::string err = error_of ([&]
	{ *found = pdb_read_names_stream (&r.pdb, info); });
      if (*found)
	*found = (r.pdb.string_table_size == sizeof data
		  && strcmp ((const char *) r.pdb.string_table + 1,
			     "a.c") == 0);
      return err;
    };

  bool found;
  SELF_CHECK (read (names, &found).empty () && found);

  /* The 12 bytes of the header plus a data size of 0xfffffff8 wrap to 4.  */
  gdb::byte_vector bad = names;
  put_u32 (bad, NAMES_STREAM_DATA_SIZE_OFFS, 0xfffffff8);
  SELF_CHECK (read (bad, &found) == "/names string data extends beyond stream");

  bad = names;
  put_u32 (bad, NAMES_STREAM_DATA_SIZE_OFFS, sizeof data + 1);
  SELF_CHECK (read (bad, &found) == "/names string data extends beyond stream");

  bad = names;
  put_u32 (bad, NAMES_STREAM_SIGNATURE_OFFS, 0x12345678);
  SELF_CHECK (read (bad, &found) == "unexpected /names signature");

  /* Check that a stream too small for its header leaves the PDB read without
     file names.  */
  bad.assign (names.begin (), names.begin () + 8);
  SELF_CHECK (read (bad, &found).empty () && !found);
}

/* Test reading the File Info substream.  Check that the DBI header's substream
   sizes, which are file data, are summed without wrapping.  */

static void
test_dbi_substreams ()
{
  pdb_per_objfile pdb (nullptr);
  pdb.dbi_data.assign (DBI_HDR_SIZE + 0x200, 0);
  put_u32 (pdb.dbi_data, DBI_HDR_SIGNATURE_OFFS, DBI_HDR_SIGNATURE_VAL);

  /* Summed in 32 bits, these would place the File Info substream inside
     the module records.  */
  put_u32 (pdb.dbi_data, DBI_HDR_MODULE_SIZE_OFFS, 0x10);
  put_u32 (pdb.dbi_data, DBI_HDR_SECT_CONTRIB_OFFS, 0xffffff00);
  put_u32 (pdb.dbi_data, DBI_HDR_SECT_MAP_OFFS, 0x90);
  put_u32 (pdb.dbi_data, DBI_HDR_SOURCE_INFO_OFFS, 0x20);
  SELF_CHECK (error_of ([&] { pdb_read_file_info_substream (&pdb); })
	      == "File Info substream out of bounds");

  put_u32 (pdb.dbi_data, DBI_HDR_SECT_CONTRIB_OFFS, 0);
  put_u32 (pdb.dbi_data, DBI_HDR_SECT_MAP_OFFS, 0);
  put_u32 (pdb.dbi_data, DBI_HDR_SOURCE_INFO_OFFS, 0x200);
  SELF_CHECK (error_of ([&] { pdb_read_file_info_substream (&pdb); })
	      == "File Info substream out of bounds");

  /* Without a File Info substream the modules name no files.  */
  put_u32 (pdb.dbi_data, DBI_HDR_MODULE_SIZE_OFFS, 0);
  put_u32 (pdb.dbi_data, DBI_HDR_SOURCE_INFO_OFFS, 0);
  SELF_CHECK (error_of ([&] { pdb_read_file_info_substream (&pdb); })
	      .empty ());

  /* Check 65536 modules.  The 16-bit module count reads 0, and the arrays have
     an entry per module.  */
  pdb_per_objfile many (nullptr);
  many.modules.resize (65536);
  uint32_t fi_size = 4 + 65536 * 2 * FILE_INFO_ELEMENT_SIZE;
  many.dbi_data.assign (DBI_HDR_SIZE + fi_size, 0);
  put_u32 (many.dbi_data, DBI_HDR_SIGNATURE_OFFS, DBI_HDR_SIGNATURE_VAL);
  put_u32 (many.dbi_data, DBI_HDR_SOURCE_INFO_OFFS, fi_size);
  SELF_CHECK (error_of ([&] { pdb_read_file_info_substream (&many); })
	      .empty ());

  /* Check arrays that are one module short.  */
  put_u32 (many.dbi_data, DBI_HDR_SOURCE_INFO_OFFS, fi_size - 2);
  SELF_CHECK (error_of ([&] { pdb_read_file_info_substream (&many); })
	      == "File info substream too small");
}

/* See pdb-internal.h.  */

test_objfile::test_objfile (size_t num_sections)
{
  gdbarch_info info;
  info.bfd_arch_info = bfd_scan_arch ("i386:x86-64");
  info.osabi = GDB_OSABI_WINDOWS;
  gdbarch *arch = gdbarch_find_by_info (info);
  if (arch == nullptr)
    return;

  objfile = objfile::make (nullptr, current_program_space, "pdb-selftest",
			   OBJF_NOT_FILENAME);
  objfile->section_offsets.assign (num_sections, 0);
  objfile->sect_index_text = 0;
  objfile->per_bfd->gdbarch = arch;
}

test_objfile::~test_objfile ()
{
  if (objfile != nullptr)
    objfile->unlink ();
}

/* Append the LEN low bytes of V to BYTES, little-endian.  */

static void
append_le (gdb::byte_vector &bytes, uint64_t v, int len)
{
  for (int i = 0; i < len; i++)
    bytes.push_back ((v >> (8 * i)) & 0xff);
}

/* See pdb-internal.h.  */

void
test_module::symbol (uint16_t kind, const gdb::byte_vector &body)
{
  /* Records are padded to 4 bytes, as the linker writes them.  */
  size_t padded = align_up (body.size () + 4, 4) - 4;
  append_le (syms, padded + 2, 2);
  append_le (syms, kind, 2);
  syms.insert (syms.end (), body.begin (), body.end ());
  syms.resize (syms.size () + padded - body.size (), 0);
}

/* See pdb-internal.h.  */

void
test_module::procedure (const char *name, uint16_t section, uint32_t offset,
			uint32_t size, uint32_t type, uint8_t flags)
{
  /* The record has no parent, end or next record, and the whole procedure lies
     between DbgStart and DbgEnd.  */
  gdb::byte_vector body;
  append_le (body, 0, 4);
  append_le (body, 0, 4);
  append_le (body, 0, 4);
  append_le (body, size, 4);
  append_le (body, 0, 4);
  append_le (body, size, 4);
  append_le (body, type, 4);
  append_le (body, offset, 4);
  append_le (body, section, 2);
  body.push_back (flags);
  body.insert (body.end (), name, name + strlen (name) + 1);
  symbol (S_GPROC32, body);
  symbol (S_END, {});
}

/* See pdb-internal.h.  */

uint32_t
test_module::file (uint32_t name_offset)
{
  uint32_t id = checksums.size ();
  /* Write the name and no checksum, and pad the entry to 4 bytes.  */
  append_le (checksums, name_offset, 4);
  append_le (checksums, 0, 4);
  return id;
}

/* See pdb-internal.h.  */

void
test_module::lines (uint16_t section, uint32_t offset, uint32_t size,
		    uint32_t file_id,
		    const std::vector<std::pair<uint32_t, uint32_t>> &rows)
{
  gdb::byte_vector sub;
  append_le (sub, offset, 4);
  append_le (sub, section, 2);
  append_le (sub, 0, 2);
  append_le (sub, size, 4);
  append_le (sub, file_id, 4);
  append_le (sub, rows.size (), 4);
  append_le (sub, 12 + 8 * rows.size (), 4);
  for (const auto &[row_offset, line] : rows)
    {
      append_le (sub, row_offset, 4);
      append_le (sub, line | 0x80000000, 4);
    }
  line_subsections.push_back (std::move (sub));
}

/* See pdb-internal.h.  */

gdb::byte_vector
test_module::stream (pdb_module_info *module) const
{
  gdb::byte_vector c13;
  auto subsection = [&] (uint32_t kind, const gdb::byte_vector &data)
    {
      append_le (c13, kind, 4);
      append_le (c13, data.size (), 4);
      c13.insert (c13.end (), data.begin (), data.end ());
      c13.resize (align_up (c13.size (), 4), 0);
    };
  if (!checksums.empty ())
    subsection (DEBUG_S_FILECHKSMS, checksums);
  for (const gdb::byte_vector &sub : line_subsections)
    subsection (DEBUG_S_LINES, sub);

  gdb::byte_vector bytes = syms;
  bytes.insert (bytes.end (), c13.begin (), c13.end ());
  module->sym_byte_size = syms.size ();
  module->c11_byte_size = 0;
  module->c13_byte_size = c13.size ();
  return bytes;
}

/* See pdb-internal.h.  */

test_module_reader::test_module_reader (const test_pdb &file,
					size_t num_sections)
  : obj (num_sections)
{
  if (obj.objfile == nullptr)
    return;

  pdb = get_or_create_pdb_per_objfile (obj.objfile);
  pdb->file_mapping.base = file.bytes.data ();
  pdb->file_mapping.size = file.bytes.size ();
  pdb_read_msf_header (pdb);
  pdb_read_stream_directory (pdb);
  for (size_t i = 0; i < num_sections; i++)
    {
      pdb->section_addresses.push_back (0x140001000 + 0x1000 * i);
      pdb->section_sizes.push_back (0x1000);
    }
}

test_module_reader::~test_module_reader ()
{
  /* The bytes belong to the test_pdb: nothing to unmap.  */
  if (pdb != nullptr)
    pdb->file_mapping.base = nullptr;
}

/* See pdb-internal.h.  */

uint32_t
test_tpi::add (uint16_t leaf, const gdb::byte_vector &body)
{
  /* The record is padded to 4 bytes with LF_PAD bytes, each counting the
     bytes left to the end.  */
  size_t padded = align_up (body.size () + 4, 4) - 4;
  append_le (records, padded + 2, 2);
  append_le (records, leaf, 2);
  records.insert (records.end (), body.begin (), body.end ());
  for (size_t left = padded - body.size (); left > 0; left--)
    records.push_back (0xf0 + left);
  return next++;
}

/* See pdb-internal.h.  */

uint32_t
test_tpi::arglist (const std::vector<uint32_t> &args)
{
  gdb::byte_vector body;
  append_le (body, args.size (), 4);
  for (uint32_t arg : args)
    append_le (body, arg, 4);
  return add (LF_ARGLIST, body);
}

/* See pdb-internal.h.  */

uint32_t
test_tpi::procedure (uint32_t ret, uint32_t arglist, uint16_t count)
{
  /* Write the return type, the near C calling convention and no
     attributes.  */
  gdb::byte_vector body;
  append_le (body, ret, 4);
  append_le (body, 0, 2);
  append_le (body, count, 2);
  append_le (body, arglist, 4);
  return add (LF_PROCEDURE, body);
}

/* See pdb-internal.h.  */

gdb::byte_vector
test_tpi::stream () const
{
  /* Write the header, which holds TPI_VERSION_V80, the header size of 56
     bytes, the index range and the records' size.  */
  gdb::byte_vector tpi;
  put_u32 (tpi, 0, 20040203);
  put_u32 (tpi, 4, 56);
  put_u32 (tpi, 8, 0x1000);
  put_u32 (tpi, 12, next);
  put_u32 (tpi, 16, records.size ());
  tpi.resize (56, 0);
  tpi.insert (tpi.end (), records.begin (), records.end ());
  return tpi;
}

/* The /names data naming a.cpp at offset 1 and b.cpp at offset 7.  */

static char test_names[] = { 0, 'a', '.', 'c', 'p', 'p', 0,
			     'b', '.', 'c', 'p', 'p', 0 };

/* See pdb-internal.h.  */

void
test_read_modules (std::vector<pdb_module_info> modules,
		   const std::vector<gdb::byte_vector> &streams,
		   const test_tpi &tpi,
		   gdb::function_view<void (pdb_per_objfile *pdb)> setup,
		   gdb::function_view<void (pdb_per_objfile *pdb)> check)
{
  test_pdb file;
  file.streams.resize (5);
  file.streams[PDB_STREAM_TPI] = tpi.stream ();
  for (size_t i = 0; i < streams.size (); i++)
    {
      modules[i].stream_number = file.streams.size ();
      file.streams.push_back (streams[i]);
    }
  file.build ();

  test_module_reader r (file, 1);
  if (r.pdb == nullptr)
    return;
  r.pdb->string_table = (const gdb_byte *) test_names;
  r.pdb->string_table_size = sizeof test_names;
  pdb_read_tpi_stream (r.pdb);
  r.pdb->modules = std::move (modules);
  if (setup != nullptr)
    setup (r.pdb);
  for (pdb_module_info &m : r.pdb->modules)
    pdb_build_module (r.pdb, &m);
  check (r.pdb);
}

/* Call test_read_modules for modules with no types and no setup.  */

static void
with_modules (std::vector<pdb_module_info> modules,
	      const std::vector<gdb::byte_vector> &streams,
	      gdb::function_view<void (pdb_per_objfile *pdb)> check)
{
  test_read_modules (std::move (modules), streams, test_tpi (), nullptr,
		     check);
}

/* Return the line at PC as its number and the base name of its file, or 0 and
   "" when PC has none.  */

static std::pair<int, std::string>
line_at (CORE_ADDR pc)
{
  symtab_and_line sal = find_sal_for_pc (pc, 0);
  if (sal.symtab == nullptr)
    return { 0, "" };
  return { sal.line, lbasename (sal.symtab->filename ()) };
}

/* Build two modules laid out as the linker lays them out.  A's procedures are
   at 0 and 0x100, and B's lies in the gap between them at 0x80.  Each module's
   lines are in its own file.  */

static void
two_modules (std::vector<pdb_module_info> *modules,
	     std::vector<gdb::byte_vector> *streams)
{
  test_module a, b;
  uint32_t fa = a.file (1), fb = b.file (7);
  a.procedure ("afirst", 1, 0, 16);
  a.procedure ("alast", 1, 0x100, 16);
  a.lines (1, 0, 16, fa, { { 0, 10 } });
  a.lines (1, 0x100, 16, fa, { { 0, 20 } });
  b.procedure ("bmid", 1, 0x80, 16);
  b.lines (1, 0x80, 16, fb, { { 0, 30 } });

  modules->resize (2);
  (*modules)[0].module_name = (char *) "a.obj";
  (*modules)[1].module_name = (char *) "b.obj";
  streams->push_back (a.stream (&(*modules)[0]));
  streams->push_back (b.stream (&(*modules)[1]));
}

/* Test that a PC belongs to the module whose procedures cover it,
   whatever the modules' DBI section contributions say.  Build modules
   where B's contribution starts above its procedure or covers A's
   procedures, and where B comes first or second.  */

static void
test_module_bounds ()
{
  struct layout
  {
    uint32_t a_offset, a_size, b_offset, b_size;
    bool b_first;
  };
  /* Place B's contribution above its procedure, then put B first with
     its contribution covering A's procedures.  */
  for (const layout &l : { layout { 0, 0x110, 0x90, 0x10, false },
			   layout { 0, 0x110, 0, 0x200, true } })
    {
      std::vector<pdb_module_info> modules;
      std::vector<gdb::byte_vector> streams;
      two_modules (&modules, &streams);
      modules[0].sc_section = modules[1].sc_section = 1;
      modules[0].sc_offset = l.a_offset;
      modules[0].sc_size = l.a_size;
      modules[1].sc_offset = l.b_offset;
      modules[1].sc_size = l.b_size;
      if (l.b_first)
	{
	  std::swap (modules[0], modules[1]);
	  std::swap (streams[0], streams[1]);
	}

      with_modules (modules, streams, [] (pdb_per_objfile *)
	{
	  SELF_CHECK (line_at (0x140001000) == std::make_pair (10,
							std::string ("a.cpp")));
	  SELF_CHECK (line_at (0x140001080) == std::make_pair (30,
							std::string ("b.cpp")));
	  SELF_CHECK (line_at (0x140001100) == std::make_pair (20,
							std::string ("a.cpp")));
	});
    }
}

/* Test that a damaged module costs only its own symbols and lines.  Build two
   modules, the C13 size of one overflowing its stream, and check that it is
   skipped and the other keeps its own.  Check also that a line whose section
   offset plus code offset passes 2^32 gets no row instead of one at the wrapped
   address.  */

static void
test_bad_module ()
{
  std::vector<pdb_module_info> modules;
  std::vector<gdb::byte_vector> streams;
  two_modules (&modules, &streams);
  modules[0].sc_section = modules[1].sc_section = 1;
  modules[0].sc_size = 0x110;
  modules[1].sc_offset = 0x80;
  modules[1].sc_size = 0x10;

  /* Do what with_modules does, but make A's C13 size overflow its stream after
     building the streams.  */
  test_pdb file;
  file.streams.resize (5);
  file.streams[PDB_STREAM_TPI] = test_tpi ().stream ();
  for (size_t i = 0; i < 2; i++)
    {
      modules[i].stream_number = file.streams.size ();
      file.streams.push_back (streams[i]);
    }
  modules[0].c13_byte_size = 0x7fffffff;
  file.build ();

  test_module_reader r (file, 1);
  if (r.pdb == nullptr)
    return;
  r.pdb->string_table = (const gdb_byte *) test_names;
  r.pdb->string_table_size = sizeof test_names;
  pdb_read_tpi_stream (r.pdb);
  r.pdb->modules = std::move (modules);
  SELF_CHECK (pdb_build_module (r.pdb, &r.pdb->modules[0]) == nullptr);
  SELF_CHECK (pdb_build_module (r.pdb, &r.pdb->modules[1]) != nullptr);
  SELF_CHECK (line_at (0x140001080) == std::make_pair (30,
						       std::string ("b.cpp")));
  SELF_CHECK (line_at (0x140001000).first == 0);
  SELF_CHECK (find_symbol_for_pc (0x140001080) != nullptr
	      && strcmp (find_symbol_for_pc (0x140001080)->search_name (),
			 "bmid()") == 0);

  /* Map a line block at section offset 0xfffffff0 whose row is 0x20 bytes in.
     The sum is not a 32-bit section offset.  */
  pdb_per_objfile pdb (r.obj.objfile);
  pdb.section_addresses = { 0x140001000 };
  pdb.section_sizes = { 0x1000 };
  SELF_CHECK (pdb.map_section_offset_to_pc (1, 0xfffffff0ull + 0x20) == 0);
  SELF_CHECK (pdb.map_section_offset_to_pc (1, 0x10) == 0x140001010);
  SELF_CHECK (pdb.map_section_offset_to_pc (2, 0x10) == 0);
}

/* Test that a procedure keeps the line at its first address when the line block
   that ends there comes after the procedure's own block.  Build two procedures
   with their blocks in that order, and check that the first procedure's line
   ends where the second starts.  */

static void
test_line_order ()
{
  test_module m;
  uint32_t f = m.file (1);
  m.procedure ("first", 1, 0, 16);
  m.procedure ("second", 1, 16, 16);
  m.lines (1, 16, 16, f, { { 0, 20 }, { 4, 21 } });
  m.lines (1, 0, 16, f, { { 0, 10 } });

  std::vector<pdb_module_info> modules (1);
  modules[0].module_name = (char *) "a.obj";
  modules[0].sc_section = 1;
  modules[0].sc_size = 32;
  std::vector<gdb::byte_vector> streams { m.stream (&modules[0]) };

  with_modules (modules, streams, [] (pdb_per_objfile *)
    {
      symtab_and_line first = find_sal_for_pc (0x140001000, 0);
      SELF_CHECK (first.line == 10 && first.pc == 0x140001000
		  && first.end == 0x140001010);
      symtab_and_line second = find_sal_for_pc (0x140001010, 0);
      SELF_CHECK (second.line == 20 && second.pc == 0x140001010
		  && second.end == 0x140001014);
    });
}

/* Test a module with line blocks but no file checksum table.  Check that its
   rows keep their lines, under an unknown file.  */

static void
test_no_checksums ()
{
  test_module m;
  m.procedure ("f", 1, 0, 16);
  m.lines (1, 0, 16, 0, { { 0, 10 } });

  std::vector<pdb_module_info> modules (1);
  modules[0].module_name = (char *) "a.obj";
  modules[0].sc_section = 1;
  modules[0].sc_size = 16;
  std::vector<gdb::byte_vector> streams { m.stream (&modules[0]) };

  with_modules (modules, streams, [] (pdb_per_objfile *)
    {
      SELF_CHECK (line_at (0x140001000)
		  == std::make_pair (10, std::string ("<unknown>")));
    });
}

/* Test a line block that passes the end of its 0x1000-byte section.  Check that
   its rows past the end, and its end marker there, are dropped, and that the
   rows inside keep their lines.  */

static void
test_line_section_bounds ()
{
  test_module m;
  uint32_t f = m.file (1);
  m.lines (1, 0xff0, 0x20, f, { { 0, 10 }, { 0x18, 11 } });

  std::vector<pdb_module_info> modules (1);
  modules[0].module_name = (char *) "a.obj";
  modules[0].sc_section = 1;
  modules[0].sc_size = 0x1000;
  std::vector<gdb::byte_vector> streams { m.stream (&modules[0]) };

  with_modules (modules, streams, [] (pdb_per_objfile *pdb)
    {
      SELF_CHECK (line_at (0x140001ff0)
		  == std::make_pair (10, std::string ("a.cpp")));
      for (compunit_symtab &cu : pdb->objfile->compunits ())
	for (symtab *s : cu.filetabs ())
	  if (const linetable *lt = s->linetable ())
	    for (int i = 0; i < lt->nitems; i++)
	      SELF_CHECK (lt->item[i].pc (pdb->objfile) < 0x140002000);
    });
}

/* Test line blocks whose file cannot be named.  Build one whose checksum entry
   names a string past the /names data, and one whose file id lies past the
   checksum table.  Check that each keeps its lines, under an unknown file, and
   costs no other block its file.  */

static void
test_bad_file ()
{
  test_module m;
  uint32_t good = m.file (1);
  uint32_t bad_name = m.file (1000);
  m.lines (1, 0, 16, good, { { 0, 10 } });
  m.lines (1, 0x10, 16, bad_name, { { 0, 20 } });
  m.lines (1, 0x20, 16, 0xffff0000, { { 0, 30 } });

  std::vector<pdb_module_info> modules (1);
  modules[0].module_name = (char *) "a.obj";
  modules[0].sc_section = 1;
  modules[0].sc_size = 0x30;
  std::vector<gdb::byte_vector> streams { m.stream (&modules[0]) };

  with_modules (modules, streams, [] (pdb_per_objfile *pdb)
    {
      SELF_CHECK (pdb->modules[0].cu != nullptr);
      SELF_CHECK (line_at (0x140001000)
		  == std::make_pair (10, std::string ("a.cpp")));
      SELF_CHECK (line_at (0x140001010)
		  == std::make_pair (20, std::string ("<unknown>")));
      SELF_CHECK (line_at (0x140001020)
		  == std::make_pair (30, std::string ("<unknown>")));
    });
}

/* Test a module compiled with a section per function (/Gy), which has a code
   contribution per function while its header names only the first.  Check that
   a PC in any of them finds the module, whose line table has its line, and that
   a PC in none of them finds no module.  */

static void
test_code_contribs ()
{
  test_module m;
  uint32_t f = m.file (1);
  m.lines (1, 0, 16, f, { { 0, 10 } });
  m.lines (1, 0x200, 16, f, { { 0, 40 } });

  std::vector<pdb_module_info> modules (1);
  modules[0].module_name = (char *) "a.obj";
  modules[0].sc_section = 1;
  modules[0].sc_size = 0x10;
  std::vector<gdb::byte_vector> streams { m.stream (&modules[0]) };

  auto setup = [] (pdb_per_objfile *pdb)
    {
      pdb->code_contribs = { { 1, 0, 0, 0x10 }, { 1, 0, 0x200, 0x10 } };
    };
  test_read_modules (modules, streams, test_tpi (), setup,
		     [] (pdb_per_objfile *pdb)
    {
      SELF_CHECK (pdb_module_at_pc (pdb, 0x140001000) == &pdb->modules[0]);
      SELF_CHECK (pdb_module_at_pc (pdb, 0x140001208) == &pdb->modules[0]);
      SELF_CHECK (pdb_module_at_pc (pdb, 0x140001100) == nullptr);

      bool found = false;
      for (symtab *s : pdb->modules[0].cu->filetabs ())
	if (const linetable *lt = s->linetable ())
	  for (int i = 0; i < lt->nitems; i++)
	    if (lt->item[i].pc (pdb->objfile) == 0x140001200)
	      found = lt->item[i].line == 40;
      SELF_CHECK (found);
    });
}

/* Return the function named NAME among the global and static symbols of
   OBJFILE's compunits, or null.  */

static symbol *
function_named (objfile *objfile, const char *name)
{
  for (compunit_symtab &cust : objfile->compunits ())
    for (const block *b : std::initializer_list<const block *>
	   { cust.blockvector ()->global_block (),
	     cust.blockvector ()->static_block () })
      for (symbol *sym : block_iterator_range (b))
	if (sym->loc_class () == LOC_BLOCK
	    && strcmp (sym->search_name (), name) == 0)
	  return sym;
  return nullptr;
}

/* Test that a procedure whose code runs past the end of its section is
   not read, so that it does not claim the addresses of other code.  Build
   one procedure longer than the section, one that starts inside it and
   ends past it, and one after them.  Check that only the last is read.  */

static void
test_procedure_range ()
{
  test_module m;
  m.procedure ("huge", 1, 0x10, 0xffffffff);
  m.procedure ("past", 1, 0xff8, 0x10);
  m.procedure ("fine", 1, 0x100, 0x10);

  std::vector<pdb_module_info> modules (1);
  modules[0].module_name = (char *) "a.obj";
  modules[0].sc_section = 1;
  modules[0].sc_size = 0x1000;
  std::vector<gdb::byte_vector> streams { m.stream (&modules[0]) };

  with_modules (modules, streams, [] (pdb_per_objfile *pdb)
    {
      SELF_CHECK (function_named (pdb->objfile, "fine()") != nullptr);
      SELF_CHECK (function_named (pdb->objfile, "huge()") == nullptr);
      SELF_CHECK (function_named (pdb->objfile, "past()") == nullptr);
      SELF_CHECK (find_symbol_for_pc (0x140001020) == nullptr);
      SELF_CHECK (find_symbol_for_pc (0x140001ffc) == nullptr);
    });
}

/* Test that a procedure takes its linkage name from its own public, not
   from that of another function.  Identical code folding leaves several
   functions at one address, each with its own decorated public there.  */

static void
test_folded_publics ()
{
  test_module m;
  m.procedure ("alpha", 1, 0, 0x10);
  m.procedure ("beta", 1, 0, 0x10);

  std::vector<pdb_module_info> modules (1);
  modules[0].module_name = (char *) "a.obj";
  modules[0].sc_section = 1;
  modules[0].sc_size = 0x10;
  std::vector<gdb::byte_vector> streams { m.stream (&modules[0]) };

  auto setup = [] (pdb_per_objfile *pdb)
    {
      pdb_publics_at &pubs = pdb->pub_mangled_names[0x140001000];
      pubs.add ("?alpha@@YAXXZ");
      pubs.add ("?beta@@YAXXZ");
    };
  test_read_modules (modules, streams, test_tpi (), setup,
		     [] (pdb_per_objfile *pdb)
    {
      symbol *alpha = function_named (pdb->objfile, "alpha()");
      symbol *beta = function_named (pdb->objfile, "beta()");
      SELF_CHECK (alpha != nullptr
		  && strcmp (alpha->linkage_name (), "?alpha@@YAXXZ") == 0);
      SELF_CHECK (beta != nullptr
		  && strcmp (beta->linkage_name (), "?beta@@YAXXZ") == 0);
    });
}

/* Test that two functions of one signature that do not return
   (CV_PFLAG_NEVER) share the marked copy of their type, and that a
   function of the signature that returns does not have it.  */

static void
test_no_return ()
{
  test_tpi tpi;
  uint32_t sig = tpi.procedure (3, tpi.arglist ({}), 0);

  test_module m;
  /* CV_PFLAG_NEVER.  */
  const uint8_t never = 0x08;
  m.procedure ("quit1", 1, 0, 0x10, sig, never);
  m.procedure ("quit2", 1, 0x10, 0x10, sig, never);
  m.procedure ("back", 1, 0x20, 0x10, sig);

  std::vector<pdb_module_info> modules (1);
  modules[0].module_name = (char *) "a.obj";
  modules[0].sc_section = 1;
  modules[0].sc_size = 0x30;
  std::vector<gdb::byte_vector> streams { m.stream (&modules[0]) };

  test_read_modules (modules, streams, tpi, nullptr,
		     [] (pdb_per_objfile *pdb)
    {
      symbol *quit1 = function_named (pdb->objfile, "quit1()");
      symbol *quit2 = function_named (pdb->objfile, "quit2()");
      symbol *back = function_named (pdb->objfile, "back()");
      SELF_CHECK (quit1 != nullptr && quit2 != nullptr && back != nullptr);
      SELF_CHECK (TYPE_NO_RETURN (quit1->type ())
		  && quit1->type () == quit2->type ());
      SELF_CHECK (!TYPE_NO_RETURN (back->type ())
		  && back->type () != quit1->type ());
    });
}

/* Test that a module whose symbol stream is shorter than its 4-byte
   signature has no symbols to read.  */

static void
test_short_symbols ()
{
  std::vector<pdb_module_info> modules (1);
  modules[0].module_name = (char *) "a.obj";
  modules[0].sc_section = 1;
  modules[0].sc_size = 0x10;
  modules[0].sym_byte_size = 2;
  std::vector<gdb::byte_vector> streams { gdb::byte_vector (2, 0) };

  test_read_modules (modules, streams, test_tpi (), nullptr,
		     [] (pdb_per_objfile *pdb)
    {
      SELF_CHECK (pdb->modules[0].cu == nullptr);
    });
}

/* Test that a malformed TPI header costs the types, not the PDB.  Build a
   header whose TypeIndexBegin is other than 0x1000.  Check that it leaves
   no type record, that the built-in types still resolve and that a
   record's index gives the unsupported type.  */

static void
test_bad_tpi ()
{
  test_tpi tpi;
  uint32_t proc = tpi.procedure (0x74, tpi.arglist ({}), 0);

  for (bool bad : { false, true })
    {
      gdb::byte_vector stream = tpi.stream ();
      /* Write a bad TypeIndexBegin, 8 bytes into the header.  */
      if (bad)
	put_u32 (stream, 8, 0x2000);

      test_pdb file;
      file.streams.resize (5);
      file.streams[PDB_STREAM_TPI] = stream;
      file.build ();
      test_module_reader r (file, 1);
      if (r.pdb == nullptr)
	return;

      pdb_read_tpi_or_none (r.pdb);
      SELF_CHECK (r.pdb->tpi.type_cache != nullptr);
      SELF_CHECK ((r.pdb->tpi.type_idx_end == 0) == bad);
      SELF_CHECK (pdb_tpi_resolve_type (r.pdb, 0x74).code () == TYPE_CODE_INT);
      type_code code = pdb_tpi_resolve_type (r.pdb, proc).code ();
      SELF_CHECK (code == (bad ? TYPE_CODE_ERROR : TYPE_CODE_FUNC));
    }
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
reading and symtab expansion.  A value of 2 also traces symbol lookups\n\
and source path conversion."),
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
  ::selftests::register_test ("pdb-named-stream-map",
			      pdb::selftests::test_named_stream_map);
  ::selftests::register_test ("pdb-names-stream",
			      pdb::selftests::test_names_stream);
  ::selftests::register_test ("pdb-dbi-substreams",
			      pdb::selftests::test_dbi_substreams);
  ::selftests::register_test ("pdb-module-bounds",
			      pdb::selftests::test_module_bounds);
  ::selftests::register_test ("pdb-bad-module",
			      pdb::selftests::test_bad_module);
  ::selftests::register_test ("pdb-line-order",
			      pdb::selftests::test_line_order);
  ::selftests::register_test ("pdb-no-checksums",
			      pdb::selftests::test_no_checksums);
  ::selftests::register_test ("pdb-line-section-bounds",
			      pdb::selftests::test_line_section_bounds);
  ::selftests::register_test ("pdb-bad-file",
			      pdb::selftests::test_bad_file);
  ::selftests::register_test ("pdb-code-contribs",
			      pdb::selftests::test_code_contribs);
  ::selftests::register_test ("pdb-procedure-range",
			      pdb::selftests::test_procedure_range);
  ::selftests::register_test ("pdb-folded-publics",
			      pdb::selftests::test_folded_publics);
  ::selftests::register_test ("pdb-no-return",
			      pdb::selftests::test_no_return);
  ::selftests::register_test ("pdb-short-symbols",
			      pdb::selftests::test_short_symbols);
  ::selftests::register_test ("pdb-bad-tpi",
			      pdb::selftests::test_bad_tpi);
#endif
}
