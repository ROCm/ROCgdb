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
   along with this program.  If not, see <http://www.gnu.org/licenses/>

   This file implements reading of Microsoft PDB debug information format.
   PDB (Program Database) is the native debug format for Microsoft compilers.
   PDB is a multi-stream file container.  All the data is stored in various
   streams while the streams are composed of blocks which don't have to be
   consecutive.  References:
   - microsoft-pdb: https://github.com/microsoft/microsoft-pdb.
   - LLVM docs: https://llvm.org/docs/PDB/.
   - DWARF2 read code: gdb/dwarf2/read.h.  */

#include "gdbtypes.h"
#include "objfiles.h"
#include "buildsym.h"
#include "complaints.h"
#include "source.h"
#include "block.h"
#include "filenames.h"
#include "pdb/pdb.h"
#include "gdbsupport/rsp-low.h"
#include "cli/cli-cmds.h"

#include <string.h>
#include <vector>

/* TODO: Remove once lazy load is implemented.  */
#define PDB_LAZY_INDEX_DISABLED true

#include "gdbsupport/common-utils.h"

/* PDB read debug statements.  */
unsigned int pdb_read_debug = 0;
/* PDB quick functions debug statements.  */
unsigned int pdb_qf_debug = 0;
/* PDB path conversion debug statements.  */
unsigned int pdb_convert_debug = 0;

/* Quick functions debug statements.  */
#define pdb_qf_printf(fmt, ...) \
  debug_prefixed_printf_cond (pdb_qf_debug >= 1, "pdb qf", fmt, \
			      ##__VA_ARGS__)

/* Path conversion debug statements.  */
#define pdb_cvt_printf(fmt, ...) \
  debug_prefixed_printf_cond (pdb_convert_debug >= 1, "pdb cvt", fmt, \
			      ##__VA_ARGS__)

/* One PDB loaded per object file.  */
static const registry<objfile>::key<struct pdb_per_objfile>
  pdb_objfile_data_key;

/* Callback-specific struct for pdb_add_symtab_linetable.  During C13 line info
   parsing we pass line info onto different callbacks.  This for the callback
   that builds the linetable symbols.  */
struct pdb_record_line_cb
{
  struct pdb_line_block_info info;
  struct pdb_per_objfile *pdb;
  struct buildsym_compunit *cu;
};

/* Convert Unix-style paths to Windows paths as GDB has problems with MSVC
   compiled PDBs that may contain paths like /home/rocm/... when built under
   MSYS2. This conversion is used before storing the file paths to lineinfo.
   When path is converted we check for the presence of the converted path and if
   it doesn't exist we return the original path. Path issues to solve:
   1. Under MSYS2, the linker prepends the work dir to the file path. E.g.
      /home/user/test/C:\Program Files\...\include/__msvc_ostream.hpp*
      We strip the work dir: E.g. /<WHATHEVER>/<DRIVE>:\ -> <DRIVE>:\
   2. The pure unix style path - e.g. /home/user/test/file.cpp
      Here we need to find the MSYS2 Windows root and prepend it to the path.
      E.g. C:/msys2/home/user/test/file.cpp.
      Finding the MSYS2 is not deterministic; relying on configure time set
      gdbdir seems wrong. Relying on gdb executable dir has limitations as well.
      For example:
	C:/msys2/mingw64/WHATHEVER/gdb -> Root is C:/msys2/mingw64
	C:/msys2/WHATHEVER/gdb -> Root is C:/msys2
      But we can't easily find the root in e.g. C:/mypath/share/gdb. Likely is
      C:/mypath but finding it programmatically seems impossible. So we require
      MSYS2_ROOT env. var. to be set in case a pure unix path is found.
      If path under MSYS2_ROOT or MSYS2_ROOT is not set we will try some well
      known paths:
       a) E.g. <DRIVE>/WHATHEVER - search for <DRIVE>:/WHATHEVER
       b) Assume MSYS2_ROOT is <DRIVE>:/msys2/mingw6/ or <DRIVE>:/msys2  */
static std::string
pdb_convert_path (const char *path)
{
  if (path == NULL || path[0] == '\0')
    return std::string ();

  /* Non-Unix paths are returned as-is.
     TODO: do a better check (?) but what to do if the checks fails ?  */
  if (path[0] != '/')
    return std::string (path);

  pdb_cvt_printf ("converting path: '%s'", path);

  /* Case 1: Embedded Windows path in a Unix prefix.
     E.g. /home/user/test/C:\Program Files\...\include/file.c
     Search for ':' to find a drive letter pattern /X:\ or /X:/ and strip
     everything before the drive letter.  */
  std::string spath (path);
  size_t pos = spath.find (':');
  if (pos != std::string::npos && pos >= 2)
   {
      if (spath[pos - 2] == '/' && isalpha ((unsigned char) spath[pos - 1])
	  && pos + 1 < spath.size ()
	  && (spath[pos + 1] == '\\' || spath[pos + 1] == '/'))
	{
	  std::string result = spath.substr (pos - 1);
	  if (access (result.c_str (), F_OK) == 0)
	    {
	      pdb_cvt_printf ("Converted: '%s' -> '%s'", path, result.c_str ());
	      return result;
	    }
	}
	/* If file is not found return the original string.
	   TODO: Continuing with other schemes just doesn't seem to make
	   sense if we found ":".  ?  */
	if (pos != std::string::npos)
	  {
	    pdb_warning ("converted path not found");
	    return spath;
	  }
    }

  /* Case 2: Unix-style path.  */

  /* First, try the MSYS2_ROOT environment variable.  */
  const char *env_root = getenv ("MSYS2_ROOT");
  if (env_root != NULL && env_root[0] != '\0')
    {
      std::string result = std::string (env_root) + path;
      /* Not clear if we need to replace anything but maybe root and path
	 have different dir. separators so make them all the same.  */
      std::replace (result.begin (), result.end (), '\\', '/');
      if (access (result.c_str (), F_OK) == 0)
      {
	pdb_cvt_printf ("MSYS2_ROOT convert: '%s' -> '%s'", path, result.c_str ());
	return result;
      }
    }

  /* Fallback (a): Drive-letter mount /c/foo -> C:/foo.  */
  if (isalpha ((unsigned char) path[1]) && (path[2] == '/' || path[2] == '\0'))
    {
      std::string result;
      result += (char) toupper ((unsigned char) path[1]);
      result += ':';
      result += (path[2] == '/') ? &path[2] : "/";
      if (access (result.c_str (), F_OK) == 0) {
	 pdb_cvt_printf ("fallback (a) drive mount: '%s' -> '%s'", path,
			 result.c_str ());
	return result;
      }
    }

  /* Fallback (b): Try well-known MSYS2 installation roots on all
     drive letters.  */
  static const char *known_suffixes[] = {
    ":/msys64/mingw64",
    ":/msys64",
    ":/msys2/mingw64",
    ":/msys2",
    NULL
  };

  for (char drive = 'A'; drive <= 'Z'; drive++)
    {
      for (const char **suffix = known_suffixes; *suffix != NULL; suffix++)
	{
	  std::string full_path = std::string (1, drive) + *suffix + path;
	  if (access (full_path.c_str (), F_OK) == 0)
	    {
	      pdb_cvt_printf ("fallback (b) known root: '%s' -> '%s'", path,
			      full_path.c_str ());
	      return full_path;
	    }
	}
    }

  /* Nothing resolved; return the original path unchanged.  */
  pdb_warning ("converted path not found");
  return std::string (path);
}

/* local function prototypes.  */

static struct pdb_per_objfile *get_pdb_per_objfile (struct objfile *objfile);
static bool pdb_read_msf_header (struct pdb_per_objfile *pdb);
static bool pdb_read_stream_directory (struct pdb_per_objfile *pdb);
static bool pdb_read_dbi_stream (struct pdb_per_objfile *pdb);
static bool pdb_read_file_info_substream (struct pdb_per_objfile *pdb);
static bool pdb_read_sections_from_bfd (struct pdb_per_objfile *pdb);
static void pdb_expand_all_modules (struct pdb_per_objfile *pdb);
static const char *pdb_string_table_get (gdb_byte *string_table,
					 uint32_t table_size,
					 uint32_t offset);
static const char *pdb_get_filename_from_file_id (struct pdb_per_objfile *pdb,
						  struct pdb_module_info *mod,
						  uint32_t file_id);
static struct pdb_per_objfile *pdb_read_pdb_file (struct objfile *objfile);
static void pdb_add_symtab_linetable (void *cb_data);

/* Get the pdb_per_objfile associated with OBJFILE, creating and registering
   a new one if none exists yet.  */
static struct pdb_per_objfile *
get_pdb_per_objfile (struct objfile *objfile)
{
  struct pdb_per_objfile *pdb = pdb_objfile_data_key.get (objfile);
  if (pdb == nullptr)
    {
      pdb = new pdb_per_objfile ();
      pdb->objfile = objfile;
      pdb_objfile_data_key.set (objfile, pdb);
    }
  return pdb;
}

/* Read SIZE bytes from the PDB file at OFFSET into BUF.  */
static bool
pdb_read_file_bytes (struct pdb_per_objfile *pdb, uint64_t offset,
		     gdb_byte *buf, size_t size)
{
  if (pdb->pdb_file_path == NULL)
    return false;

  FILE *f = fopen (pdb->pdb_file_path, "rb");
  if (f == NULL)
    return false;

  if (fseek (f, (long) offset, SEEK_SET) != 0)
    {
      fclose (f);
      return false;
    }

  size_t nr = fread (buf, 1, size, f);
  fclose (f);
  return nr == size;
}

/* Read MSF SuperBlock header.
   https://llvm.org/docs/PDB/MsfFile.html.  */
static bool
pdb_read_msf_header (struct pdb_per_objfile *pdb)
{
  /* Read the SuperBlock from the file and get the block size
     information in order to be able to read streams.  */
  gdb_byte header[MSF_HEADER_SIZE];
  if (!pdb_read_file_bytes (pdb, 0, header, sizeof (header)))
    {
      pdb_error ("Failed to read MSF SuperBlock");
      return false;
    }

  pdb->block_size = UINT32_CAST (header + MSF_BLOCK_SIZE_OFFS);
  pdb->num_blocks = UINT32_CAST (header + MSF_NUM_BLOCKS_OFFS);
  pdb->num_dir_bytes = UINT32_CAST (header + MSF_NUM_DIRECTORY_BYTES_OFFS);
  pdb->block_map_addr = UINT32_CAST (header + MSF_BLOCK_MAP_ADDR_OFFS);

  pdb_dbg_printf ("SuperBlock: block Size=%u, No.blocks=%u, "
		  "Num Dir Bytes=%u, Block Map Addr=%u",
		  (unsigned) pdb->block_size,
		  (unsigned) pdb->num_blocks,
		  (unsigned) pdb->num_dir_bytes,
		  (unsigned) pdb->block_map_addr);

  /* Validate block size.
     TODO: What is the correct validation ?  */
  if (pdb->block_size < 256 || pdb->block_size > 8192 ||
      (pdb->block_size & (pdb->block_size - 1)) != 0)
    {
      pdb_error ("Invalid PDB block size: %u", pdb->block_size);
      return false;
    }

  /* Validate the file is large enough for the number of blocks specified.  */
  uint64_t expected_file_size = pdb->num_blocks * pdb->block_size;
  if (expected_file_size > pdb->msf_size)
    {
      pdb_error ("PDB file too small: expected at least %llu bytes for %u  "
		 "blocks of %u bytes each, but file is only %lu bytes",
		 (unsigned long long) expected_file_size,
		 pdb->num_blocks, pdb->block_size,
		 (unsigned long) pdb->msf_size);
      return false;
    }

  return true;
}

/* Read stream directory in order to get the blocks for each stream.
   See pdb.h and https://llvm.org/docs/PDB/MsfFile.html.  */
static bool
pdb_read_stream_directory (struct pdb_per_objfile *pdb)
{
  /* The stream directory specifies the number of streams, the size of each
     stream and the blocks that belong to each stream.  To access the directory
     we first need to assemble it as it might be split across multiple blocks.
     The block map in the MSF SuperBlock header is an array of uint32_t block
     numbers that belong to the directory.  We need to read these blocks to
     assemble the complete directory.  We read each block by simply reading
     uint32_t values, one after another.  The number of directory bytes is in
     the MSF header.  Layout:
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

  /* Buffer containing the complete stream directory from MSF blocks.  */
  std::unique_ptr<gdb_byte[]> dir(new gdb_byte[pdb->num_dir_bytes]);

  uint32_t num_dir_blocks
    = (pdb->num_dir_bytes + pdb->block_size - 1) / pdb->block_size;

  uint32_t entries_in_block = pdb->block_size / UINT32_SIZE;

  /* Read the block map one block at a time from the file.  The block map starts
     at block_map_addr and may span multiple blocks, with each block holding
     entries_in_block uint32_t entries.  */
  std::unique_ptr<gdb_byte[]> block_map_buf(new gdb_byte[pdb->block_size]);
  uint32_t block_map_idx = UINT32_MAX;  /* Force first read.  */

  for (uint32_t i = 0; i < num_dir_blocks; i++)
    {
      uint32_t cur_map_idx = i / entries_in_block;
      uint32_t entry_in_block = i % entries_in_block;

      /* Read a new block map block when we cross a block boundary.  */
      if (cur_map_idx != block_map_idx)
	{
	  block_map_idx = cur_map_idx;
	  uint32_t map_block = pdb->block_map_addr + block_map_idx;

	  /* Verify the block map block is within bounds.  */
	  if (map_block >= pdb->num_blocks
	    || (uint64_t) map_block * pdb->block_size >= pdb->msf_size)
	    {
	      pdb_error ("Block map block index out of range: block %u, max %u",
			 map_block, pdb->num_blocks);
	      return false;
	    }

	  if (!pdb_read_file_bytes (pdb, map_block * pdb->block_size,
				    block_map_buf.get(), pdb->block_size))
	    {
	      printf_unfiltered ("Failed to read block map block %u",
				 map_block);
	      return false;
	    }
	}

      uint32_t *block_map = UINT32_PTR (block_map_buf.get());
      uint32_t block_idx = block_map[entry_in_block];

      /* Read directory block from the file.  Last block may be partial.  */
      uint32_t sz = std::min (pdb->block_size,
			      pdb->num_dir_bytes - i * pdb->block_size);
      if (!pdb_read_file_bytes (pdb, block_idx * pdb->block_size,
				dir.get() + i * pdb->block_size, sz))
	{
	  pdb_error ("Failed to read directory block %u", block_idx);
	  return false;
	}
    }

  /* Finally we got the stream directory.  Parse it to fill the stream_sizes
     and stream_blocks in pdb_per_objfile.  */
  pdb->num_streams = UINT32_CAST (dir.get());
  pdb->stream_sizes = (uint32_t *)
    obstack_alloc (&pdb->objfile->objfile_obstack,
		   pdb->num_streams * UINT32_SIZE);
  memcpy (pdb->stream_sizes, dir.get() + 4, pdb->num_streams * UINT32_SIZE);

  pdb->stream_blocks = (uint32_t **)
    obstack_alloc (&pdb->objfile->objfile_obstack,
		   pdb->num_streams * UINT32_PTR_SIZE);

  /* Stream data is zeroed so that the stream data presence check works.  */
  pdb->stream_data = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack,
				     pdb->num_streams, gdb_byte *);

  /* TODO: defines  */
  uint32_t offset = 4 + pdb->num_streams * 4;
  for (uint32_t i = 0; i < pdb->num_streams; i++)
    {
      uint32_t stream_size = pdb->stream_sizes[i];
      if (stream_size == PDB_NO_STREAM)
	{
	  pdb->stream_blocks[i] = NULL;
	  continue;
	}

      uint32_t num_blocks = (stream_size + pdb->block_size - 1) /
			     pdb->block_size;
      pdb->stream_blocks[i] = (uint32_t *)
	obstack_alloc (&pdb->objfile->objfile_obstack,
		       num_blocks * UINT32_SIZE);
      memcpy (pdb->stream_blocks[i], dir.get() + offset,
	      num_blocks * UINT32_SIZE);
      offset += num_blocks * UINT32_SIZE;
    }

  /* Print stream count and each stream size at dbg level 1.
     TODO: maybe move to level 2 as there might be many streams.  */
  pdb_dbg_printf_v ("stream directory: %u streams", pdb->num_streams);
  for (uint32_t i = 0; i < pdb->num_streams; i++)
    {
      uint32_t sz = pdb->stream_sizes[i];
      if (sz == PDB_NO_STREAM)
	{
	  pdb_dbg_printf_v ("  stream %u: <unused>", i);
	  continue;
	}
      pdb_dbg_printf_v ("  stream %u: %u bytes", i, sz);

      if (pdb_read_debug >= 3 && pdb->stream_blocks[i] != NULL)
	{
	 /* Print the block stream list for each stream at level 2.
	    TODO: maybe move to level 3 as there might be many streams.
	    but level 2 should print first X streams.  */
	  uint32_t num_blocks = (sz + pdb->block_size - 1) / pdb->block_size;
	  std::string blks;
	  for (uint32_t block_idx = 0; block_idx < num_blocks; block_idx++)
	    {
	      if (block_idx > 0)
		blks += ", ";
	      blks += std::to_string (pdb->stream_blocks[i][block_idx]);
	    }
	     pdb_dbg_printf_t ("    blocks: [%s]", blks.c_str ());
	}
    }
  return true;
}

/* See pdb.h  */
gdb_byte_ptr
pdb_read_stream (struct pdb_per_objfile *pdb, uint32_t stream_idx)
{
  if (stream_idx >= pdb->num_streams)
    return nullptr;

  uint32_t stream_size = pdb->stream_sizes[stream_idx];
  if (stream_size == PDB_NO_STREAM)
    return nullptr;

  uint32_t num_blocks = (stream_size + pdb->block_size - 1) / pdb->block_size;

  auto stream_data = std::make_unique<gdb_byte[]> (stream_size);

  /* Collect all the blocks that belong to this stream and assemble the stream
     data.  The block ids are assigned during stream block parsing.  */
  for (uint32_t i = 0; i < num_blocks; i++)
    {
      uint32_t block_idx = pdb->stream_blocks[stream_idx][i];
      /* For the last block, copy only the remaining stream bytes; for other
	 blocks copy the whole block .  */
      uint32_t copy_size
	= std::min (pdb->block_size, stream_size - i * pdb->block_size);

      if (!pdb_read_file_bytes (pdb, block_idx * pdb->block_size,
				stream_data.get () + i * pdb->block_size,
				copy_size))
	{
	  pdb_error ("Failed to read stream/block %u/%u (file block %u)",
		     stream_idx, i, block_idx);
	  return nullptr;
	}
    }
  return stream_data;
}

/* Read the /names stream and fill the global string table.
   This is not per module table so it's not worth expanding the names
   lazily. E.g. if line info is needed for e.g. a break command in just about
   any file, this function will have to be called.  */
static bool
pdb_read_names_stream (struct pdb_per_objfile *pdb)
{
  pdb->string_table = NULL;
  pdb->string_table_size = 0;

  auto pdb_stream_buf = pdb_read_stream (pdb, PDB_STREAM_PDB);
  if (pdb_stream_buf == nullptr)
    {
      pdb_error ("Failed to read PDB info stream (stream 1)");
      return false;
    }

  gdb_byte *pdb_stream = pdb_stream_buf.get ();

  uint32_t pdb_stream_size = pdb->stream_sizes[PDB_STREAM_PDB];

  pdb_dbg_printf ("PDB info stream: size=%u",pdb_stream_size);

  if (pdb_stream_size < INFO_STREAM_MIN_SIZE)
    {
      pdb_error ("PDB info stream too small (%u bytes)", pdb_stream_size);
      return false;
    }

  /* Decode the info stream header to verify offsets */
  uint32_t info_version = UINT32_CAST(pdb_stream + 0);
  uint32_t info_signature = UINT32_CAST(pdb_stream + 4);
  uint32_t info_age = UINT32_CAST(pdb_stream + 8);

  pdb_dbg_printf ("PDB Info Stream: version:%u, signature:0x%08X, age=%u",
		  info_version, info_signature, info_age);

  /* Get to Named Stream map. */
  gdb_byte *map = pdb_stream + INFO_STREAM_HASH_MAP_OFFS;
  /* Everything in the stream except for the header belongs to the map  */
  uint32_t map_size = pdb_stream_size - INFO_STREAM_HASH_MAP_OFFS;

  gdb_byte *map_end = map + map_size;

  /* Check that there is at least one entry. For this, it seems that we need
     4 bytes each for any of these: StringBuffer Size, HashTable Size/Capacity,
     PresentWordCount and DeletedWordCount: 20 bytes */
  if (map_size < NAMES_MAP_STRBUF_SIZE_OFFS + 5 * NAMES_MAP_UINT32_SIZE)
   {
     /* TODO - check what happens if there is no map. Will that result in the
	map_size of zero ? In that case we could check that first and throw a
	warning (e.g. /names not available) while this check actually checks
	for PDB corruption.  */
     pdb_warning ("PDB info stream too small for named stream map header");
     return false;
   }

  /* Get pointer to StringBuffer and it's size.  */
  gdb_byte *strbuf = map + NAMES_MAP_STRBUF_OFFS;
  uint32_t strbuf_size = UINT32_CAST (map + NAMES_MAP_STRBUF_SIZE_OFFS);

  /* Print String Buffer content.  */
  pdb_dbg_printf_v ("  StringBuffer contents:");
  for (uint32_t i = 0; i < strbuf_size; )
    {
      auto name = CSTR(strbuf + i);
      pdb_dbg_printf_v ("    [%u] '%s'", i, name);
      i += strlen (name) + 1;
      if (i > strbuf_size)
	break;
    }

  /* Get pointer to Hash Table.  */
  gdb_byte* ht = strbuf + strbuf_size;

  uint32_t hash_size = UINT32_CAST (ht + 0 /* TODO define  */);
  uint32_t hash_capacity = UINT32_CAST (ht + NAMES_MAP_UINT32_SIZE);
  uint32_t present_words = UINT32_CAST (ht + HASH_TABLE_PRESENT_CNT_OFFS);
  uint32_t *present_bits = UINT32_PTR (ht + HASH_TABLE_PRESENT_WORD_OFFS);

  pdb_dbg_printf ("Hash table: size=%u, capacity=%u, present_words=%u",
		  hash_size, hash_capacity, present_words);

   /* Check that hash table header, present bits and hash entries fit
      within the map.  We still don't have deleted words. */
  gdb_byte *temp_ptr = (ht + HASH_TABLE_PRESENT_WORD_OFFS
			+ present_words * HASH_TABLE_BYTES_PER_WORD
			+ HASH_TABLE_BYTES_PER_WORD);
  if (temp_ptr > map_end)
    {
      pdb_error ("PDB info stream too small for hash table header "
		 "and present bit vector");
      return false;
    }

  gdb_byte *deleted_cnt_ptr = ht + HASH_TABLE_PRESENT_WORD_OFFS
			      + present_words * HASH_TABLE_BYTES_PER_WORD;
  uint32_t deleted_words = UINT32_CAST(deleted_cnt_ptr);

  /* Finally get to the key/value pairs table.  */
  gdb_byte *table = deleted_cnt_ptr + HASH_TABLE_BYTES_PER_WORD
		    + deleted_words * HASH_TABLE_BYTES_PER_WORD;

  /* Walk entries whose bit is set in present_bits, until /names is found.  */

  uint32_t names_stream = PDB_NO_STREAM;
  for (uint32_t i = 0; i < hash_capacity; i++)
    {

     /* We could do one check before the for loop and verify that the table is
       within the bounds (have hash_capacity entries). However, the table can
       be smaller we observe it goes up to the number of valid entries.  So it
       seems the expectation is that the parser will stop after parsing first
       hash_size valid entries. So, instead of counting the valid entries we
       just verify the table pointer as we move through the entries.  */
      if (table + HASH_TABLE_ENTRY_SIZE > map_end)
	{
	  pdb_warning ("Table size corruped past the %ith entry", i);
	  break;
	}

      /* Check if bit i is set in present_bits.  */
      uint32_t word_idx = i / HASH_TABLE_BIT_WORD_SIZE;
      uint32_t bit_idx = i % HASH_TABLE_BIT_WORD_SIZE;
      if (word_idx >= present_words ||
	  !(present_bits[word_idx] & (1 << bit_idx)))
	continue;

      uint32_t key = UINT32_CAST (table);    /* offset into StringBuffer.  */
      table += HASH_TABLE_KEY_SIZE;
      uint32_t value = UINT32_CAST (table);  /* stream number.  */
      table += HASH_TABLE_VAL_SIZE;

      /* Look up the name.  */
      if (key < strbuf_size)
	{
	  auto name = CSTR(strbuf + key);
	  pdb_dbg_printf_v (" Named stream: '%s' -> stream %u", name, value);
	  if (strcmp (name, "/names") == 0)
	    {
	      pdb_dbg_printf ("  Found /names stream at index %u", value);
	      names_stream = value;
	      break;
	    }
	}
    }

  if (names_stream == PDB_NO_STREAM || names_stream >= pdb->num_streams)
    {
      pdb_warning ("/names stream not found");
      return false;
    }

  /* Read the /names stream and fill the global string table.
     Cache it — string_table points into it for the objfile lifetime.  */

  auto names_buf = pdb_read_stream (pdb, names_stream);
  if (names_buf == nullptr)
    {
      pdb_warning ("No data in /names stream");
      return false;
    }

  gdb_byte *names_data = pdb->stream_data[names_stream] = names_buf.release ();

  uint32_t names_size = pdb->stream_sizes[names_stream];

  if (names_size < NAMES_STREAM_MIN_SIZE)
    {
      pdb_warning ("/names stream too small (%u bytes)", names_size);
      return false;
    }

  uint32_t names_sig = UINT32_CAST (names_data + NAMES_STREAM_SIGNATURE_OFFS);
  uint32_t data_sz = UINT32_CAST (names_data + NAMES_STREAM_DATA_SIZE_OFFS);

  if (names_sig != NAMES_STREAM_SIGNATURE_VAL)
    pdb_error ("unexpected /names signature");

  if (NAMES_STREAM_MIN_SIZE + data_sz > names_size)
    {
      pdb_error ("/names string data extends beyond stream");
      return false;
    }

  pdb->string_table = names_data + NAMES_STREAM_MIN_SIZE;
  pdb->string_table_size = data_sz;

  /* Global string table loaded.  */

  /* Print all strings at debug level 2.  */
  if (pdb_read_debug >= 2)
    {
      gdb_byte *st = pdb->string_table;
      gdb_byte *st_end = st + data_sz;
      for (gdb_byte *s = st; s < st_end; s++)
	{
	  if (*s != '\0' && (s == st || *(s-1) == '\0'))
	    {
	      debug_prefixed_printf_cond (true, "pdb",
					  "  String table [offset=%u]: %s",
					  (uint32_t)(s - st), CSTR(s));
	    }
	}
    }

  return true;
}

/* Read DBI (Debug Info) stream.  Reads headers and Module Info Records in order
   to obtain the sizes and stream numbers of the debug information for each
   module (object file).  The sizes are used to allocate enough space needed to
   hold the debug information for each module.  */
static bool
pdb_read_dbi_stream (struct pdb_per_objfile *pdb)
{
  /* Cache DBI — module names and file_info_names_buffer point into it.  */
  auto dbi_buf = pdb_read_stream (pdb, PDB_STREAM_DBI);
  if (dbi_buf == nullptr)
    {
      pdb_error ("Failed to read DBI stream");
      return false;
    }
  pdb->stream_data[PDB_STREAM_DBI] = dbi_buf.release ();

  /* Parse DBI header.  */

  gdb_byte *dbi_hdr = pdb->stream_data[PDB_STREAM_DBI];
  uint32_t signature = UINT32_CAST (dbi_hdr + DBI_HDR_SIGNATURE_OFFS);

  if (signature != 0xFFFFFFFF)
    {
      pdb_error ("Invalid DBI signature: 0x%x", signature);
      return false;
    }

  /* Read sizes from DBI header.  */

  uint32_t mod_info_size = UINT32_CAST (dbi_hdr + DBI_HDR_MODULE_SIZE_OFFS);
  pdb->gsi_stream = UINT16_CAST (dbi_hdr + DBI_HDR_GSI_STREAM_OFFS);
  pdb->psgsi_stream = UINT16_CAST (dbi_hdr + DBI_HDR_PSGSI_STREAM_OFFS);
  pdb->sym_record_stream = UINT16_CAST (dbi_hdr + DBI_HDR_SYM_RECORD_STREAM_OFFS);

  pdb_dbg_printf ("DBI header: signature=0x%08x module_info_size=%u"
		  " gsi=%u psgsi=%u sym_record_stream=%u",
		   signature, mod_info_size,
		   pdb->gsi_stream, pdb->psgsi_stream,
		   pdb->sym_record_stream);

  /* Module info starts right after the DBI header.  */
  gdb_byte *mod_info = dbi_hdr + DBI_HDR_MODULE_INFO_OFFS;
  gdb_byte *mod_info_end = mod_info + mod_info_size;

  if (mod_info_size == 0)
    {
      /* FIXME - maybe complaint? */
      pdb_warning ("Module info empty, no modules to parse");
      pdb->num_modules = 0;
      pdb->modules = NULL;
      return true;
    }


  /* Allocate modules.  First pass will only count the modules
     for easier allocation in the second pass.  */
  pdb->num_modules = 0;
  gdb_byte *p = mod_info;

  while (p + MODI_FIXED_SIZE < mod_info_end)
    {
      /* Skip 64-byte fixed header to get to module names.  */
      p += MODI_FIXED_SIZE;

      /* Module name (null-terminated).  */
      gdb_byte *nul = (gdb_byte *) memchr (p, 0, mod_info_end - p);
      if (nul == NULL)
	{
	  pdb_error ("Truncated module name in module info");
	  break;
	}
      p = nul + 1;

      /* Object file name (null-terminated).  */
      nul = (gdb_byte *) memchr (p, 0, mod_info_end - p);
      if (nul == NULL)
	{
	  pdb_error ("Truncated obj file name in module info");
	  break;
	}
      p = nul + 1;

      /* Align to 4 bytes.  */
      p = (gdb_byte *) align_up ((uintptr_t)p, 4);

      pdb->num_modules++;

      /* Safety limit.  */
      if (pdb->num_modules > 10000)
	{
	  pdb_error ("Too many modules: %u", pdb->num_modules);
	  break;
	}
    }

  pdb_dbg_printf ("Found %u modules", pdb->num_modules);

  pdb->modules = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack,
				 pdb->num_modules,
				 struct pdb_module_info);

  /* Parse module info records.  */
  p = mod_info;
  for (uint32_t i = 0; i < pdb->num_modules; i++)
    {
      struct pdb_module_info *module = &pdb->modules[i];
      module->module_index = i;

      /* MODI fixed header fields (0x40 bytes per LLVM/MS spec).  */
      module->stream_number = UINT16_CAST (p + MODI_STREAM_NUM_OFFS);
      module->flags         = UINT16_CAST (p + MODI_FLAGS_OFFS);
      module->sym_byte_size = UINT32_CAST (p + MODI_SYM_BYTES_OFFS);
      module->c11_byte_size = UINT32_CAST (p + MODI_C11_BYTES_OFFS);
      module->c13_byte_size = UINT32_CAST (p + MODI_C13_BYTES_OFFS);

      /* Section contribution.  */
      module->sc_section = UINT32_CAST (p + MODI_SC_ISECT_OFFS);
      module->sc_offset  = UINT32_CAST (p + MODI_SC_OFFSET_OFFS);
      module->sc_size    = UINT32_CAST (p + MODI_SC_SIZE_OFFS);

      /* Module name starts at offset 0x40.  */
      gdb_byte *name_ptr = p + MODI_FIXED_SIZE;

      module->module_name = (char *)name_ptr;
      gdb_byte *name_end = (gdb_byte *) memchr (name_ptr, 0,
						mod_info_end - name_ptr);
      if (name_end == NULL)
	{
	  pdb_error ("Module name not null-terminated");
	  return false;
	}
      gdb_byte *next_ptr = name_end + 1;

      /* Read object file name  *
	TODO: Deallocate DBI stream ? Seems it's small so better not.  */
      module->obj_file_name = (char *)next_ptr;
      gdb_byte *obj_file_end = (gdb_byte *) memchr (next_ptr, 0,
						    mod_info_end - next_ptr);
      if (obj_file_end == NULL)
	{
	  pdb_error ("Object file name not null-terminated");
	  return false;
	}
      next_ptr = obj_file_end + 1;

      /* Align to 4-byte boundary for next MODI record.  */
      p = (gdb_byte *) align_up ((uintptr_t)next_ptr, 4);

      pdb_dbg_printf_v ("  Module %u: stream=%u %s"
			" (sym=%u, c11=%u, c13=%u)",
			i, module->stream_number,
			module->module_name ? module->module_name : "<empty>",
			module->sym_byte_size,
			module->c11_byte_size,
			module->c13_byte_size);
    }

  pdb_dbg_printf ("Finished parsing all modules");
  return true;
}

/* Read section addresses from BFD.  */
static bool
pdb_read_sections_from_bfd (struct pdb_per_objfile *pdb)
{
  asection *sect;
  uint32_t num_sections = 0;

  if (pdb->objfile->obfd == NULL)
    {
      /* FIXME - likely complaint? */
      pdb_warning ("No BFD available for section mapping");
      return false;
    }

  /* Count sections.  */
  for (sect = pdb->objfile->obfd->sections; sect != NULL; sect = sect->next)
    num_sections++;

  if (num_sections == 0)
    {
      /* FIXME - likely complaint? */
      pdb_warning ("No sections found in BFD");
      return false;
    }

  pdb->num_sections = num_sections;
  pdb->section_addresses = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack,
					   num_sections, CORE_ADDR);
  uint32_t idx = 0;
  for (sect = pdb->objfile->obfd->sections; sect != NULL; sect = sect->next)
    {
      /* Get virtual memory address of this section.  */
      pdb->section_addresses[idx] = bfd_section_vma (sect);
      idx++;
    }

  return true;
}

/* Map PDB (section, offset) to actual PC address.
   See pdb.h.  */
CORE_ADDR
pdb_map_section_offset_to_pc (struct pdb_per_objfile *pdb,
			      uint16_t section,
			      uint32_t offset)
{
  if (section == 0 || section > pdb->num_sections)
    {
      pdb_complaint ("Invalid section number %u (max %u)",
		     section, pdb->num_sections);
      return 0;
    }

  /* Get section base address (VMA from PE header).  */
  CORE_ADDR section_base = pdb->section_addresses[section - 1];

  /* Apply relocation offset (difference between linked and loaded address).  */
  CORE_ADDR reloc_offset = pdb->objfile->section_offsets[section - 1];

  /* Calculate final address: section VMA + relocation + offset.  */
  CORE_ADDR result = section_base + reloc_offset + offset;

  return result;
}

/* Read a module stream and extract C13 file checksums.  Returns an owned
   buffer containing the stream data; the caller keeps it alive as long as
   the data is needed (for symbol parsing, line walking, etc.).
   File checksums are copied onto the objfile obstack so they survive
   after the buffer is freed.  */
gdb_byte_ptr
pdb_read_module_stream (struct pdb_per_objfile *pdb,
			struct pdb_module_info *module)
{
  if (module->stream_number == 0xFFFF)
    return nullptr;

  auto buf = pdb_read_stream (pdb, module->stream_number);
  if (buf == nullptr)
    return nullptr;

  gdb_byte *stream = buf.get ();
  uint32_t stream_size = pdb->stream_sizes[module->stream_number];

  /* Validate that sym + c11 + c13 sizes fit within the stream.  */
  uint64_t fixed_part_size = (uint64_t) module->sym_byte_size
			     + (uint64_t) module->c11_byte_size;

  if (fixed_part_size > stream_size)
    {
      pdb_error ("Module %u: sym+c11 sizes exceed stream size",
			 module->stream_number);
      return nullptr;
    }

  if (module->c13_byte_size > stream_size - fixed_part_size)
    {
      pdb_error ("Module %u: c13 size exceeds stream size",
			 module->stream_number);
      return nullptr;
    }

  if (module->c13_byte_size == 0)
    return buf;

  gdb_byte *c13_start = stream + module->sym_byte_size +
			module->c11_byte_size;
  uint32_t c13_size = module->c13_byte_size;

  /* Parse C13 subsections.  */
  gdb_byte *p = c13_start;
  gdb_byte *c13_end = c13_start + c13_size;

  /* Skip CV signature if present.  */
  if (p + CV_SIGNATURE_SIZE <= c13_end && UINT32_CAST(p) == CV_SIGNATURE_C13)
    p += CV_SIGNATURE_SIZE;

  /* Parse subsections.  */
  while (p + C13_SUBSECT_HEADER_SIZE <= c13_end)
    {
      uint32_t subsect_type = UINT32_CAST (p);
      uint32_t subsect_size = UINT32_CAST (p + 4);
      p += C13_SUBSECT_HEADER_SIZE;

      if (p + subsect_size > c13_end)
	{
	  pdb_error ("C13 subsection extends beyond stream");
	  break;
	}

      switch (subsect_type)
	{
	case DEBUG_S_LINES:
	  /* Processed on demand by pdb_walk_c13_line_blocks when
	     pdb_build_module expands this module.  */
	  break;

	case DEBUG_S_FILECHKSMS:
	  /* Copy the checksums block onto the objfile obstack so it
	     survives after the module stream is freed.  */
	  {
	    gdb_byte *copy = (gdb_byte *)
	      obstack_alloc (&pdb->objfile->objfile_obstack, subsect_size);
	    memcpy (copy, p, subsect_size);
	    module->file_checksums = copy;
	    module->file_checksums_size = subsect_size;
	  }
	  break;

	default:
	  break;
	}

      p += subsect_size;

      /* Align to 4-byte boundary.  */
      p = (gdb_byte *) align_up ((uintptr_t)p, 4);
    }

  return buf;
}

/* Get string from the global string table (/names stream) at given offset.  */
static const char *
pdb_string_table_get (gdb_byte *string_table,
		      uint32_t table_size,
		      uint32_t offset)
{
  if (string_table == NULL || offset >= table_size)
    return NULL;

  auto str = CSTR (string_table + offset);
  if (memchr (str, '\0', table_size - offset) == NULL)
    return NULL;

  return str;
}

/* Get filename from file_id by looking up in the FILECHKSMS subsection.  */
static const char *
pdb_get_filename_from_file_id (struct pdb_per_objfile *pdb,
			       struct pdb_module_info *module,
			       uint32_t file_id)
{
  if (module->file_checksums == NULL)
    {
      printf_unfiltered ("  file_checksums is NULL");
      return NULL;
    }

  /* Access filename through the global string table from /names stream.  */
  if (pdb->string_table == NULL)
    return NULL;

  /* Parse file checksums to find the entry for this file_id.  */
  gdb_byte *p = module->file_checksums;
  gdb_byte *end = p + module->file_checksums_size;

  /* file_id is a byte offset into the DEBUG_S_FILECHKSMS subsection so it
     cannot exceed the checksums size.  */
  if (file_id >= module->file_checksums_size)
    {
      pdb_error ("file_id %u beyond file checksums size %u",
		 file_id, module->file_checksums_size);
      return NULL;
    }

  /* Move pointer to the start of file checksums. */
  p += file_id;

  /* Check that the file checksum record header fits within the buffer.  */
  if (p + PDB_FILECHKSUM_HDR_SIZE > end)
    {
      pdb_error ("file checksum record truncated at offset %u", file_id);
      return NULL;
    }

  struct CV_FileChecksum *checksum = (struct CV_FileChecksum *)p;

  auto filename = pdb_string_table_get (pdb->string_table,
					  pdb->string_table_size,
					  checksum->file_name_offset);

  /* Return filename, fall back to module name, then "<unknown>".  */
  return filename ? filename
       : module->module_name ? module->module_name
       : "<unknown>";
}

/* Walk all DEBUG_S_LINES subsections in a module's C13 data.  For each file
   block, fills the pdb_line_block_info at the start of CB_DATA and invokes
   CALLBACK(CB_DATA).  MODULE_STREAM is the raw module stream data.  */
void
pdb_walk_c13_line_blocks (struct pdb_per_objfile *pdb,
			  struct pdb_module_info *module,
			  gdb_byte *module_stream,
			  pdb_line_block_fn callback,
			  void *cb_data)
{
  if (module->c13_byte_size == 0)
    {
      pdb_complaint ("no C13 line info in module %s",
		     module->module_name ? module->module_name : "<unknown>");
      return;
    }

  gdb_byte *c13_start = module_stream + module->sym_byte_size
			+ module->c11_byte_size;
  gdb_byte *c13_end = c13_start + module->c13_byte_size;
  gdb_byte *p = c13_start;

  /* Skip CV signature if present.  */
  if (p + CV_SIGNATURE_SIZE <= c13_end && UINT32_CAST (p) == CV_SIGNATURE_C13)
    p += CV_SIGNATURE_SIZE;

  /* See pdb.h for C13 block layout.  */

  /* Do not access past the end of the C13.  */
  while (p + C13_SUBSECT_HEADER_SIZE <= c13_end)
    {
      /* Read subsection header.
	 TODO: Introduce a struct for the subsection header ?.  */
      uint32_t subsect_type = UINT32_CAST (p);
      uint32_t subsect_size = UINT32_CAST (p + 4);
      p += C13_SUBSECT_HEADER_SIZE;

      if (p + subsect_size > c13_end)
	{
	  pdb_error ("C13 subsection type 0x%x size %u overflows stream",
		     subsect_type, subsect_size);
	  break;
	}

      if (subsect_type != DEBUG_S_LINES)
	{
	  p += subsect_size;
	  /* Align to 4-byte boundary.  */
	  p = (gdb_byte *) align_up ((uintptr_t)p, 4);
	  continue;
	}

      gdb_byte *sp = p;
      gdb_byte *sp_end = p + subsect_size;

      /* Ensure there's enough space for the header.  */
      if (sp + sizeof (struct CV_LineSection) > sp_end)
	{
	  pdb_error ("DEBUG_S_LINES subsection too small for header"
		     " (%u bytes)", subsect_size);
	  break;
	}

      struct CV_LineSection *ls = (struct CV_LineSection *)sp;
      sp += sizeof (struct CV_LineSection);

      /* Iterate file blocks within this subsection.  */
      while (sp + sizeof (struct CV_FileBlock) <= sp_end)
	{
	  struct CV_FileBlock *fb = (struct CV_FileBlock *)sp;
	  uint32_t file_id = fb->file_id;
	  uint32_t num_lines = fb->num_lines;
	  uint32_t block_size = fb->block_size;

	  if (block_size < sizeof (struct CV_FileBlock)
	      || sp + block_size > sp_end)
	    break;

	  gdb_byte *line_data = sp + sizeof (struct CV_FileBlock);
	  uint32_t line_entries_size = num_lines * sizeof (struct CV_Line);
	  if (line_data + line_entries_size > sp + block_size)
	    break;

	  /* Resolve filename from file checksums + string table.  */
	  auto fname = pdb_get_filename_from_file_id (pdb, module, file_id);
	  if (fname == nullptr)
	    fname = "<unknown>";

	  /* Fill the base struct at the start of cb_data.  */
	  struct pdb_line_block_info *info
	    = (struct pdb_line_block_info *)cb_data;
	  info->filename = fname;
	  info->line_sect = ls;
	  info->lines = (struct CV_Line *)line_data;
	  info->num_lines = num_lines;

	  pdb_dbg_printf_v ("Line block: file=%s, section=%u, offset=%u, "
			    "size=%u, num_lines=%u", fname, ls->seci,
			    ls->offs, ls->code_sz, num_lines);

	  callback (cb_data);

	  sp += block_size;
	}

	/* Advance past this subsection and align to 4-byte boundary.  */
	p += subsect_size;
	p = (gdb_byte *) align_up ((uintptr_t)p, 4);
    }
}

/* Callback for pdb_walk_c13_line_blocks.
   Record line entries into buildsym_compunit.  */
static void
pdb_add_symtab_linetable (void *cb_data)
{
  struct pdb_record_line_cb *cb = (struct pdb_record_line_cb *)cb_data;
  struct pdb_line_block_info *info = &cb->info;
  struct pdb_per_objfile *pdb = cb->pdb;
  struct buildsym_compunit *cu = cb->cu;

  uint16_t sect = info->line_sect->seci;
  uint32_t sect_offs = info->line_sect->offs;

  /* Convert MSYS2/Unix paths to Windows paths for source display.  */
  std::string path = pdb_convert_path (info->filename);

  pdb_dbg_printf_v ("add linetable: file='%s', lines=%u, section=%u, offs=0x%x",
		    path.c_str(), info->num_lines, sect, sect_offs);

  cu->start_subfile (path.c_str (), path.c_str ());

  struct subfile *current_subfile = cu->get_current_subfile ();

  for (uint32_t i = 0; i < info->num_lines; i++)
    {
      uint32_t offs = info->lines[i].offs;
      uint32_t line_num = info->lines[i].line_start;

      CORE_ADDR pc = pdb_map_section_offset_to_pc (pdb, sect, sect_offs + offs);
      if (pc == 0)
	{
	  pdb_dbg_printf_v ("line %u: skipped (PC=0)", line_num);
	  continue;
	}

      CORE_ADDR unrelocated = pc - pdb->objfile->text_section_offset ();
      unrelocated_addr unrelocated_pc{unrelocated};
      /* Always mark PDB line entries as statements.
	 FIXME: Check how reliably this flag is set by compiler and verify
	 if GDB regularly skips non-statement entries.  */
      linetable_entry_flags flags = LEF_IS_STMT;

      cu->record_line (current_subfile, line_num, unrelocated_pc, flags);

      pdb_dbg_printf_v ("line %u: unrelocated_pc=0x%" PRIx64 " added",
			line_num, unrelocated);
    }

  pdb_dbg_printf_v ("total lines recorded");
}

/* See pdb.h.  */
struct compunit_symtab *
pdb_build_module (struct pdb_per_objfile *pdb,
		  struct pdb_module_info *module)
{
  if (module->expanded)
    return module->cu;

  module->expanded = true;

  /* Read module stream — owned locally, auto-freed at scope exit.  */
  auto stream_buf = pdb_read_module_stream (pdb, module);

  /* Cache file list by reading File Info substream.  */
  pdb_read_module_files (pdb, module);

  /* Skip modules with no code contribution (e.g.  linker-generated,
     resource modules, import libs).  */
  if (module->sc_section == 0)
    return NULL;

  /* Compute module address range based on section contribution from DBI.  */
  CORE_ADDR low_pc = pdb_map_section_offset_to_pc (pdb, module->sc_section,
						   module->sc_offset);
  CORE_ADDR high_pc = low_pc + module->sc_size;
  if (high_pc <= low_pc)
    high_pc = low_pc + 1;

  /* CU name from module_name.  Skip names that are clearly not source
     files (e.g. linker-generated "?" or "* Linker *").  */
  const char *cu_name = module->module_name ? module->module_name : "";

  struct buildsym_compunit *cu
    = new buildsym_compunit (pdb->objfile, cu_name, "", language_c, low_pc);

  /* Record line tables (only if C13 data exists).  */
  if (module->c13_byte_size > 0 && stream_buf != nullptr)
    {
      struct pdb_record_line_cb rcb;
      rcb.pdb = pdb;
      rcb.cu = cu;
      pdb_walk_c13_line_blocks (pdb, module, stream_buf.get (),
				pdb_add_symtab_linetable, &rcb);
      pdb_dbg_printf ("C13 line processing complete.");
    }

  /* Parse CodeView symbol records and collect function ranges.  */
  pdb_range_pair_vec func_ranges;
  if (stream_buf != nullptr)
    pdb_parse_symbols (pdb, module, stream_buf.get (), cu, 0, &func_ranges);

  /* Register function ranges on the static block so the CU-level scope
     covers all functions even when /O2 scatters them to non-contiguous
     addresses (similar to DWARF's DW_AT_ranges).  */
  struct block *static_block
    = cu->end_compunit_symtab_get_static_block (high_pc, false, true);

  std::vector<blockrange> rangevec;
  for (auto &range : func_ranges)
    rangevec.emplace_back (range.first, range.second - 1);
  if (!rangevec.empty ())
    static_block->set_ranges (make_blockranges (pdb->objfile, rangevec));

  module->cu = cu->end_compunit_symtab_from_static_block (static_block, false);
  delete cu;

  /* Print the finalized linetable.  */
  if (module->cu != nullptr)
    {
      for (auto *stab : module->cu->filetabs ())
	{
	  const struct linetable *lt = stab->linetable ();
	  pdb_dbg_printf_v ("  symtab '%s': %d line entries",
			    stab->filename (), lt ? lt->nitems : 0);
	  if (lt != nullptr)
	    {
	      for (int i = 0; i < lt->nitems; i++)
		pdb_dbg_printf_v ("    [%d] line=%d, pc=0x%" PRIx64,
				  i, lt->item[i].line,
				  CORE_ADDR (lt->item[i].unrelocated_pc ()));
	    }
	}
    }

  pdb_dbg_printf_v ("Module '%s' expanded: low_pc=0x%08" PRIx64
		    " high_pc=0x%08" PRIx64
		    " sym size=%u c11 size=%u c13 size=%u",
		    cu_name, low_pc, high_pc, module->sym_byte_size,
		    module->c11_byte_size,  module->c13_byte_size);

  /* stream_buf auto-freed here — file_checksums was already copied to
     the obstack, so nothing references the stream anymore.  */

  return module->cu;
}

/* Expand all modules.  Called during initial PDB load in case lazy loading
   is not requested.  */
static void
pdb_expand_all_modules (struct pdb_per_objfile *pdb)
{
  for (uint32_t i = 0; i < pdb->num_modules; i++)
    pdb_build_module (pdb, &pdb->modules[i]);

  /* Load global variables from the SymRecordStream (S_GDATA32 records
     indexed by GSI).  These don't belong to any module stream and are
     not loaded by pdb_build_module.  */
  if (pdb->sym_record_data != nullptr && pdb->sym_record_size > 0)
    {
      struct buildsym_compunit *cu
	= new buildsym_compunit (pdb->objfile, "<pdb-globals>", "",
				 language_c, 0);
      pdb_load_global_syms (pdb, cu);
      cu->end_compunit_symtab (0);
      delete cu;
    }

  pdb_dbg_printf ("All %u modules expanded", pdb->num_modules);
}

/* Load the PDB file for the given pdb context.  Validates the MSF magic, stores
   the file size in pdb->msf_size, and saves the path.  The file is NOT loaded
   into memory — blocks are read on demand.  */
static bool
pdb_load_pdb_file (struct pdb_per_objfile *pdb)
{
  struct objfile *objfile = pdb->objfile;

  /* Find the PDB file on disk.  */
  std::string pdb_path = pdb_find_pdb_file (objfile);
  if (pdb_path.empty ())
    return false;

  /* Open, validate magic, and get file size.  */
  FILE *pdb_file = fopen (pdb_path.c_str (), "rb");
  if (pdb_file == NULL)
    return false;

  gdb_byte magic[PDB_MSF_MAGIC_SIZE];
  size_t nr = fread (magic, 1, PDB_MSF_MAGIC_SIZE, pdb_file);
  if (nr < PDB_MSF_MAGIC_SIZE
      || memcmp (magic, PDB_MSF_MAGIC, PDB_MSF_MAGIC_SIZE) != 0)
    {
      fclose (pdb_file);
      pdb_error ("Bad MSF signature: %s", pdb_path.c_str ());
      return false;
    }

  fseek (pdb_file, 0, SEEK_END);
  long file_size = ftell (pdb_file);
  fclose (pdb_file);

  if (file_size <= 0)
    {
      pdb_warning ("PDB file is empty: %s", pdb_path.c_str ());
      return false;
    }

  pdb_dbg_printf ("PDB file validated: %s (%lu bytes)",
		  pdb_path.c_str (), (unsigned long) file_size);

  pdb->pdb_file_path = obstack_strdup (&objfile->objfile_obstack,
					pdb_path.c_str ());
  pdb->msf_size = file_size;
  return true;
}

/* Main entry point: PDB parsing starts by reading the MSF SuperBlock, which
   contains basic info about the file and location of the stream directory.
   Each stream spans multiple blocks that are not necessarily contiguous and the
   layout of every single stream is described in the  stream directory.
   Next, we parse the /names stream which contains the global string table.
   Then we continue with the DBI stream which contains the main debug info.
   From the DBI stream we get the list of modules and their associated source
   files, including the size of various debug information for each module.
   The actual debug info for a module is read on demand.  */
static struct pdb_per_objfile *
pdb_read_pdb_file (struct objfile *objfile)
{
  if (objfile->flags & OBJF_READNEVER)
    {
      pdb_dbg_printf ("OBJF_READNEVER set");
      return nullptr;
    }

  gdb_assert (objfile->obfd != nullptr);

  /* Initialize PDB reader — create and register per_objfile.  */
  struct pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);

  /* Validate PDB magic and record file size.  */
  if (!pdb_load_pdb_file (pdb))
    {
      pdb_error ("Failed to load PDB file");
      return nullptr;
    }

  /* Read MSF structure.  */
  if (!pdb_read_msf_header (pdb))
    {
      pdb_error ("Failed to read MSF header");
      return nullptr;
    }

  pdb_dbg_printf ("MSF header processing complete");

  if (!pdb_read_stream_directory (pdb))
    {
      pdb_error ("Failed to read stream directory");
      return nullptr;
    }

  pdb_dbg_printf ("Stream directory processing complete");

  /* Read the /names stream for global string table.  */
  if (!pdb_read_names_stream (pdb))
    pdb_warning ("/names stream not available. filenames may be incomplete. ");

  pdb_dbg_printf ("Names stream processing complete");

  /* Read DBI stream (contains modules and line info).  */
  if (!pdb_read_dbi_stream (pdb))
    {
      pdb_error ("DBI stream failed to read");
      return nullptr;
    }

  pdb_dbg_printf ("DBI stream processing complete");

  if (!pdb_read_file_info_substream (pdb))
    {
      pdb_error ("File Info substream failed to read");
      return nullptr;
    }

  pdb_dbg_printf ("File Info substream processing complete");


  /* Read and parse TPI/IPI type streams — each type gets stored in the pdb
     but is not yet processed - it's index is recorded so it'll be expaned
     and cached as soon as the sumbol references it.  */
  if (!pdb_read_tpi_stream (pdb))
    {
      pdb_dbg_printf ("Failed to read TPI stream — type resolution will be limited");
      /* Non-fatal: continue without TPI (symbols will use int fallback for compound types) */
    }

  /* Read section addresses from BFD for (section, offset) -> PC mapping.  */
  if (!pdb_read_sections_from_bfd (pdb))
    {
      pdb_warning ("Failed to read sections from BFD");
      return nullptr;
    }

  pdb_dbg_printf ("BFD sections processing complete");

  /* Read and cache the symbol record stream.  */
  pdb_read_sym_record_stream (pdb);

  /* Build minimal symbols from S_PUB32 records in the public symbol stream.  */
  pdb_build_minsyms (pdb);

  /* TODO: Initialize the GSI hash table for name-based module lookup.  */
  // pdb_init_gsi_table (pdb);

  /* Store PDB for the main - used by info commands.  */
  pdb_register_loaded_pdb (pdb);

  printf_unfiltered ("\nUsing PDB file: %s\n", pdb->pdb_file_path);
  return pdb;
}

/* With OBJF_READNOW the PDB reader expands all modules immediately.
   Below are minimal quick functions to support this use model.  */

struct pdb_readnow_functions : public quick_symbol_functions
{
  bool has_symbols (struct objfile *objfile) override
  {
    pdb_qf_printf ("has_symbols called");
    return true;
  }

  bool has_unexpanded_symtabs (struct objfile *objfile) override
  {
    pdb_qf_printf ("has_unexpanded_symtabs called");
    return false;
  }

  struct symtab *find_last_source_symtab (struct objfile *objfile) override
  {
    pdb_qf_printf ("find_last_source_symtab called");
    return nullptr;
  }

  void forget_cached_source_info (struct objfile *objfile) override
  {
    /* PDB has no internal file name cache like DWARF's per_cu cache.
       Symtab fullname cleanup is handled by GDB generically. */
  }

  enum language lookup_global_symbol_language
    (struct objfile *objfile, const char *name,
     domain_search_flags domain, bool *symbol_found_p) override
  {
    pdb_qf_printf ("lookup_global_symbol_language called for '%s'", name);
    fflush (stderr);
    *symbol_found_p = false;
    return language_unknown;
  }

  void print_stats (struct objfile *objfile, bool print_bcache) override {}
  void dump (struct objfile *objfile) override {}
  void expand_all_symtabs (struct objfile *objfile) override
  {
    pdb_qf_printf ("expand_all_symtabs called");
  }

  iteration_status search
    (struct objfile *objfile,
     search_symtabs_file_matcher file_matcher,
     const lookup_name_info *lookup_name,
     search_symtabs_symbol_matcher symbol_matcher,
     compunit_symtab_iteration_callback compunit_callback,
     block_search_flags search_flags,
     domain_search_flags domain,
     search_symtabs_lang_matcher lang_matcher = nullptr) override
  {
    (void) lookup_name;
    (void) symbol_matcher;
    (void) search_flags;
    (void) domain;
    (void) lang_matcher;

    if (compunit_callback == nullptr)
      return iteration_status::keep_going;

    for (compunit_symtab &cu : objfile->compunits ())
      {
	/* If a file matcher is supplied, check whether any filetab
	   in this compunit matches.  */
	if (file_matcher != nullptr)
	  {
	    bool match = false;
	    for (symtab *s : cu.filetabs ())
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

  struct compunit_symtab *find_pc_sect_compunit_symtab
    (struct objfile *objfile, bound_minimal_symbol msymbol,
     CORE_ADDR pc, struct obj_section *section, int warn_if_readin) override
  {
    (void) msymbol;
    (void) section;
    (void) warn_if_readin;
    pdb_qf_printf ("find_pc_sect called for 0x%" PRIx64, pc);

    /* Iterate PDB modules directly to find matching PC. */
    struct pdb_per_objfile *pdb = pdb_objfile_data_key.get (objfile);
    if (pdb == nullptr)
      return nullptr;

    for (uint32_t i = 0; i < pdb->num_modules; i++)
      {
	struct pdb_module_info *module = &pdb->modules[i];

	/* Skip unexpanded modules. */
	if (module->cu == nullptr)
	  continue;

	const blockvector *bv = module->cu->blockvector ();
	if (bv == nullptr)
	  {
	    pdb_qf_printf ("find_pc_sect: module %d bv is null", i);
	    continue;
	  }

	const struct block *b = bv->global_block ();
	if (b != nullptr && b->start () <= pc && pc < b->end ())
	  {
	    pdb_qf_printf ("find_pc_sect: module %d matched [0x%" PRIx64 "-0x%" PRIx64 "]",
			   i, (uint64_t) b->start (), (uint64_t) b->end ());
	    fflush (stderr);
	    return module->cu;
	  }
      }

    pdb_qf_printf ("find_pc_sect: no match");
    fflush (stderr);
    return nullptr;
  }

  struct symbol *find_symbol_by_address
    (struct objfile *objfile, CORE_ADDR address) override
  {
    pdb_qf_printf ("find_symbol_by_address called for 0x%" PRIx64,
		   address);
    fflush (stderr);
    return nullptr;
  }

  void map_symbol_filenames (struct objfile *objfile,
			     symbol_filename_listener fun,
			     bool need_fullname) override {
	     pdb_qf_printf (" map_symbol_filenames called" );
	   }
};

/* Entry point: Read PDB debug info for a PE/COFF objfile.
   Returns true if PDB data was found (and processed).
   Called from coff_symfile_read, analogous to dwarf2_initialize_objfile.  */
bool
pdb_initialize_objfile (struct objfile *objfile)
{
  /* PDB read only if PE/COFF.  */
  if (bfd_get_flavour (objfile->obfd.get ()) != bfd_target_coff_flavour){
    return false;
  }

  struct pdb_per_objfile *pdb = pdb_read_pdb_file (objfile);
  if (pdb == nullptr)
    return false;

  /* Expand all modules eagerly (TODO: lazy loading).  */
  if (PDB_LAZY_INDEX_DISABLED || objfile->flags & OBJF_READNOW)
    {
      pdb_dbg_printf ("readnow requested");

      pdb_expand_all_modules (pdb);
      objfile->qf.emplace_front (new pdb_readnow_functions);
    }
  else
    {
      /* TODO: Build a cooked index similar to DWARF's cooked_index.  */
    }

  return true;
}

/* Process the File Info substream from the DBI stream.
   Create the Names Buffer and precompute module pointers into the Names Buffer.
   Layout:
     uint16_t  NumModules;
     uint16_t  NumSourceFiles;
     uint16_t  ModFileCounts[NumModules];             file count per module
     uint32_t  FileNameOffsets[Sum(ModFileCounts)];   offsets into NamesBuffer
     char      NamesBuffer[];                         filename strings.  */
static bool
pdb_read_file_info_substream (struct pdb_per_objfile *pdb)
{
  if (pdb->stream_data[PDB_STREAM_DBI] == NULL)
    {
      pdb_error ("Can't read file info: DBI stream not loaded");
      return false;
    }

  gdb_byte *dbi = pdb->stream_data[PDB_STREAM_DBI];

  uint32_t mod_info_size     = UINT32_CAST (dbi + DBI_HDR_MODULE_SIZE_OFFS);
  uint32_t sect_contrib_size = UINT32_CAST (dbi + DBI_HDR_SECT_CONTRIB_OFFS);
  uint32_t sect_map_size     = UINT32_CAST (dbi + DBI_HDR_SECT_MAP_OFFS);
  uint32_t source_info_size  = UINT32_CAST (dbi + DBI_HDR_SOURCE_INFO_OFFS);

  uint32_t offset = DBI_HDR_SIZE + mod_info_size
		    + sect_contrib_size + sect_map_size;

  if (offset + source_info_size > pdb->stream_sizes[PDB_STREAM_DBI])
    {
      pdb_error ("File Info substream out of bounds");
      return false;
    }

  if (source_info_size < 4)
    {
      pdb_error ("File Info substream too small");
      return false;
    }

  gdb_byte *fi = dbi + offset;
  gdb_byte *fi_end = fi + source_info_size;

  /* Avoid repeated summation in each module by precomputing per-module
     file_name_offsets for each module.  */
  uint16_t fi_num_mods = UINT16_CAST (fi + FILE_INFO_HDR_NUM_MODULES_OFFS);

  if (fi_num_mods != pdb->num_modules)
    {
      pdb_error ("File Info module count (%u), expected %u",
		 fi_num_mods, pdb->num_modules);
      return false;
    }

  gdb_byte *mod_file_counts = fi + FILE_INFO_MOD_INDICES_OFFS
				+ fi_num_mods * FILE_INFO_ELEMENT_SIZE;

  /* File count array must fit within the substream.  */
  if (mod_file_counts + fi_num_mods * FILE_INFO_ELEMENT_SIZE > fi_end)
    {
      pdb_error ("File info substream too small");
      return false;
    }

  /* FileNameOffsets array starts after ModFileCounts.  */
  gdb_byte *offsets_base = mod_file_counts
			  + fi_num_mods * FILE_INFO_ELEMENT_SIZE;

  /* FileNameOffsets array doesn't have module boundaries explicitly marked.
     Thus, we iterate over the array based on ModFileCounts, find the boundaries
     and record the file name offsets for all the modules.  */
  uint32_t sum = 0;

  for (uint16_t i = 0; i < fi_num_mods; i++)
    {
      pdb->modules[i].file_name_offsets = UINT32_PTR (offsets_base + sum * 4);
      uint16_t fc = UINT16_CAST (mod_file_counts + i * FILE_INFO_ELEMENT_SIZE);
      pdb->modules[i].num_files = fc;
      sum += fc;  /* Skip past this module's offsets.  */
    }

  /* Names Buffer follows FileNameOffsets and contains null-terminated
     filename strings.  */
  gdb_byte *names_buf = offsets_base + sum * 4;
  if (names_buf < fi_end)
    {
      pdb->file_info_names_buffer = names_buf;
      pdb->file_info_names_buffer_size = (fi_end - names_buf);
    }
  else
    {
      pdb_error ("File info names buffer out of bounds");
    }

  return true;
}

/* Get module's file names from its precomputed file_name_offsets[] pointers
   into the global Names Buffer.  */
void
pdb_read_module_files (struct pdb_per_objfile *pdb,
		       struct pdb_module_info *module)
{
  /* Already populated.  */
  if (module->files != NULL)
    return;

  if (module->file_name_offsets == NULL
      || pdb->file_info_names_buffer == NULL)
    return;

  uint16_t file_count = module->num_files;
  if (file_count == 0)
    return;

  uint32_t *my_offsets = module->file_name_offsets;
  gdb_byte *nb = pdb->file_info_names_buffer;
  uint32_t nb_size = pdb->file_info_names_buffer_size;

  module->files = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack,
				   file_count, const char *);
  for (uint16_t i = 0; i < file_count; i++)
    {
      uint32_t name_offset = my_offsets[i];
      if (name_offset < nb_size)
	module->files[i] = CSTR (nb + name_offset);
      else
	module->files[i] = NULL;
    }
}

/* Resolve the i-th filename for a module from the cached files[] array.
   Returns NULL if index is out of range or files not yet loaded.
   Caller must call pdb_read_module_files first.  */
const char *
pdb_module_file_name (struct pdb_per_objfile *pdb,
		      struct pdb_module_info *module,
		      uint32_t index)
{
  if (index >= module->num_files || module->files == NULL)
    return NULL;

  return module->files[index];
}

/* Parse module's DEBUG_S_FILECHKSMS.  Populates module->files_c13[] with
   filename pointers into the  /names stream.  */
void
pdb_read_module_files_c13 (struct pdb_per_objfile *pdb,
			   struct pdb_module_info *module)
{
  if (module->file_checksums == NULL || module->file_checksums_size == 0)
    return;

  /* Skip if files already read.  */
  if (module->files_c13_read)
    return;
  module->files_c13_read = true;

  gdb_byte *p = module->file_checksums;
  gdb_byte *end = p + module->file_checksums_size;

  /* Count the number of files.  */
  uint32_t count = 0;
  while (p + PDB_FILECHKSUM_HDR_SIZE <= end)
    {
      uint8_t ck_size = *(p + 4);  /* checksum_size field  */
      p += PDB_FILECHKSUM_HDR_SIZE + ck_size;
      p = (gdb_byte *) (((uintptr_t) p + 3) & ~(uintptr_t) 3);
      count++;
    }

  if (count == 0)
    return;

  /* Allocate file pointers.  */
  module->files_c13  = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack,
				       count, struct pdb_file_info);

  p = module->file_checksums;
  while (p + PDB_FILECHKSUM_HDR_SIZE <= end && module->num_files_c13 < count)
    {
      struct CV_FileChecksum *ck = (struct CV_FileChecksum *) p;
      uint8_t ck_size = ck->checksum_size;

      auto fname = pdb_string_table_get (
	  pdb->string_table, pdb->string_table_size,
	  ck->file_name_offset);

      struct pdb_file_info *fi = &module->files_c13[module->num_files_c13++];
      fi->filename = fname;
      fi->checksum_type = ck->checksum_type;

      if (ck_size > 0 && p + PDB_FILECHKSUM_HDR_SIZE + ck_size <= end)
	{
	  fi->checksum_size = ck_size;
	  fi->checksum = p + PDB_FILECHKSUM_HDR_SIZE;
	}

      p += PDB_FILECHKSUM_HDR_SIZE + ck_size;
      p = (gdb_byte *) (((uintptr_t) p + 3) & ~(uintptr_t) 3);
    }
}

INIT_GDB_FILE (pdb_read)
{
  pdb_init_loclist ();

  add_setshow_zuinteger_cmd ("pdb-read", no_class, &pdb_read_debug,
			     _("Set debugging of the PDB reader."),
			     _("Show debugging of the PDB reader."),
			     _("When enabled (non-zero), debugging messages are printed during PDB "
			       "reading and symtab expansion.  A value of 1 (one) provides basic "
			       "information.  A value greater than 1 provides more verbose information."),
			    NULL,
			    NULL,
			    &setdebuglist, &showdebuglist);

  add_setshow_zuinteger_cmd ("pdb-convert", no_class, &pdb_convert_debug, _("\
Set debugging of the PDB file name converter."), _("\
Show debugging of the PDB file name converter."), _("\
When enabled (non-zero), debugging messages are printed during PDB\n\
file name conversion.  A value of 1 (one) provides basic\n\
information.  A value greater than 1 provides more verbose information."),
			    NULL,
			    NULL,
			    &setdebuglist, &showdebuglist);

  add_setshow_zuinteger_cmd ("pdb-qf", no_class, &pdb_qf_debug, _("\
Set debugging of the PDB quick functions."), _("\
Show debugging of the PDB quick functions."), _("\
When enabled (non-zero), debugging messages are printed during PDB\n\
quick function processing.  A value of 1 (one) provides basic\n\
information.  A value greater than 1 provides more verbose information."),
			    NULL,
			    NULL,
			    &setdebuglist, &showdebuglist);

  /* Symbol name filter for targeted diagnostics in pdb_parse_symbols.  */
  static std::string pdb_sym_debug_name;
  add_setshow_string_noescape_cmd ("pdb-sym-name", no_class,
			    &pdb_sym_debug_name, _("\
Set symbol name filter for PDB symbol parsing diagnostics."), _("\
Show symbol name filter for PDB symbol parsing diagnostics."), _("\
When set, pdb_parse_symbols prints detailed info only for records\n\
whose name matches this string.  Example: set debug pdb-sym-name myglobalvar"),
			    NULL,
			    NULL,
			    &setdebuglist, &showdebuglist);
}
