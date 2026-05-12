/* PDB inlined-function (S_INLINESITE) binary-annotation decoder.

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

/* This file decodes the binary-annotation bytestream carried by an
   S_INLINESITE record into the inlined body's code chunks, maps those chunks
   onto relocated PC ranges, and records the inlined line-table entries.

   References:
     - microsoft-pdb cvinfo.h: https://github.com/microsoft/microsoft-pdb  */

#include "symtab.h"
#include "objfiles.h"
#include "buildsym.h"
#include "source.h"
#include "pdb/pdb-internal.h"

#include <algorithm>
#include <optional>
#include <vector>
#include <string>

namespace pdb
{

/* Inline function binary-annotation opcodes.  An S_INLINESITE record carries
   a set of opcodes that we need to interpret in order to decode the inlined
   function's code chunks and line-table entries.  Each opcode is a compressed
   integer and has a variable number of operands (also compressed integers).  */
enum pdb_inline_binannot_op
{
  /* End of the annotation program.  No operands.  */
  INLINE_BA_INVALID = 0,

  /* Set the current code offset to an absolute value.  Operand: the absolute
     code offset.  Unused here (TODO: Check if MSVC even emits this one).  */
  INLINE_BA_CODE_OFFSET = 1,

  /* Selects the base that following code offsets are measured from.
     Operand: the base index — 0 is the enclosing procedure.  Each chunk is
     tagged with the active base index; the offset resets to 0 for the new
     base.  Only base 0 is resolved here; chunks tagged with another base are
     dropped when mapped to PC ranges.  */
  INLINE_BA_CHANGE_CODE_OFFSET_BASE = 2,

  /* Advance the current code offset to the start of the next code chunk.
     Operand: the byte distance to add to the current code offset.  */
  INLINE_BA_CHANGE_CODE_OFFSET = 3,

  /* Emit one code chunk that starts at the current code offset.
     Operand: the chunk's byte length.  */
  INLINE_BA_CHANGE_CODE_LENGTH = 4,

  /* Switch the current source file for the chunks that follow (used when an
     inlined body spans multiple source files).  Operand is an offset into the
     module's DEBUG_S_FILECHKSMS subsection pointing to the new file.  */
  INLINE_BA_CHANGE_FILE = 5,

  /* Move the current source line.  Operand: a zig-zag-encoded signed delta
     added to the current source line.  */
  INLINE_BA_CHANGE_LINE_OFFSET = 6,

  /* Extend the current line span to cover several source lines.
     Operand: the number of extra lines.  Unused.  */
  INLINE_BA_CHANGE_LINE_END_DELTA = 7,

  /* Mark whether the current span is a statement or an expression boundary.
     Operand: the range kind.  Unused.  */
  INLINE_BA_CHANGE_RANGE_KIND = 8,

  /* Set the source start column of the current span.
     Operand: the start column.  Unused.  */
  INLINE_BA_CHANGE_COLUMN_START = 9,

  /* Move the source end column of the current span.
     Operand: a zig-zag-encoded signed end-column delta.  Unused.  */
  INLINE_BA_CHANGE_COLUMN_END_DELTA = 10,

  /* Advance the code offset and the source line together (the common case,
     one annotation per statement).  Operand: one packed value whose low
     nibble (bits 0-3) is added to the current code offset and whose high bits
     (operand >> 4) are a zig-zag-encoded signed line delta.  */
  INLINE_BA_CHANGE_CODE_OFFSET_AND_LINE_OFFSET = 11,

  /* Emit one code chunk and advance the code offset together.  Operands: two
     values, the chunk byte length then the code-offset distance; the offset
     advances the chunk's start and the length sizes it.  */
  INLINE_BA_CHANGE_CODE_LENGTH_AND_CODE_OFFSET = 12,

  /* Set the source end column of the current span.
     Operand: the end column.  Unused.  */
  INLINE_BA_CHANGE_COLUMN_END = 13,
};

/* Read a compressed integer from [*pp, END), advancing *pp.  The encoding is
   always UNSIGNED and the number of bytes (1/2/4) is selected by the lead
   byte's high bits.  Operands that are signed are zig-zag-decoded afterwards
   by pdb_decode_inline_signed.  Returns false on truncation or an invalid
   lead byte.  */

static bool
pdb_read_cv_uint (const gdb_byte **pp, const gdb_byte *end, uint32_t *out)
{
  const gdb_byte *p = *pp;
  if (p >= end)
    return false;

  uint8_t b0 = *p;
  if ((b0 & 0x80) == 0)
    {
      *out = b0;
      *pp = p + 1;
      return true;
    }
  if ((b0 & 0xC0) == 0x80)
    {
      if (p + 2 > end)
	return false;
      *out = ((uint32_t) (b0 & 0x3F) << 8) | p[1];
      *pp = p + 2;
      return true;
    }
  if ((b0 & 0xE0) == 0xC0)
    {
      if (p + 4 > end)
	return false;
      *out = ((uint32_t) (b0 & 0x1F) << 24) | ((uint32_t) p[1] << 16)
	     | ((uint32_t) p[2] << 8) | p[3];
      *pp = p + 4;
      return true;
    }
  return false;
}

/* Decode an unsigned binary-annotation operand as a signed inline line delta
   (CodeView zig-zag encoding).  */

static int32_t
pdb_decode_inline_signed (uint32_t u)
{
  return (u & 1) ? -(int32_t) (u >> 1) : (int32_t) (u >> 1);
}

/* See pdb-internal.h.  */

std::vector<pdb_inline_chunk>
pdb_decode_inline_annotations (const gdb_byte *annot, size_t len)
{
  std::vector<pdb_inline_chunk> chunks;
  const gdb_byte *p = annot;
  const gdb_byte *end = annot + len;
  uint32_t code_offset = 0;
  int32_t line_delta = 0;
  uint32_t file_id = PDB_INLINE_FILE_ID_BASE;
  uint32_t base_index = 0;

  /* A range is open once its start is known and closed once its end is;
     an offset transition supplies the end of the range before it.  */
  bool have_range_start = false;
  bool have_range_end = false;
  uint32_t range_start = 0;
  uint32_t range_end = 0;
  int32_t cur_line = 0;
  bool have_next_line = false;
  int32_t next_line = 0;
  bool have_next_file_id = false;
  uint32_t next_file_id = PDB_INLINE_FILE_ID_BASE;
  bool terminal = false;

  auto advance_code = [&] (uint32_t delta) {
    if (!have_range_start)
      {
	have_range_start = true;
	range_start = code_offset;
      }
    else if (!have_range_end)
      {
	have_range_end = true;
	range_end = range_start + delta;
	}
  };
  auto add_line_delta = [&] (uint32_t u) {
    line_delta += pdb_decode_inline_signed (u);
    if (!have_range_start)
      cur_line = line_delta;
    else
      {
	have_next_line = true;
	next_line = line_delta;
      }
  };
  auto set_file = [&] (uint32_t fid) {
    if (!have_range_start)
      file_id = fid;
    else
      {
	have_next_file_id = true;
	next_file_id = fid;
      }
  };
  auto flush_range = [&] (bool is_terminal) {
    chunks.push_back ({ range_start, range_end, cur_line, file_id,
			base_index });

    if (have_next_file_id)
      {
	file_id = next_file_id;
	have_next_file_id = false;
      }
    if (have_next_line)
      {
	cur_line = next_line;
	have_next_line = false;
      }

    /* A length ends the run; otherwise the next range picks up where this
       one stopped.  */
    if (is_terminal)
      have_range_start = false;
    else
      range_start = range_end;
    have_range_end = false;
  };

  while (p < end)
    {
      uint32_t op;
      if (!pdb_read_cv_uint (&p, end, &op) || op == INLINE_BA_INVALID)
	break;

      terminal = false;

      switch (op)
	{
	case INLINE_BA_CODE_OFFSET:
	case INLINE_BA_CHANGE_CODE_OFFSET:
	  {
	    uint32_t delta;
	    if (!pdb_read_cv_uint (&p, end, &delta))
	      return chunks;
	    code_offset += delta;
	    advance_code (delta);
	    break;
	  }
	case INLINE_BA_CHANGE_CODE_OFFSET_BASE:
	  {
	    uint32_t base;
	    if (!pdb_read_cv_uint (&p, end, &base))
	      return chunks;
	    base_index = base;
	    code_offset = 0;
      have_range_start = false;
      have_range_end = false;
	    break;
	  }
	case INLINE_BA_CHANGE_FILE:
	  {
	    uint32_t fid;
	    if (!pdb_read_cv_uint (&p, end, &fid))
	      return chunks;
	    set_file (fid);
	    break;
	  }
	case INLINE_BA_CHANGE_LINE_OFFSET:
	  {
	    uint32_t operand;
	    if (!pdb_read_cv_uint (&p, end, &operand))
	      return chunks;
	    add_line_delta (operand);
	    break;
	  }
	case INLINE_BA_CHANGE_CODE_OFFSET_AND_LINE_OFFSET:
	  {
	    uint32_t packed;
	    if (!pdb_read_cv_uint (&p, end, &packed))
	      return chunks;
	    code_offset += packed & 0x0F;
	    add_line_delta (packed >> 4);
      advance_code (packed & 0x0F);
	    break;
	  }
	case INLINE_BA_CHANGE_CODE_LENGTH:
	  {
	    uint32_t length;
	    if (!pdb_read_cv_uint (&p, end, &length))
	      return chunks;
	    advance_code (length);
	    code_offset += length;
	    terminal = true;
	    break;
	  }
	case INLINE_BA_CHANGE_CODE_LENGTH_AND_CODE_OFFSET:
	  {
	    uint32_t length, off;
	    if (!pdb_read_cv_uint (&p, end, &length)
		|| !pdb_read_cv_uint (&p, end, &off))
	      return chunks;
	    code_offset += off;
	    advance_code (off);
	    /* The offset ends any range already open; the length then
	       describes a range of its own.  */
      if (have_range_end)
	      flush_range (false);
	    advance_code (length);
	    code_offset += length;
	    terminal = true;
	    break;
	  }
	case INLINE_BA_CHANGE_LINE_END_DELTA:
	case INLINE_BA_CHANGE_RANGE_KIND:
	case INLINE_BA_CHANGE_COLUMN_START:
	case INLINE_BA_CHANGE_COLUMN_END_DELTA:
	case INLINE_BA_CHANGE_COLUMN_END:
	  {
	    /* Not used for code ranges or line tracking; consume the operand
	       so parsing stays aligned to the next opcode.  */
	    uint32_t ignored;
	    if (!pdb_read_cv_uint (&p, end, &ignored))
	      return chunks;
	    break;
	  }
	default:
	  /* Unknown opcode: operand count is unknown, so stop.  */
	  return chunks;
	}

      if (have_range_start && have_range_end)
	flush_range (terminal);
    }

  return chunks;
}

/* See pdb-internal.h.  */

pdb_range_pair_vec
pdb_inline_chunk_ranges (pdb_per_objfile *pdb,
			 const std::vector<pdb_inline_chunk> &chunks,
			 const std::vector<pdb_code_origin> &bases)
{
  pdb_range_pair_vec ranges;
  for (const pdb_inline_chunk &c : chunks)
    {
      if (c.base_index >= bases.size ())
	continue;
      const pdb_code_origin &b = bases[c.base_index];
      CORE_ADDR lo = pdb->map_section_offset_to_pc (b.sect,
						    b.off + c.start_off);
      CORE_ADDR hi = pdb->map_section_offset_to_pc (b.sect,
						    b.off + c.end_off);
      if (lo != 0 && hi > lo)
	ranges.emplace_back (lo, hi);
    }
  return ranges;
}

/* DEBUG_S_INLINEELINES (0xF6) - an array in the module's debug data.  For
   each function that was inlined, it records the source file and the line
   number where that function's body begins.  The S_INLINESITE binary
   annotations only carry deltas (line +N, ChangeFile), so they need this data
   to resolve the actual source line numbers.

   Layout [cvinfo.h: CV_InlineeSourceLine / CV_InlineeSourceLineEx]:
       uint32_t signature           CV_INLINEE_SOURCE_LINE_SIGNATURE (0) or
				    ..._SIGNATURE_EX (1); EX adds the trailer
       then repeating entries:
       uint32_t inlinee             IPI item id; matches S_INLINESITE.inlinee
				    — the lookup key
       uint32_t fileId              offset into DEBUG_S_FILECHKSMS (the file)
       uint32_t sourceLineNum       1-based first line of the inlinee's body
       — EX form only:
       uint32_t countOfExtraFiles
       uint32_t extraFileId[countOfExtraFiles]

   Scan MODULE's DEBUG_S_INLINEELINES array for the entry whose inlinee field
   equals INLINEE_ID — i.e. where that inlined function's body starts in the
   source.  When found, write its fileId to *FILE_ID and its sourceLineNum to
   *BASE_LINE (the starting point the S_INLINESITE line deltas are added to)
   and return true.  Return false when the module has no such array or no entry
   matches.  */

static bool
pdb_lookup_inlinee_source (pdb_module_info *module, uint32_t inlinee_id,
			 uint32_t *file_id, uint32_t *base_line)
{
  const gdb_byte *p = module->inlinee_lines;
  if (p == nullptr || module->inlinee_lines_size < 4)
    return false;

  const gdb_byte *end = p + module->inlinee_lines_size;
  uint32_t signature = read_u32 (p);
  p += 4;
  bool has_extra = (signature == CV_INLINEE_SOURCE_LINE_SIGNATURE_EX);

  while (p + 12 <= end)
    {
      uint32_t id = read_u32 (p);
      uint32_t fid = read_u32 (p + 4);
      uint32_t line = read_u32 (p + 8);
      p += 12;

      if (has_extra)
	{
	  if (p + 4 > end)
	    break;
	  uint32_t extra = read_u32 (p);
	  p += 4 + (uint64_t) extra * 4;
	}

      if (id == inlinee_id)
	{
	  *file_id = fid;
	  *base_line = line;
	  return true;
	}
    }

  return false;
}

/* See pdb-internal.h.  */

void
pdb_record_inline_lines
  (pdb_per_objfile *pdb, pdb_module_info *module, buildsym_compunit *cu,
   uint32_t inlinee_id, const std::vector<pdb_code_origin> &bases,
   const std::vector<pdb_inline_chunk> &chunks,
   const std::vector<const std::vector<pdb_inline_line_span> *> &enclosing,
   const pdb_symbol_lines *sym_lines,
   std::vector<pdb_inline_line_span> *spans)
{
  uint32_t base_file_id;
  uint32_t base_line;
  if (!pdb_lookup_inlinee_source (module, inlinee_id, &base_file_id, &base_line))
    return;

  /* Resolve and start a subfile lazily, re-resolving only when a chunk's file
     id differs from the previous one (ChangeFile annotations switch files
     mid-body).  */
  subfile *sf = nullptr;
  uint32_t cur_file_id = PDB_INLINE_FILE_ID_BASE;
  bool have_subfile = false;

  /* True when some line already starts at PC, so the line before it
     already stops there.  */
  auto line_starts_at = [] (const subfile *s, unrelocated_addr pc) {
    for (const linetable_entry &e : s->line_vector_entries)
      if (e.unrelocated_pc () == pc)
	return true;
    return false;
  };

  /* The caller's line in effect at PC: from the innermost enclosing inline
     site covering PC, else the procedure's C13 row there.  */
  auto caller_line_at = [&] (CORE_ADDR pc) -> std::optional<pdb_inline_line_span>
    {
      for (const std::vector<pdb_inline_line_span> *site : enclosing)
	for (const pdb_inline_line_span &s : *site)
	  if (s.start <= pc && pc < s.end)
	    return s;

      if (sym_lines == nullptr)
	return {};

      const auto &rows = sym_lines->rows;
      auto it = std::upper_bound (rows.begin (), rows.end (), pc,
				  [] (CORE_ADDR a,
				      const pdb_symbol_lines::row &r)
				  { return a < r.pc; });
      if (it == rows.begin () || std::prev (it)->line == 0)
	return {};
      const pdb_symbol_lines::row &r = *std::prev (it);
      return pdb_inline_line_span { r.pc, pc, r.line, r.subfile };
    };

  CORE_ADDR text_off = pdb->objfile->text_section_offset ();

  for (size_t i = 0; i < chunks.size (); i++)
    {
      const pdb_inline_chunk &c = chunks[i];
      if (c.base_index >= bases.size ())
	continue;
      const pdb_code_origin &b = bases[c.base_index];
      CORE_ADDR pc = pdb->map_section_offset_to_pc (b.sect,
						    b.off + c.start_off);
      if (pc == 0)
	continue;

      uint32_t file_id = (c.file_id == PDB_INLINE_FILE_ID_BASE)
			  ? base_file_id : c.file_id;
      if (!have_subfile || file_id != cur_file_id)
	{
	  const char *filename
	    = pdb_get_filename_from_file_id (pdb, module, file_id);
	  if (filename == nullptr)
	    continue;
	  std::string path = pdb_convert_path (filename);
	  sf = cu->start_subfile (path.c_str (), path.c_str ());
	  cur_file_id = file_id;
	  have_subfile = true;
	}

      unrelocated_addr unrelocated_pc { pc - text_off };
      int line = (int) ((int64_t) base_line + c.line_delta);
      if (line < 0)
	line = 0;
      cu->record_line (sf, line, unrelocated_pc, LEF_IS_STMT);

      CORE_ADDR end_pc = pdb->map_section_offset_to_pc (b.sect,
							b.off + c.end_off);
      if (spans != nullptr && end_pc > pc)
	spans->push_back ({ pc, end_pc, line, sf });

      /* The next chunk continues the body, so it already bounds this one.  */
      if (i + 1 < chunks.size () && chunks[i + 1].base_index == c.base_index
	  && chunks[i + 1].start_off == c.end_off)
	continue;

      if (end_pc == 0)
	continue;

      unrelocated_addr end_unrelocated { end_pc - text_off };
      std::optional<pdb_inline_line_span> caller = caller_line_at (end_pc);

      if ((!caller.has_value () || caller->subfile != sf)
	  && !line_starts_at (sf, end_unrelocated))
	cu->record_line (sf, 0, end_unrelocated, LEF_IS_STMT);

      if (caller.has_value ()
	  && !line_starts_at (caller->subfile, end_unrelocated))
	cu->record_line (caller->subfile, caller->line, end_unrelocated,
			 LEF_IS_STMT);
    }
}

} /* namespace pdb */
