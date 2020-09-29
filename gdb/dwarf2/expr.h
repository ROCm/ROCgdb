/* DWARF 2 Expression Evaluator.

   Copyright (C) 2001-2020 Free Software Foundation, Inc.

   Contributed by Daniel Berlin <dan@dberlin.org>.

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

#if !defined (DWARF2EXPR_H)
#define DWARF2EXPR_H

#include "leb128.h"
#include "gdbtypes.h"

class dwarf_entry;
class dwarf_entry_factory;
struct dwarf2_per_objfile;

/* The expression evaluator works with a dwarf_expr_context, describing
   its current state and its callbacks.  */
struct dwarf_expr_context
{
  /* We should ever only pass in the PER_OBJFILE and the ADDR_SIZE
     information should be retrievable from there.  The PER_OBJFILE
     contains a pointer to the PER_BFD information anyway and the
     address size information must be the same for the whole BFD.   */

  dwarf_expr_context (struct dwarf2_per_objfile *per_objfile,
		      int addr_size);
  virtual ~dwarf_expr_context ();

  void push_address (CORE_ADDR addr, bool in_stack_memory);
  struct value *eval_exp (const gdb_byte *addr, size_t len, bool as_lval,
			  struct dwarf2_per_cu_data *per_cu,
			  struct frame_info *frame,
			  const struct property_addr_info *addr_info = nullptr,
			  struct type *type = nullptr,
			  struct type *subobj_type = nullptr,
			  LONGEST subobj_offset = 0);

private:
  /* The stack of values.  */
  std::vector<dwarf_entry *> stack;

  /* Target architecture to use for address operations.  */
  struct gdbarch *gdbarch;

  /* Target address size in bytes.  */
  int addr_size;

  /* DW_FORM_ref_addr size in bytes.  If -1 DWARF is executed from a frame
     context and operations depending on DW_FORM_ref_addr are not allowed.  */
  int ref_addr_size;

  /* The current depth of dwarf expression recursion, via DW_OP_call*,
     DW_OP_fbreg, DW_OP_push_object_address, etc., and the maximum
     depth we'll tolerate before raising an error.  */
  int recursion_depth, max_recursion_depth;

  /* We evaluate the expression in the context of this objfile.  */
  dwarf2_per_objfile *per_objfile;

  /* Frame information used for the evaluation.  */
  struct frame_info *frame;

  /* Compilation unit used for the evaluation.  */
  struct dwarf2_per_cu_data *per_cu;

  /* Property address info used for the evaluation.  */
  const struct property_addr_info *addr_info;

  /* Factory in charge of the dwarf entry's life cycle.  */
  dwarf_entry_factory *entry_factory;

  void eval (const gdb_byte *addr, size_t len);
  struct type *address_type () const;
  void push (dwarf_entry *value);
  bool stack_empty_p () const;
  dwarf_entry *add_piece (ULONGEST bit_size, ULONGEST bit_offset);
  dwarf_entry *create_select_composite (ULONGEST piece_bit_size,
					ULONGEST pieces_count);
  dwarf_entry *create_extend_composite (ULONGEST piece_bit_size,
					ULONGEST pieces_count);
  void execute_stack_op (const gdb_byte *op_ptr, const gdb_byte *op_end);
  void pop ();
  dwarf_entry *fetch (int n);
  CORE_ADDR fetch_address (int n);
  bool fetch_in_stack_memory (int n);
  struct value *fetch_result (struct type *type,
			      struct type *subobj_type,
			      LONGEST subobj_offset,
			      bool as_lval);
  void get_frame_base (const gdb_byte **start, size_t *length);
  struct type *get_base_type (cu_offset die_cu_off, int size);
  void dwarf_call (cu_offset die_cu_off);
  void push_dwarf_reg_entry_value (enum call_site_parameter_kind kind,
				   union call_site_parameter_u kind_u,
				   int deref_size);
  dwarf_entry* dwarf_entry_deref (dwarf_entry *entry, struct type *type,
				  size_t size = 0);
  dwarf_entry *gdb_value_to_dwarf_entry (struct value *value);
  struct value *dwarf_entry_to_gdb_value (dwarf_entry *entry,
					  struct type *type,
					  struct type *subobj_type = nullptr,
					  LONGEST subobj_offset = 0);
};

void dwarf_expr_require_composition (const gdb_byte *, const gdb_byte *,
				     const char *);

int dwarf_block_to_dwarf_reg (const gdb_byte *buf, const gdb_byte *buf_end);

int dwarf_block_to_dwarf_reg_deref (const gdb_byte *buf,
				    const gdb_byte *buf_end,
				    CORE_ADDR *deref_size_return);

int dwarf_block_to_fb_offset (const gdb_byte *buf, const gdb_byte *buf_end,
			      CORE_ADDR *fb_offset_return);

int dwarf_block_to_sp_offset (struct gdbarch *gdbarch, const gdb_byte *buf,
			      const gdb_byte *buf_end,
			      CORE_ADDR *sp_offset_return);

/* Wrappers around the leb128 reader routines to simplify them for our
   purposes.  */

static inline const gdb_byte *
gdb_read_uleb128 (const gdb_byte *buf, const gdb_byte *buf_end,
		  uint64_t *r)
{
  size_t bytes_read = read_uleb128_to_uint64 (buf, buf_end, r);

  if (bytes_read == 0)
    return NULL;
  return buf + bytes_read;
}

static inline const gdb_byte *
gdb_read_sleb128 (const gdb_byte *buf, const gdb_byte *buf_end,
		  int64_t *r)
{
  size_t bytes_read = read_sleb128_to_int64 (buf, buf_end, r);

  if (bytes_read == 0)
    return NULL;
  return buf + bytes_read;
}

static inline const gdb_byte *
gdb_skip_leb128 (const gdb_byte *buf, const gdb_byte *buf_end)
{
  size_t bytes_read = skip_leb128 (buf, buf_end);

  if (bytes_read == 0)
    return NULL;
  return buf + bytes_read;
}

extern const gdb_byte *safe_read_uleb128 (const gdb_byte *buf,
					  const gdb_byte *buf_end,
					  uint64_t *r);

extern const gdb_byte *safe_read_sleb128 (const gdb_byte *buf,
					  const gdb_byte *buf_end,
					  int64_t *r);

extern const gdb_byte *safe_skip_leb128 (const gdb_byte *buf,
					 const gdb_byte *buf_end);

extern CORE_ADDR aspace_address_to_flat_address (CORE_ADDR address,
						 unsigned int address_space);

#endif /* dwarf2expr.h */
