/* DWARF 2 Expression Evaluator.

   Copyright (C) 2001-2020 Free Software Foundation, Inc.
   Copyright (C) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

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
#include "gdb/frame.h"
#include "gdbsupport/refcounted-object.h"

struct dwarf2_per_objfile;
class dwarf_entry;
class dwarf_entry_factory;

/* The closure base class describing a different lval_computed
   closure cases. These classes are interfaces to appropriated
   dwarf_entry subclasses encapsulate in them.  */
class computed_closure : public refcounted_object
{
public:
  enum closure_kind {Computed, ImplicitPtr, Composite};

  /* Not supposed to be called on it's own.  */
  computed_closure (dwarf_entry *entry);

  virtual ~computed_closure (void);

  /* Checker method to determine the objects base class.

     Note that every class is a base class to itself.  */

  bool is_closure (closure_kind kind) const
  {
    return m_kind == kind || kind == Computed;
  }

  dwarf_entry *get_entry (void)
  {
    return m_entry;
  }

protected:
  /* Kind of the underlying class.  */
  closure_kind m_kind = Computed;

private:
  /* Entry that this class encapsulates.  */
  dwarf_entry *m_entry;
};

/* The closure class for computed values that
   hold an implicit pointer location description.  */
class implicit_pointer_closure : public computed_closure
{
public:
  implicit_pointer_closure (dwarf_entry *entry,
                            struct dwarf2_per_cu_data *per_cu,
                            dwarf2_per_objfile *per_objfile) :
                            computed_closure {entry}, m_per_cu (per_cu),
                            m_per_objfile (per_objfile) {m_kind = ImplicitPtr;}

  sect_offset get_die_offset (void);

  LONGEST get_offset (void);

  dwarf2_per_cu_data *get_per_cu (void)
  {
    return m_per_cu;
  }

  dwarf2_per_objfile *get_per_objfile (void)
  {
    return m_per_objfile;
  }


private:
  /* Compilation unit context of the pointer.  */
  struct dwarf2_per_cu_data *m_per_cu;

  /* Per object fggile data of the pointer.  */
  dwarf2_per_objfile *m_per_objfile;
};

/* The closure class for computed values
   that hold an composite location description.  */
class composite_closure : public computed_closure
{
public:
  composite_closure (dwarf_entry *entry, struct frame_id frame_id) :
                     computed_closure {entry}, m_frame_id (frame_id)
                     {m_kind = Composite;}

  struct frame_id get_frame_id (void) const
  {
    return m_frame_id;
  }

private:
  /* Frame ID context of the composite.  */
  struct frame_id m_frame_id;
};

/* The expression evaluator works with a dwarf_expr_context, describing
   its current state and its callbacks.  */
struct dwarf_expr_context
{
  dwarf_expr_context (dwarf2_per_objfile *per_objfile);
  virtual ~dwarf_expr_context ();

  void push_address (CORE_ADDR addr, bool in_stack_memory);
  void eval (const gdb_byte *addr, size_t len);
  struct value *fetch_value (bool as_lval = true,
                             struct type *type = nullptr,
                             struct type *subobj_type = nullptr,
                             LONGEST subobj_offset = 0);

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

  /* Factory in charge of the dwarf entry's life cycle.  */
  dwarf_entry_factory *entry_factory;

  /* Return a context frame information.  */
  virtual struct frame_info* get_context_frame (void) = 0;

  /* Return a context compilation unit information.  */
  virtual struct dwarf2_per_cu_data *get_per_cu (void) = 0;

  /* Read LENGTH bytes at ADDR into BUF.  */
  virtual bool read_passed_in_mem (gdb_byte *buf, CORE_ADDR addr,
                                   size_t length)
  {
    return false;
  }

  /* Return the location expression for the frame base attribute, in
     START and LENGTH.  The result must be live until the current
     expression evaluation is complete.  */
  virtual void get_frame_base (const gdb_byte **start, size_t *length) = 0;

  /* Return the CFA for the frame.  */
  virtual CORE_ADDR get_frame_cfa () = 0;

  /* Return the PC for the frame.  */
  virtual CORE_ADDR get_frame_pc ()
  {
    error (_("%s is invalid in this context"), "DW_OP_implicit_pointer");
  }

  /* Return the thread-local storage address for
     DW_OP_GNU_push_tls_address or DW_OP_form_tls_address.  */
  virtual CORE_ADDR get_tls_address (CORE_ADDR offset) = 0;

  /* Execute DW_AT_location expression for the DWARF expression
     subroutine in the DIE at DIE_CU_OFF in the CU.  Do not touch
     STACK while it being passed to and returned from the called DWARF
     subroutine.  */
  virtual void dwarf_call (cu_offset die_cu_off) = 0;

  /* Execute "variable value" operation on the DIE at SECT_OFF.  */
  virtual struct value *dwarf_variable_value (sect_offset sect_off) = 0;

  /* Return the base type given by the indicated DIE at DIE_CU_OFF.
     This can throw an exception if the DIE is invalid or does not
     represent a base type.  SIZE is non-zero if this function should
     verify that the resulting type has the correct size.  */
  virtual struct type *get_base_type (cu_offset die_cu_off, int size)
  {
    /* Anything will do.  */
    return builtin_type (this->gdbarch)->builtin_int;
  }

  /* Push on DWARF stack an entry evaluated for DW_TAG_call_site's
     parameter matching KIND and KIND_U at the caller of specified BATON.
     If DEREF_SIZE is not -1 then use DW_AT_call_data_value instead of
     DW_AT_call_value.  */
  virtual void push_dwarf_reg_entry_value (enum call_site_parameter_kind kind,
					   union call_site_parameter_u kind_u,
					   int deref_size) = 0;

  /* Return the address indexed by DW_OP_addrx or DW_OP_GNU_addr_index.
     This can throw an exception if the index is out of range.  */
  virtual CORE_ADDR get_addr_index (unsigned int index) = 0;

  /* Return the `object address' for DW_OP_push_object_address.  */
  virtual CORE_ADDR get_object_address () = 0;

  /* Get context closure callbacks structure used for
     lval_computed resolution.  */
  virtual const struct lval_funcs *get_closure_callbacks (void) = 0;

  /* Indicator if the evaluation requiers a frame information.  */
  virtual void eval_needs_frame (void)
  {
    return;
  }

  /* Query if a context can apply a dereference operation.
     If not a dummy value will be generated as a result.  */
  virtual bool context_can_deref (void)
  {
    return true;
  }

  /* Fetch a value pointed to by an implicit pointer. DIE_OFFSET
     holds an offset to the pointed die, OFFSET holds an offset
     into that die and a TYPE holds an expected type of the value.  */
  virtual struct value *deref_implicit_pointer (sect_offset die_offset,
                                                LONGEST offset,
                                                struct type *type);

private:
  struct type *address_type () const;
  void push (dwarf_entry *entry);
  bool stack_empty_p () const;
  dwarf_entry *add_piece (ULONGEST size, ULONGEST bit_offset);
  void execute_stack_op (const gdb_byte *op_ptr, const gdb_byte *op_end);
  dwarf_entry *fetch (int n);
  void pop ();

  /* Convert DWARF entry to the matching struct value representation of the
     given TYPE type. SUBOBJ_TYPE information if specified, will be used for
     more precise description of the source variable type information. Where
     SUBOBJ_OFFSET defines an offset into the DWARF entry contents.  */
  struct value *dwarf_entry_to_gdb_value (dwarf_entry *entry,
                                          struct type *type,
                                          struct type *subobj_type = nullptr,
                                          LONGEST subobj_offset = 0);

  /* Convert struct value to the matching DWARF entry representation.
     Used for non-standard DW_OP_GNU_variable_value operation support.  */
  dwarf_entry *gdb_value_to_dwarf_entry (struct value *value);

  /* Apply dereference operation on the DWARF ENTRY. In the case of a value
     entry, the entry will be implicitly converted to the appropriate location
     description before the operation is applied. If the SIZE is specified,
     it must be equal or smaller then the TYPE type size. Ii SIZE is smaller
     then the type size, the value will be zero extended to the difference.

     If the context doesn't support the dereference operation, a dummy value
     of 1 will be returned.  */
  dwarf_entry* dwarf_entry_deref (dwarf_entry *entry, struct type *type,
                                  size_t size = 0);

  /* Apply dereference operation on a DWARF ENTRY, where the result of the
     operation will be reinterpeted based on a TYPE information. This function
     will default back to the dwarf_entry_deref function if the DWARF entry is
     anything then register location description.

     If the context doesn't support the dereference operation a dummy value
     of 1 will be returned.  */
  dwarf_entry* dwarf_entry_deref_reinterpret (dwarf_entry *entry,
                                              struct type *type);

  /* Compare two DWARF entries ARG1 and ARG2 for equality in a context of a
     value entry comparison. If ARG1 or ARG2 are not value entries, they will
     be implicitly converted before the operation is applied.  */
  bool dwarf_entry_equal_op (dwarf_entry *arg1, dwarf_entry *arg2);

  /* Compare if DWARF entry ARG1 is lesser the DWARF entry ARG2 in a context
     of a value entry comparison. If ARG1 or ARG2 are not value entries, they
     will be implicitly converted before the operation is applied.  */
  bool dwarf_entry_less_op (dwarf_entry *arg1, dwarf_entry *arg2);
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

/* Shared piece closure callback functions.  */

extern void rw_closure_value (struct value *v, struct value *from);

extern void *copy_value_closure (const struct value *v);

extern void free_value_closure (struct value *v);

#endif /* dwarf2expr.h */
