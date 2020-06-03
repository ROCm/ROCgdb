/* DWARF 2 Expression Evaluator.

   Copyright (C) 2001-2020 Free Software Foundation, Inc.
   Copyright (C) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

   Contributed by Daniel Berlin (dan@dberlin.org)

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

#include "defs.h"
#include "symtab.h"
#include "gdbtypes.h"
#include "value.h"
#include "gdbcore.h"
#include "dwarf2.h"
#include "dwarf2/expr.h"
#include "dwarf2/loc.h"
#include "dwarf2/read.h"
#include "gdbsupport/underlying.h"
#include "gdbsupport/refcounted-object.h"
#include "gdbarch.h"
#include "objfiles.h"

/* Base class that describes entries found on a DWARF expression
   evaluation stack.  */

class dwarf_entry : public refcounted_object
{
public:
  enum entry_kind {Entry, Value, Location, Register, Memory,
                   Undefined, Implicit, ImplicitPtr, Composite};

  /* Not expected to be called on it's own.  */
  dwarf_entry (void) = default;

  dwarf_entry (const dwarf_entry &entry)
  {
    m_kind = entry.m_kind;
  }

  virtual ~dwarf_entry (void) = default;

  /* Checker method to determine the objects base classes.

     Note that every class is a base class to itself.  */
  virtual bool is_entry (entry_kind kind) const
  {
    return m_kind == kind || kind == Entry;
  }

protected:
  /* Kind of the underlying class.  */
  entry_kind m_kind = Entry;
 };

/* Value entry found on a DWARF expression evaluation stack.  */

class dwarf_value : public dwarf_entry
{
public:
  dwarf_value (const gdb_byte* contents, size_t size,
               struct type *type) : dwarf_entry {}
  {
    size_t type_len = TYPE_LENGTH (type);
    m_contents.reset ((gdb_byte *) xzalloc (type_len));

    if (type_len < size)
      size = type_len;

    memcpy (m_contents.get (), contents, size);
    m_type = type;
    m_kind = Value;
  }

  dwarf_value (ULONGEST value, struct type *type) : dwarf_entry {}
  {
    m_contents.reset ((gdb_byte *) xzalloc (TYPE_LENGTH (type)));

    pack_unsigned_long (m_contents.get (), type, value);
    m_type = type;
    m_kind = Value;
  }

  dwarf_value (LONGEST value, struct type *type) : dwarf_entry {}
  {
    m_contents.reset ((gdb_byte *) xzalloc (TYPE_LENGTH (type)));

    pack_long (m_contents.get (), type, value);
    m_type = type;
    m_kind = Value;
  }

  dwarf_value (const dwarf_value &value) : dwarf_entry {value}
  {
    struct type *type = value.m_type;
    size_t type_len = TYPE_LENGTH (type);

    m_contents.reset ((gdb_byte *) xzalloc (type_len));

    memcpy (m_contents.get (), value.m_contents.get (), type_len);
    m_type = type;
  }

  virtual ~dwarf_value (void) = default;

  const gdb_byte* get_contents (void) const
  {
    return m_contents.get ();
  }

  struct type* get_type (void) const
  {
    return m_type;
  }
  LONGEST to_long () const
  {
    return unpack_long (m_type, m_contents.get ());
  }

private:
  /* Value contents as a stream of bytes in a target byte-order.  */
  gdb::unique_xmalloc_ptr<gdb_byte> m_contents;

  /* Type of the value held by the entry.  */
  struct type *m_type;
};

/* Location description entry found on a DWARF expression evaluation stack.

   Types of location descirbed can be: register location, memory location,
   implicit location, implicit pointer location, undefined location and
   composite location (made out of any of the location type including
   another composite location).  */

class dwarf_location : public dwarf_entry
{
public:
  /* Not expected to be called on it's own.  */
  dwarf_location (LONGEST offset = 0, LONGEST bit_suboffset = 0) :
                  dwarf_entry {}, m_initialised (true)
  {
    m_offset = offset;
    m_offset += bit_suboffset / 8;
    m_bit_suboffset = bit_suboffset % 8;
    m_kind = Location;
  }

  dwarf_location (const dwarf_location &location) :
                  dwarf_entry {location}, m_offset (location.m_offset),
                  m_bit_suboffset (location.m_bit_suboffset),
                  m_initialised (location.m_initialised) {}

  virtual ~dwarf_location (void) = default;

  LONGEST get_offset (void) const
  {
    return m_offset;
  };

  LONGEST get_bit_suboffset (void) const
  {
    return m_bit_suboffset;
  };

  void add_bit_offset (LONGEST bit_offset)
  {
    LONGEST bit_total_offset = m_bit_suboffset + bit_offset;
    if (bit_total_offset < 0)
      {
        bit_total_offset = 0 - bit_total_offset;
        m_offset -= bit_total_offset / 8;
        m_bit_suboffset -= bit_total_offset % 8;
      }
    else
      {
        m_offset += bit_total_offset / 8;
        m_bit_suboffset += bit_total_offset % 8;
      }
  };

  void set_initialised (bool initialised)
  {
    m_initialised = initialised;
  };

  bool is_initialised (void) const
  {
    return m_initialised;
  };

  virtual bool is_entry (entry_kind kind) const override
  {
    return dwarf_entry::is_entry (kind) || kind == Location;
  }

private:
  /* Byte offset into the location.  */
  LONGEST m_offset;

  /* Bit suboffset of the last byte.  */
  LONGEST m_bit_suboffset;

  /* Indicator if a location is initialised.

     Used for non-standard DW_OP_GNU_uninit operation.  */
  bool m_initialised;
};

/* Undefined location description entry. This is a special location
   description type that describes the location description that is 
   not known.  */

class dwarf_undefined : public dwarf_location
{
public:
  dwarf_undefined (LONGEST offset = 0, LONGEST bit_suboffset = 0) :
                   dwarf_location {offset, bit_suboffset}
                   {m_kind = Undefined;}
  dwarf_undefined (const dwarf_undefined &undefined_entry) :
                   dwarf_location {undefined_entry} {}

  virtual ~dwarf_undefined (void) = default;
};

/* Memory location description entry describing a location in an address
   space M_ADDRESS_SPACE, where the offset into the location is an offset
   into the whole addressable region of that address space.  */

class dwarf_memory : public dwarf_location
{
public:
  dwarf_memory (LONGEST offset, LONGEST bit_suboffset = 0,
                unsigned int address_space = 0, bool stack = false) :
                dwarf_location {offset, bit_suboffset},
                m_address_space (address_space), m_stack (stack)
                {m_kind = Memory;}

  dwarf_memory (const dwarf_memory &memory_entry) :
                dwarf_location {memory_entry},
                m_address_space (memory_entry.m_address_space),
                m_stack (memory_entry.m_stack) {}

  virtual ~dwarf_memory (void) = default;

  unsigned int get_address_space (void) const
  {
    return m_address_space;
  };

  void set_address_space (unsigned int address_space)
  {
    m_address_space = address_space;
  };


  bool in_stack (void) const
  {
    return m_stack;
  };

  void set_stack (bool stack)
  {
    m_stack = stack;
  };
private:
  /* Address space referenced by the entry.  */
  unsigned int m_address_space;

  /* Indicator if the location belongs to a stack region of the memory.  */
  bool m_stack;
};

/* Register location description entry.  */

class dwarf_register : public dwarf_location
{
public:
  dwarf_register (unsigned int regnum, bool on_entry = false,
                  LONGEST offset = 0, LONGEST bit_suboffset = 0) :
                  dwarf_location {offset, bit_suboffset},
                  m_regnum (regnum), m_on_entry (on_entry)
                  {m_kind = Register;}

  dwarf_register (const dwarf_register &register_entry) :
                  dwarf_location {register_entry},
                  m_regnum (register_entry.m_regnum),
                  m_on_entry (register_entry.m_on_entry) {}

  virtual ~dwarf_register (void) = default;

  unsigned int get_regnum (void) const
  {
    return m_regnum;
  };

  bool is_on_entry (void) const
  {
    return m_on_entry;
  };

private:
  /* DWARF register number.  */
  unsigned int m_regnum;

  /* Indicates that a location is on the frame
     entry described in CFI of the previous frame.  */
  bool m_on_entry;
};

/* Implicit location description entry.

   Describes a location description not found on a target architecture
   but instead saved in gdb allocated buffer.  */

class dwarf_implicit : public dwarf_location
{
public:

  dwarf_implicit (const gdb_byte* contents, size_t size) : dwarf_location {}
  {
    m_contents.reset ((gdb_byte *) xzalloc (size));

    memcpy (m_contents.get (), contents, size);
    m_size = size;
    m_kind = Implicit;
  }

  dwarf_implicit (const dwarf_implicit &implicit_entry) :
                  dwarf_location {implicit_entry}
  {
    size_t size = implicit_entry.m_size;
    m_contents.reset ((gdb_byte *) xzalloc (size));

    memcpy (m_contents.get (), implicit_entry.m_contents.get (), size);
    m_size = size;
  }

  virtual ~dwarf_implicit (void) = default;

  const gdb_byte* get_contents (void) const
  {
    return m_contents.get ();
  }

  size_t get_size (void) const
  {
    return m_size;
  }

private:
  /* Implicit location contents as a stream of bytes in target byte-order.  */
  gdb::unique_xmalloc_ptr<gdb_byte> m_contents;

  /* Contents byte stream size.  */
  size_t m_size;
};

/* Implicit pointer location description entry.  */

class dwarf_implicit_pointer : public dwarf_location
{
public:
  dwarf_implicit_pointer (sect_offset die_offset,
                          LONGEST offset, LONGEST bit_suboffset = 0) :
                          dwarf_location {offset, bit_suboffset},
                          m_die_offset (die_offset)
                          {m_kind = ImplicitPtr;}

  dwarf_implicit_pointer (const dwarf_implicit_pointer &implicit_ptr_entry) :
                          dwarf_location {implicit_ptr_entry},
                          m_die_offset(implicit_ptr_entry.m_die_offset) {}

  virtual ~dwarf_implicit_pointer (void) = default;

  sect_offset get_die_offset (void) const
  {
    return m_die_offset;
  }

private:
  /* DWARF die offset pointed by the implicit pointer.  */
  sect_offset m_die_offset;
};

/* Composite location description entry.  */

class dwarf_composite : public dwarf_location
{
public:
  dwarf_composite (LONGEST offset = 0, LONGEST bit_suboffset = 0) :
                   dwarf_location {offset, bit_suboffset}
                   {m_kind = Composite;}

  dwarf_composite (const dwarf_composite &composite_entry) :
                   dwarf_location {composite_entry}
  {
    /* We do a shallow copy of the pieces because they are not
       not expected to be modified after they are already formed.  */
    for (unsigned int i = 0; i < composite_entry.m_pieces.size (); i++)
      {
        dwarf_location* location = composite_entry.m_pieces[i].second;

        location->incref ();
        m_pieces.emplace_back (composite_entry.m_pieces[i].first, location);
      }

    m_completed = composite_entry.m_completed;
  }

  /* Composite location gets detached from it's factory object for the
     purpose of lval_computed resolution, which means that it needs
     to take care of garbage collection of it's pieces.  */

  ~dwarf_composite ()
  {
    for (unsigned int i = 0; i < m_pieces.size (); i++)
      {
        dwarf_location* location = m_pieces[i].second;

        location->decref ();

        if (location->refcount () == 0)
      delete location;
      }
  }

  void
  add_piece (dwarf_location* location, ULONGEST bit_size)
  {
    gdb_assert (location != nullptr);
    location->incref ();
    m_pieces.emplace_back (bit_size, location);
  }

  const dwarf_location*
  get_piece_at (unsigned int index) const
  {
    gdb_assert (index < m_pieces.size ());
    return m_pieces[index].second;
  }

  ULONGEST
  get_bit_size_at (unsigned int index) const
  {
    gdb_assert (index < m_pieces.size ());
    return m_pieces[index].first;
  }

  size_t get_pieces_num (void) const
  {
    return m_pieces.size ();
  }

  void set_completed (bool completed)
  {
    m_completed = completed;
  };

  bool is_completed (void) const
  {
    return m_completed;
  };

private:
  /* Vector of location description pieces and their respective sizes.  */
  std::vector<std::pair<ULONGEST, dwarf_location *>> m_pieces;

  bool m_completed = false;
 };

/* Cookie for gdbarch data.  */

static struct gdbarch_data *dwarf_arch_cookie;

/* This holds gdbarch-specific types used by the DWARF expression
   evaluator.  See comments in execute_stack_op.  */

struct dwarf_gdbarch_types
{
  struct type *dw_types[3];
};

/* Factory class for creating and lifetime managment of
   all DWARF entries found on a DWARF evaluation stack.  */

class dwarf_entry_factory
{
public:
  dwarf_entry_factory (void) = default;
  ~dwarf_entry_factory (void);

  /* Creates a value entry of a given TYPE and copies a SIZE
     number of bytes from the CONTENTS byte stream to the entry.  */

  dwarf_value *create_value (const gdb_byte* contents, size_t size,
                             struct type *type);

  /* Creates a value entry of a TYPE type and copies the NUM
     value to it's contents byte stream.  */

  dwarf_value *create_value (ULONGEST num, struct type *type);

  /* Creates a value entry of a TYPE type and copies the NUM
     value to it's contents byte stream.  */

  dwarf_value *create_value (LONGEST num, struct type *type);

  /* Creates an undefined location description entry.  */

  dwarf_undefined *create_undefined (void);

  /* Creates a memory location description entry.  */

  dwarf_memory *create_memory (LONGEST offset, LONGEST bit_suboffset = 0,
                               unsigned int addr_space = 0,
                               bool stack = false);

  /* Creates a register location description entry.  */

  dwarf_register *create_register (unsigned int regnum, bool on_entry = false,
                                   LONGEST offset = 0,
                                   LONGEST bit_suboffset = 0);

  /* Creates an implicit location description entry and copies SIZE number
     of bytes from the CONTENTS byte stream to the to the location.  */

  dwarf_implicit *create_implicit (const gdb_byte* content, size_t size);

  /* Creates an implicit pointer location description entry.  */

  dwarf_implicit_pointer *create_implicit_pointer (sect_offset die_offset,
                                                   LONGEST offset,
                                                   LONGEST bit_suboffset = 0);

  /* Creates a composite location description entry.  */

  dwarf_composite *create_composite (LONGEST offset = 0,
                                     LONGEST bit_suboffset = 0);

  /* Creates a deep copy of the DWARF ENTRY.  */
  dwarf_entry *copy_entry (dwarf_entry *entry);

  /* Converts an entry to a location description entry. If the entry
     is a location description entry a dynamic cast is applied.

     In a case of a value entry, the value is implicitly
     converted to a memory location description entry.  */

  dwarf_location *entry_to_location (dwarf_entry *entry);

  /* Converts an entry to a value entry. If the entry is a value
     entry a dynamic cast is applied.

     In the case of a location description entry it is implicitly
     converted to value entry of a DEFAULT_TYPE type.

     Note that only memory location description entry to value
     entry conversion is currently supported. */

  dwarf_value *entry_to_value (dwarf_entry *entry, struct type *default_type);

  /* Converts a value entry to the matching struct value representation of
     a given TYPE. Where OFFSET defines an offset into the value contents.  */

  struct value *value_to_gdb_value (const dwarf_value *value,
                                    struct type *type, LONGEST offset = 0);

  /* Executes OP operation between ARG1 and ARG2 and returns
     a new value entry containing a result of that operation.  */

  dwarf_value *value_binary_op (const dwarf_value *arg1,
                                const dwarf_value *arg2, enum exp_opcode op);

  /* Executes a negation operation on ARG and returns a new
     value entry containing the result of that operation.  */

  dwarf_value *value_negation_op (const dwarf_value *arg);

  /* Executes a complement operation on ARG and returns
     a new value entry containing the result of that
     operation.  */

  dwarf_value *value_complement_op (const dwarf_value *arg);

  /* Executes a cast operation on ARG and returns a new
     value entry containing the result of that operation.  */

  dwarf_value *value_cast_op (const dwarf_value *arg, struct type *type);

private:
  /* Records entry for garbage collection.  */
  void record_entry (dwarf_entry *entry);

  /* List of all entries created by the factory.  */
  std::vector<dwarf_entry *> dwarf_entries;
 };

 /* Allocate and fill in dwarf_gdbarch_types for an arch.  */

static void *
dwarf_gdbarch_types_init (struct gdbarch *gdbarch)
{
  struct dwarf_gdbarch_types *types
    = GDBARCH_OBSTACK_ZALLOC (gdbarch, struct dwarf_gdbarch_types);

  /* The types themselves are lazily initialized.  */

  return types;
}

/* Require that TYPE be an integral type; throw an exception if not.  */

static void
dwarf_require_integral (struct type *type)
{
  if (type->code () != TYPE_CODE_INT
      && type->code () != TYPE_CODE_CHAR
      && type->code () != TYPE_CODE_BOOL)
    error (_("integral type expected in DWARF expression"));
}

/* Return the unsigned form of TYPE.  TYPE is necessarily an integral
   type.  */

static struct type *
get_unsigned_type (struct gdbarch *gdbarch, struct type *type)
{
  switch (TYPE_LENGTH (type))
    {
    case 1:
      return builtin_type (gdbarch)->builtin_uint8;
    case 2:
      return builtin_type (gdbarch)->builtin_uint16;
    case 4:
      return builtin_type (gdbarch)->builtin_uint32;
    case 8:
      return builtin_type (gdbarch)->builtin_uint64;
    default:
      error (_("no unsigned variant found for type, while evaluating "
	       "DWARF expression"));
    }
}

/* Return the signed form of TYPE.  TYPE is necessarily an integral
   type.  */

static struct type *
get_signed_type (struct gdbarch *gdbarch, struct type *type)
{
  switch (TYPE_LENGTH (type))
    {
    case 1:
      return builtin_type (gdbarch)->builtin_int8;
    case 2:
      return builtin_type (gdbarch)->builtin_int16;
    case 4:
      return builtin_type (gdbarch)->builtin_int32;
    case 8:
      return builtin_type (gdbarch)->builtin_int64;
    default:
      error (_("no signed variant found for type, while evaluating "
	       "DWARF expression"));
    }
}

/* Throws an exception for accessing outside the
   bounds of an object reference by a synthetic pointer.  */

static void
invalid_synthetic_pointer (void)
{
  error (_("access outside bounds of object "
         "referenced via synthetic pointer"));
}

/* Throws an exception about the invalid DWARF expression.  */

static void
ill_formed_expression (void)
{
  error (_("Ill formed DWARF expression"));
}

/* Reads a REG register contents in a given FRAME context.

   Data read is offsetted by the OFFSET, and number of bytes read is defined
   by the LENGTH. The data is then copied into a caller managed BUF buffer.

   If the register is optimized out or unavailable for the given FRAME,
   the OPTIMIZED and UNAVAILABLE outputs are set.  */

static void
read_from_register (struct frame_info *frame, int reg,
                    CORE_ADDR offset, int length, gdb_byte *buf,
                    int *optimized, int *unavailable)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);
  int regsize = register_size (gdbarch, reg);
  int numregs = gdbarch_num_cooked_regs (gdbarch);
  gdb::byte_vector temp_buf (regsize);
  enum lval_type lval;
  CORE_ADDR address;
  int realnum;

  /* If a register is wholly inside the OFFSET, skip it.  */
  if (frame == NULL || !regsize
      || (offset + length) > regsize || numregs < reg)
    {
      (*optimized) = 0;
      (*unavailable) = 1;
      return;
    }

  frame_register (frame, reg, optimized, unavailable,
                  &lval, &address, &realnum, temp_buf.data ());

  if (!(*optimized) && !(*unavailable))
     memcpy (buf, (char *) temp_buf.data () + offset, length);

  return;
}

/* Writes a REG register contents in a given FRAME context.

   Data writen is offsetted by the OFFSET, and number of bytes writen is
   defined by the LENGTH. The data is copied from a caller managed BUF buffer.

   If the register is optimized out or unavailable for the given FRAME,
   the OPTIMIZED and UNAVAILABLE outputs are set. */

static void
write_to_register (struct frame_info *frame, int reg,
                   CORE_ADDR offset, int length, gdb_byte *buf,
                   int *optimized, int *unavailable)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);
  int regsize = register_size (gdbarch, reg);
  int numregs = gdbarch_num_cooked_regs (gdbarch);
  gdb::byte_vector temp_buf (regsize);
  enum lval_type lval;
  CORE_ADDR address;
  int realnum;

  /* If a register is wholly inside of OFFSET, skip it.  */
  if (frame == NULL || !regsize
     || (offset + length) > regsize || numregs < reg)
    {
      (*optimized) = 0;
      (*unavailable) = 1;
      return;
    }

  frame_register (frame, reg, optimized, unavailable,
                  &lval, &address, &realnum, temp_buf.data ());

    if (!(*optimized) && !(*unavailable))
  {
    memcpy ((char *) temp_buf.data () + offset, buf, length);

    put_frame_register (frame, reg, temp_buf.data ());
  }

  return;
}

/* Helper for read_from_memory and write_from_memory.  */

static void
xfer_from_memory (CORE_ADDR address, gdb_byte *readbuf,
		  const gdb_byte *writebuf,
                  size_t length, bool stack, int *unavailable)
{
  (*unavailable) = 0;

  enum target_object object
    = stack ? TARGET_OBJECT_STACK_MEMORY : TARGET_OBJECT_MEMORY;

  ULONGEST xfered_total = 0;

  while (xfered_total < length)
    {
      ULONGEST xfered_partial;

      enum target_xfer_status status
	= target_xfer_partial (current_top_target (), object, NULL,
			       (readbuf != nullptr
				? readbuf + xfered_total
				: nullptr),
			       (writebuf != nullptr
				? writebuf + xfered_total
				: nullptr),
			       address + xfered_total, length - xfered_total,
			       &xfered_partial);

      if (status == TARGET_XFER_OK)
	{
	  xfered_total += xfered_partial;
	  QUIT;
	}
      else if (status == TARGET_XFER_UNAVAILABLE)
	{
	  (*unavailable) = 1;
	  return;
	}
      else if (status == TARGET_XFER_EOF)
	memory_error (TARGET_XFER_E_IO, address + xfered_total);
      else
	memory_error (status, address + xfered_total);
    }
}

/* Reads a contents from a given memory ADDRESS.

   Data read is copied to a caller managed BUF buffer, where a number of bytes
   read is defined by the LENGTH. If the memory address specified belongs to
   a memory region representing a target stack, the STACK argument can be
   used to send that information to the read memory target hook.

   If the memory address is unavailable, the UNAVAILABLE output is set.  */

static void
read_from_memory (CORE_ADDR address, gdb_byte *buffer,
                  size_t length, bool stack, int* unavailable)
{
  xfer_from_memory (address, buffer, nullptr, length, stack, unavailable);
}

/* Write a contents from a given memory ADDRESS.

   Data writen is copied from a caller managed BUF buffer, where a number
   of bytes writen is defined by the LENGTH. If the memory address specified
   belongs to a memory region representing a target stack, the STACK argument
   can be used to send that information to the write memory target hook.

   If the memory address is unavailable, the UNAVAILABLE output is set.  */

static void
write_to_memory (CORE_ADDR address, const gdb_byte *buffer,
                 size_t length, bool stack, int *unavailable)
{
  xfer_from_memory (address, nullptr, buffer, length, stack, unavailable);
}

/* Return the number of bytes overlapping a contiguous chunk of N_BITS
   bits whose first bit is located at bit offset START.  */

static size_t
bits_to_bytes (ULONGEST start, ULONGEST n_bits)
{
  return (start % 8 + n_bits + 7) / 8;
}

/* Reads a contents from a location specified by the DWARF location
   description entry LOCATION.

   The read operation is performed in a FRAME context. Number of bits read is
   defined by the BIT_SIZE and the data read is copied to a caller managed BUF
   buffer. BIG_ENDIAN defines an endianes of the target, BITS_TO_SKIP is a bit
   offset into a location and BUF_BIT_OFFSET defines BUF buffers bit offset.

   Note that some location types can be read without a FRAME context.

   If the location is optimized out or unavailable, the OPTIMIZED and
   UNAVAILABLE outputs are set.  */

static void
read_from_location (const dwarf_location* location, struct frame_info *frame,
                    LONGEST bits_to_skip, gdb_byte *buf, int buf_bit_offset,
                    size_t bit_size, bool big_endian,
                    int* optimized, int* unavailable)
{
  LONGEST offset = location->get_offset ();
  LONGEST bit_suboffset = location->get_bit_suboffset ();
  gdb::byte_vector temp_buf;

  /* Read from an undefined locations is always marked as optimized.  */
  if (location->is_entry (dwarf_entry::Undefined))
    {
      (*unavailable) = 0;
      (*optimized) = 1;
    }
  else if (location->is_entry (dwarf_entry::Register))
    {
      const dwarf_register *register_entry
          = static_cast<const dwarf_register*> (location);
      struct gdbarch *arch = get_frame_arch (frame);
      int reg = dwarf_reg_to_regnum_or_error (arch,
                                              register_entry->get_regnum ());
      ULONGEST reg_bits = 8 * register_size (arch, reg);
      LONGEST this_size;

      if (big_endian)
    bits_to_skip += reg_bits - (offset * 8 + bit_suboffset + bit_size);
      else
    bits_to_skip += offset * 8 + bit_suboffset;

      this_size = bits_to_bytes (bits_to_skip, bit_size);
      temp_buf.resize (this_size);

      if (register_entry->is_on_entry ())
        frame = get_prev_frame_always (frame);

      if (frame == NULL)
        internal_error (__FILE__, __LINE__, _("invalid frame information"));

      /* Can only read from register on a byte granularity so
         additional buffer is required.  */
      read_from_register (frame, reg, bits_to_skip / 8, this_size,
                          temp_buf.data (), optimized, unavailable);

      /* Only copy data if valid.  */
      if (!(*optimized) && !(*unavailable))
    copy_bitwise (buf, buf_bit_offset, temp_buf.data (),
                  bits_to_skip % 8, bit_size, big_endian);
    }
  else if (location->is_entry (dwarf_entry::Memory))
    {
      const dwarf_memory *memory_entry
          = static_cast<const dwarf_memory *> (location);
      CORE_ADDR start_address = offset + (bit_suboffset + bits_to_skip) / 8;
      unsigned int address_space = memory_entry->get_address_space ();

      start_address = aspace_address_to_flat_address (start_address,
                                                      address_space);

      (*optimized) = 0;
      bits_to_skip += bit_suboffset;

      if (bits_to_skip % 8 == 0 && bit_size % 8 == 0
          && buf_bit_offset % 8 == 0)
    {
      /* Everything is byte-aligned; no buffer needed.  */
      read_from_memory (start_address, buf + buf_bit_offset / 8, bit_size / 8,
                        memory_entry->in_stack (), unavailable);
    }
      else
    {
      LONGEST this_size = bits_to_bytes (bits_to_skip, bit_size);
      temp_buf.resize (this_size);

      /* Can only read from memory on a byte granularity
         so additional buffer is required.  */
      read_from_memory (start_address, temp_buf.data (), this_size,
                        memory_entry->in_stack (), unavailable);

      if (!(*unavailable))
        copy_bitwise (buf, buf_bit_offset, temp_buf.data (),
                      bits_to_skip % 8, bit_size, big_endian);
    }
    }
  else if (location->is_entry (dwarf_entry::Implicit))
    {
      const dwarf_implicit *implicit_entry
          = static_cast<const dwarf_implicit*> (location);

      ULONGEST literal_bit_size = 8 * implicit_entry->get_size ();

      (*optimized) = 0;
      (*unavailable) = 0;

      /* Cut off at the end of the implicit value.  */
      bits_to_skip += offset * 8 + bit_suboffset ;
      if (bits_to_skip >= literal_bit_size)
    {
      (*unavailable) = 1;
      return;
    }

      if (bit_size > literal_bit_size - bits_to_skip)
    bit_size = literal_bit_size - bits_to_skip;

      copy_bitwise (buf, buf_bit_offset, implicit_entry->get_contents (),
                    bits_to_skip, bit_size, big_endian);
    }
  else if (location->is_entry (dwarf_entry::ImplicitPtr))
    {
      /* This case should be handled on a higher levels. */
      gdb_assert (false);
    }
  else if (location->is_entry (dwarf_entry::Composite))
    {
      const dwarf_composite *composite_entry
          = static_cast<const dwarf_composite *> (location);
      unsigned int pieces_num = composite_entry->get_pieces_num ();
      unsigned int i;

      if (!composite_entry->is_completed ())
        ill_formed_expression ();

      bits_to_skip += offset * 8 + bit_suboffset ;

      for (i = 0; i < pieces_num; i++)
    {
      LONGEST piece_bit_size = composite_entry->get_bit_size_at (i);

      if (bits_to_skip < piece_bit_size)
        break;

      bits_to_skip -= piece_bit_size;
    }

      for (; i < pieces_num; i++)
    {
      LONGEST piece_bit_size = composite_entry->get_bit_size_at (i);

      if (piece_bit_size > bit_size)
        piece_bit_size = bit_size;

      read_from_location (composite_entry->get_piece_at (i), frame,
                          bits_to_skip, buf, buf_bit_offset, piece_bit_size,
                          big_endian, optimized, unavailable);

      if (bit_size == piece_bit_size || (*optimized) || (*unavailable))
        break;

      buf_bit_offset += piece_bit_size;
      bit_size -= piece_bit_size;
    }
    }
  else
    internal_error (__FILE__, __LINE__, _("invalid location type"));
}

/* Write a contents to a location specified by the DWARF location description
   entry LOCATION.

   The write operation is performed in a FRAME context. Number of bits writen
   is defined by the BIT_SIZE and the data writen is copied from a caller
   managed BUF buffer. BIG_ENDIAN defines an endianes of the target,
   BITS_TO_SKIP is a bit offset into a location and BUF_BIT_OFFSET defines BUF
   buffers bit offset.

   Note that some location types can be writen without a FRAME context.

   If the location is optimized out or unavailable, the OPTIMIZED and
   UNAVAILABLE outputs are set.  */

static void
write_to_location (const dwarf_location* location, struct frame_info *frame,
                   LONGEST bits_to_skip, const gdb_byte *buf,
                   int buf_bit_offset, size_t bit_size,
                   bool big_endian, int* optimized, int* unavailable)
{
  LONGEST offset = location->get_offset();
  LONGEST bit_suboffset = location->get_bit_suboffset();
  gdb::byte_vector temp_buf;

  /* Write to an undefined locations is always marked as optimized.  */
  if (location->is_entry (dwarf_entry::Undefined))
    {
      (*unavailable) = 0;
      (*optimized) = 1;
    }
  else if (location->is_entry (dwarf_entry::Register))
    {
      const dwarf_register *register_entry
          = static_cast<const dwarf_register*> (location);
      struct gdbarch *arch = get_frame_arch (frame);
      int gdb_regnum
          = dwarf_reg_to_regnum_or_error (arch, register_entry->get_regnum ());
      ULONGEST reg_bits = 8 * register_size (arch, gdb_regnum);
      LONGEST this_size;

      if (register_entry->is_on_entry ())
        frame = get_prev_frame_always (frame);

      if (frame == NULL)
        internal_error (__FILE__, __LINE__, _("invalid frame information"));

      if (big_endian)
    bits_to_skip += reg_bits - (offset * 8 + bit_suboffset + bit_size);
      else
    bits_to_skip += offset * 8 + bit_suboffset;

      this_size = bits_to_bytes (bits_to_skip, bit_size);
      temp_buf.resize (this_size);
       /* Write mode.  */
      if (bits_to_skip % 8 != 0 || bit_size % 8 != 0)
    {
      /* Contents is copied non-byte-aligned into the register.
         Need some bits from original register value.  */
      read_from_register (frame, gdb_regnum, bits_to_skip / 8, this_size,
                          temp_buf.data (), optimized, unavailable);
    }

      copy_bitwise (temp_buf.data (), bits_to_skip % 8, buf,
                    buf_bit_offset, bit_size, big_endian);

      write_to_register (frame, gdb_regnum, bits_to_skip / 8, this_size,
                         temp_buf.data (), optimized, unavailable);
    }
  else if (location->is_entry (dwarf_entry::Memory))
    {
      const dwarf_memory *memory_entry
          = static_cast<const dwarf_memory *> (location);
      CORE_ADDR start_address
          = (offset * 8 + bit_suboffset + bits_to_skip) / 8;
      unsigned int address_space = memory_entry->get_address_space ();
      LONGEST this_size;

      bits_to_skip += bit_suboffset;
      (*optimized) = 0;

      start_address = aspace_address_to_flat_address (start_address,
                                                      address_space);

      if (bits_to_skip % 8 == 0 && bit_size % 8 == 0
          && buf_bit_offset % 8 == 0)
    {
      /* Everything is byte-aligned; no buffer needed.  */
      write_to_memory (start_address, (buf + buf_bit_offset / 8), bit_size / 8,
                       memory_entry->in_stack (), unavailable);
    }

      this_size = bits_to_bytes (bits_to_skip, bit_size);
      temp_buf.resize (this_size);

      if (bits_to_skip % 8 != 0 || bit_size % 8 != 0)
    {
      if (this_size <= 8)
        {
          /* Perform a single read for small sizes.  */
          read_memory (start_address, temp_buf.data (), this_size);
        }
      else
        {
          /* Only the first and last bytes can possibly have
             any bits reused.  */
          read_memory (start_address, temp_buf.data (), 1);
          read_memory (start_address + this_size - 1,
                       &temp_buf[this_size - 1], 1);
        }
    }

      copy_bitwise (temp_buf.data (), bits_to_skip % 8, buf,
                    buf_bit_offset, bit_size, big_endian);

      write_to_memory (start_address, temp_buf.data (), this_size,
                       memory_entry->in_stack (), unavailable);
    }
  else if (location->is_entry (dwarf_entry::Implicit))
    {
      (*optimized) = 1;
      (*unavailable) = 0;
    }
  else if (location->is_entry (dwarf_entry::ImplicitPtr))
    {
      (*optimized) = 1;
      (*unavailable) = 0;
    }
  else if (location->is_entry (dwarf_entry::Composite))
    {
      const dwarf_composite *composite_entry
          = static_cast<const dwarf_composite*> (location);
      unsigned int pieces_num = composite_entry->get_pieces_num ();
      unsigned int i;

      if (!composite_entry->is_completed ())
        ill_formed_expression ();

      bits_to_skip += offset * 8 + bit_suboffset ;

      for (i = 0; i < pieces_num; i++)
    {
      LONGEST piece_bit_size = composite_entry->get_bit_size_at (i);

      if (bits_to_skip < piece_bit_size)
        break;

      bits_to_skip -= piece_bit_size;
    }

      for (; i < pieces_num; i++)
    {
      LONGEST piece_bit_size = composite_entry->get_bit_size_at (i);

      if (piece_bit_size > bit_size)
        piece_bit_size = bit_size;

      write_to_location (composite_entry->get_piece_at (i), frame,
                         bits_to_skip, buf, buf_bit_offset, piece_bit_size,
                         big_endian, optimized, unavailable);

      if (bit_size == piece_bit_size || (*optimized) || (*unavailable))
        break;

      buf_bit_offset += piece_bit_size;
      bit_size -= piece_bit_size;
    }
    }
  else
    internal_error (__FILE__, __LINE__, _("invalid location type"));
}

dwarf_entry_factory::~dwarf_entry_factory ()
{
  for (unsigned int i = 0; i < dwarf_entries.size (); i++)
  {
    dwarf_entry* entry = dwarf_entries[i];

    entry->decref ();

    if (entry->refcount () == 0)
      delete entry;
  }
}

void
dwarf_entry_factory::record_entry (dwarf_entry *entry)
{
  entry->incref ();
  dwarf_entries.push_back (entry);
}

dwarf_value *
dwarf_entry_factory::create_value (const gdb_byte* content, size_t size,
                                   struct type *type)
{
  dwarf_value *value = new dwarf_value (content, size, type);
  record_entry (value);
  return value;
}

dwarf_value *
dwarf_entry_factory::create_value (ULONGEST num, struct type *type)
{
  dwarf_value *value = new dwarf_value (num, type);
  record_entry (value);
  return value;
}

dwarf_value *
dwarf_entry_factory::create_value (LONGEST num, struct type *type)
{
  dwarf_value *value = new dwarf_value (num, type);
  record_entry (value);
  return value;
}

dwarf_undefined *
dwarf_entry_factory::create_undefined ()
{
  dwarf_undefined *undefined_entry = new dwarf_undefined ();
  record_entry (undefined_entry);
  return undefined_entry;
}

dwarf_memory *
dwarf_entry_factory::create_memory (LONGEST offset, LONGEST bit_suboffset,
                                    unsigned int address_space, bool stack)
{
  dwarf_memory *memory_entry
      = new dwarf_memory (offset, bit_suboffset, address_space, stack);
  record_entry (memory_entry);
  return memory_entry;
}

dwarf_register *
dwarf_entry_factory::create_register (unsigned int regnum, bool on_entry,
                                      LONGEST offset, LONGEST bit_suboffset)
{
  dwarf_register *register_entry
      = new dwarf_register (regnum, offset, bit_suboffset);
  record_entry (register_entry);
  return register_entry;
}

dwarf_implicit *
dwarf_entry_factory::create_implicit (const gdb_byte* content, size_t size)
{
  dwarf_implicit *implicit_entry = new dwarf_implicit (content, size);
  record_entry (implicit_entry);
  return implicit_entry;
}

dwarf_implicit_pointer *
dwarf_entry_factory::create_implicit_pointer (sect_offset die_offset,
                                              LONGEST offset,
                                              LONGEST bit_suboffset)
{
  dwarf_implicit_pointer *implicit_pointer_entry
      = new dwarf_implicit_pointer (die_offset, offset, bit_suboffset);
  record_entry (implicit_pointer_entry);
  return implicit_pointer_entry;
}

dwarf_composite *
dwarf_entry_factory::create_composite (LONGEST offset, LONGEST bit_suboffset)
{
  dwarf_composite *composite_entry
      = new dwarf_composite (offset, bit_suboffset);
  record_entry (composite_entry);
  return composite_entry;
}

dwarf_entry *dwarf_entry_factory::copy_entry (dwarf_entry *entry)
{
  dwarf_entry *entry_copy;

  if (entry->is_entry (dwarf_entry::Value))
    entry_copy = new dwarf_value (*static_cast<dwarf_value*> (entry));
  else if (entry->is_entry (dwarf_entry::Undefined))
    entry_copy = new dwarf_undefined (*static_cast<dwarf_undefined*> (entry));
  else if (entry->is_entry (dwarf_entry::Memory))
    entry_copy = new dwarf_memory (*static_cast<dwarf_memory*> (entry));
  else if (entry->is_entry (dwarf_entry::Register))
    entry_copy = new dwarf_register (*static_cast<dwarf_register*> (entry));
  else if (entry->is_entry (dwarf_entry::Implicit))
    entry_copy = new dwarf_implicit (*static_cast<dwarf_implicit*> (entry));
  else if (entry->is_entry (dwarf_entry::ImplicitPtr))
  {
    dwarf_implicit_pointer* implicit_pointer_entry
       = static_cast<dwarf_implicit_pointer*> (entry);
    entry_copy = new dwarf_implicit_pointer (*implicit_pointer_entry);
  }
  else if (entry->is_entry (dwarf_entry::Composite))
    entry_copy = new dwarf_composite (*static_cast<dwarf_composite*> (entry));
  else
    entry_copy = new dwarf_entry (*entry);

  record_entry (entry_copy);
  return entry_copy;
}

/* We only need to support dwarf_value to gdb struct value conversion
   here so that we can utilize the existing unary and binary operations
   on struct value's.
   
   We could implement them for the dwarf_value's but that would lead
   to code duplication with no real gain.  */
struct value *
dwarf_entry_factory::value_to_gdb_value (const dwarf_value *value,
                                         struct type *type, LONGEST offset)
{
  struct value *retval;
  size_t type_len = TYPE_LENGTH (type);

  if (offset + type_len > TYPE_LENGTH (value->get_type ()))
    invalid_synthetic_pointer ();

  retval = allocate_value (type);
  memcpy (value_contents_raw (retval),
          value->get_contents () + offset, type_len);

  return retval;
}

dwarf_location *
dwarf_entry_factory::entry_to_location (dwarf_entry *entry)
{
  struct type *type;
  struct gdbarch *gdbarch;
  LONGEST offset;
  dwarf_value *value;

  /* If the given entry is already a location,
     just send it back to the caller.  */
  if (entry->is_entry (dwarf_entry::Value))
    value = static_cast<dwarf_value*> (entry);
  else
    {
      gdb_assert (entry->is_entry (dwarf_entry::Location));
      return static_cast<dwarf_location*> (entry);
    }

  type = value->get_type ();
  gdbarch = get_type_arch (type);

  if (gdbarch_integer_to_address_p (gdbarch))
    offset = gdbarch_integer_to_address (gdbarch, type,
                                         value->get_contents ());

  offset = unpack_long (type, value->get_contents ());

  return create_memory (offset);
}

dwarf_value *
dwarf_entry_factory::entry_to_value (dwarf_entry *entry,
                                     struct type *default_type)
{
  dwarf_location *location;

  /* If the given entry is already a value,
     just send it back to the caller.  */
  if(entry->is_entry (dwarf_entry::Location))
    location = static_cast<dwarf_location *> (entry);
  else
    {
      gdb_assert (entry->is_entry (dwarf_entry::Value));
      return static_cast<dwarf_value *> (entry);
    }

  /* We only support memory location to value conversion at this point.
     It is hard to define how would that conversion work for other
     location types.  */
  if (!location->is_entry (dwarf_entry::Memory))
    ill_formed_expression ();

  if (static_cast<dwarf_memory *> (entry)->get_address_space ())
    ill_formed_expression ();

  return create_value (location->get_offset (), default_type);
}

/* We use the existing struct value operations to avoid code dupplication.

   Vector types are planned to be promoted to base type's in the future
   anyway which means that the subset we actually need from these operations
   is just going to grow.  */

dwarf_value *
dwarf_entry_factory::value_binary_op (const dwarf_value *arg1,
                                      const dwarf_value *arg2,
                                      enum exp_opcode op)
{
  struct value *arg1_value = value_to_gdb_value (arg1, arg1->get_type ());
  struct value *arg2_value = value_to_gdb_value (arg2, arg2->get_type ());
  struct value *result = value_binop (arg1_value, arg2_value, op);

  struct type *type = value_type (result);
  return create_value (value_contents_raw (result), TYPE_LENGTH (type), type);
}

dwarf_value *
dwarf_entry_factory::value_negation_op (const dwarf_value *arg)
{
  struct value *result
      = value_neg (value_to_gdb_value (arg, arg->get_type ()));
  struct type *type = value_type (result);
  return create_value (value_contents_raw (result), TYPE_LENGTH (type), type);
}

dwarf_value *
dwarf_entry_factory::value_complement_op (const dwarf_value *arg)
{
  struct value *result
      = value_complement (value_to_gdb_value (arg, arg->get_type ()));
  struct type *type = value_type (result);
  return create_value (value_contents_raw (result), TYPE_LENGTH (type), type);
}

dwarf_value *
dwarf_entry_factory::value_cast_op (const dwarf_value *arg, struct type *type)
{
  struct value *result
      = value_cast (type, value_to_gdb_value (arg, arg->get_type ()));
  return create_value (value_contents_raw (result), TYPE_LENGTH (type), type);
}

computed_closure::computed_closure (dwarf_entry *entry)
{
  entry->incref ();
  m_entry = entry;
}

computed_closure::~computed_closure (void)
{
  m_entry->decref ();

  if (m_entry->refcount () == 0)
    delete m_entry;
}

sect_offset
implicit_pointer_closure::get_die_offset (void)
{
  dwarf_entry *entry = this->get_entry ();

  if (!entry->is_entry (dwarf_entry::ImplicitPtr))
    internal_error (__FILE__, __LINE__, _("invalid location type"));

  return static_cast<dwarf_implicit_pointer *> (entry)->get_die_offset ();
}

LONGEST
implicit_pointer_closure::get_offset (void)
{
  dwarf_entry *entry = this->get_entry ();

  if (!entry->is_entry (dwarf_entry::ImplicitPtr))
    internal_error (__FILE__, __LINE__, _("invalid location type"));

  return static_cast<dwarf_implicit_pointer *> (entry)->get_offset ();
}

/* Create a new context for the expression evaluator.  */

dwarf_expr_context::dwarf_expr_context (dwarf2_per_objfile *per_objfile)
: gdbarch (NULL),
  addr_size (0),
  ref_addr_size (0),
  recursion_depth (0),
  max_recursion_depth (0x100),
  per_objfile (per_objfile)
{
  entry_factory = new dwarf_entry_factory();
}

dwarf_expr_context::~dwarf_expr_context ()
{
  delete entry_factory;
}

/* Push VALUE onto the stack.  */

void
dwarf_expr_context::push (dwarf_entry* entry)
{
  stack.emplace_back (entry);
}

/* Push VALUE onto the stack.  */

void
dwarf_expr_context::push_address (CORE_ADDR addr, bool in_stack_memory)
 {
  stack.emplace_back (entry_factory->create_memory (addr, 0, 0,
                                                    in_stack_memory));
 }

/* Pop the top item off of the stack.  */

void
dwarf_expr_context::pop ()
{
  if (stack.empty ())
    error (_("dwarf expression stack underflow"));

  stack.pop_back ();
}

/* Retrieve the N'th item on the stack.  */

dwarf_entry *
dwarf_expr_context::fetch (int n)
{
  if (stack.size () <= n)
     error (_("Asked for position %d of stack, "
	      "stack only has %zu elements on it."),
	    n, stack.size ());
  return stack[stack.size () - (1 + n)];
}

struct value *
dwarf_expr_context::deref_implicit_pointer (sect_offset die_offset,
                                            LONGEST offset, struct type *type)
{
  return value_one (type);
}

/* Return the type used for DWARF operations where the type is
   unspecified in the DWARF spec.  Only certain sizes are
   supported.  */

struct type *
dwarf_expr_context::address_type () const
{
  struct dwarf_gdbarch_types *types
    = (struct dwarf_gdbarch_types *) gdbarch_data (this->gdbarch,
                                                   dwarf_arch_cookie);
  int ndx;

  if (this->addr_size == 2)
    ndx = 0;
  else if (this->addr_size == 4)
    ndx = 1;
  else if (this->addr_size == 8)
    ndx = 2;
  else
    error (_("Unsupported address size in DWARF expressions: %d bits"),
           8 * this->addr_size);

  if (types->dw_types[ndx] == NULL)
    types->dw_types[ndx]
      = arch_integer_type (this->gdbarch,
                           8 * this->addr_size,
                           0, "<signed DWARF address type>");

  return types->dw_types[ndx];
}

dwarf_entry*
dwarf_expr_context::dwarf_entry_deref_reinterpret (dwarf_entry *entry,
                                                   struct type *type)
{
  /* Some targets read from a register in a special way depending
     on a type information.

     This is the only location for which the reinterpret makes sense.  */
  if (entry->is_entry (dwarf_entry::Register))
    {
      struct gdbarch *gdbarch;
      gdb::byte_vector temp_buf;
      int reg, regnum, optimized, unavailable;
      struct frame_info *frame = get_context_frame ();

      eval_needs_frame ();

      /* This is needed for symbol_needs functionality where the evaluator
         context is not fully present, so the operation needs to be faked.  */
      if (!context_can_deref ())
    return entry_factory->create_value ((LONGEST) 1, type);

      gdbarch = get_frame_arch (frame);
      regnum = static_cast<const dwarf_register *> (entry)->get_regnum ();
      reg = dwarf_reg_to_regnum_or_error (gdbarch, regnum);
      temp_buf.resize (TYPE_LENGTH (type));

      if (gdbarch_convert_register_p (gdbarch, regnum, type))
    {
      gdbarch_register_to_value (gdbarch, frame, reg, type, temp_buf.data (),
                                 &optimized, &unavailable);

      if (optimized)
        throw_error (OPTIMIZED_OUT_ERROR,
                     _("Can't do read-modify-write to "
                     "update bitfield; containing word "
                     "has been optimized out"));
      if (unavailable)
        throw_error (NOT_AVAILABLE_ERROR,
                     _("Can't dereference "
                     "update bitfield; containing word "
                     "is unavailable"));

      return entry_factory->create_value (temp_buf.data (),
                                          TYPE_LENGTH (type), type);
    }
    }

  return dwarf_entry_deref (entry, type);
}

dwarf_entry*
dwarf_expr_context::dwarf_entry_deref (dwarf_entry *entry,
                                       struct type *type, size_t size)
{
  int optimized, unavailable;
  gdb::byte_vector temp_buf;
  bool big_endian = type_byte_order (type) == BFD_ENDIAN_BIG;
  dwarf_location *location = entry_factory->entry_to_location (entry);
  struct frame_info *frame = get_context_frame ();

  if (location->is_entry (dwarf_entry::Register))
    eval_needs_frame ();

  /* This is needed for symbol_needs functionality where the evaluator
     context is not fully present, so the operation needs to be faked.  */
  if (!context_can_deref ())
    return entry_factory->create_value ((LONGEST) 1, type);

  size = size != 0 ? size : TYPE_LENGTH (type);

  if (size > TYPE_LENGTH (type))
    ill_formed_expression ();

  /* This is one of the places where both struct value and dwarf_entry
     classes need to cooperate so the struct value needs to be converted.
     Implicit pointers dereferencing is a special case of location
     dereferencing  */
  if (location->is_entry (dwarf_entry::ImplicitPtr))
    {
      sect_offset die_offset
          = static_cast<dwarf_implicit_pointer *> (location)->get_die_offset ();
      struct value* gdb_value
          = deref_implicit_pointer (die_offset, 0, type);
      return gdb_value_to_dwarf_entry (gdb_value);
    }
   else if (location->is_entry (dwarf_entry::Memory))
    {
      /* Covers the case where we have a passed in memory that is not part of the
         target and requiers for the location description to address it instead of
         addressing the actual target memory.

         This case only seems posible with expression baton resolution at the moment
         and currently only supports memory read from a memory location description.

         FIXME: I still need to analyze why this is even the case but for now
                this code fixes the situation.  */
      const dwarf_memory *memory_entry
          = static_cast<const dwarf_memory *> (location);

      CORE_ADDR start_address = memory_entry->get_offset ();
      LONGEST bit_suboffset = memory_entry->get_bit_suboffset ();
      bool in_stack = memory_entry->in_stack ();
      unsigned int address_space = memory_entry->get_address_space ();
      LONGEST this_size = bits_to_bytes (bit_suboffset, size * 8);
      /* Using two buffers here because the copy_bitwise doesn't
         seem to support in place copy.  */
      gdb::byte_vector temp_buf_bitwise (this_size);
      temp_buf.resize (this_size);

      /* We shouldn't have the case where we read from a passed in memory
         and the same memory being marked as stack or in some other address
         space. */
      if (!in_stack && !address_space
          && read_passed_in_mem (temp_buf.data (), start_address, this_size))
      {
        copy_bitwise (temp_buf_bitwise.data (), 0, temp_buf.data (),
                      bit_suboffset, size * 8, big_endian);
        return entry_factory->create_value (temp_buf_bitwise.data (),
                                            size, type);
      }
    }

  temp_buf.resize (size);

  read_from_location (location, frame, 0, temp_buf.data (), 0, size * 8,
                      big_endian, &optimized, &unavailable);

  if (optimized)
    throw_error (OPTIMIZED_OUT_ERROR,
                 _("Can't do read-modify-write to "
                 "update bitfield; containing word "
                 "has been optimized out"));
  if (unavailable)
    throw_error (NOT_AVAILABLE_ERROR,
                 _("Can't dereference "
                 "update bitfield; containing word "
                 "is unavailable"));

  return entry_factory->create_value (temp_buf.data (), size, type);
}

dwarf_entry *
dwarf_expr_context::gdb_value_to_dwarf_entry (struct value *value)
{
  LONGEST offset = value_offset (value);

  if (value_optimized_out (value))
    return entry_factory->create_undefined ();

  switch (value_lval_const (value))
    {
    /* We can only convert struct value to a location because we
       can't distinguish between the implicit value and not_lval.  */
    case not_lval:
      {
        gdb_byte *contents_start = value_contents_raw (value) + offset;
        int type_length = TYPE_LENGTH (value_type (value));
        return entry_factory->create_implicit (contents_start, type_length);
      }
    case lval_memory:
      return entry_factory->create_memory (value_address (value) + offset,
                                           0, 0, value_stack (value));
    case lval_register:
      return entry_factory->create_register (VALUE_REGNUM (value),
                                             false, offset);
    case lval_computed:
      {
        /* Dwarf entry is enclosed by the closure
           so we just need to unwrap it here.  */
        computed_closure *closure
            = ((computed_closure *) value_computed_closure (value));

        dwarf_entry *entry = closure->get_entry ();

        if (!entry->is_entry (dwarf_entry::Location))
      internal_error (__FILE__, __LINE__, _("invalid closure type"));

        static_cast<dwarf_location *> (entry)->add_bit_offset (offset * 8);
        return entry;
      }
    default:
      internal_error (__FILE__, __LINE__, _("invalid location type"));
  }
}

struct value *
dwarf_expr_context::dwarf_entry_to_gdb_value (dwarf_entry *entry,
                                              struct type *type,
                                              struct type *subobj_type,
                                              LONGEST subobj_offset)
{
  struct gdbarch *gdbarch = get_type_arch (type);
  struct value *retval = NULL;

  if (subobj_type == nullptr)
    subobj_type = type;

  if (entry->is_entry (dwarf_entry::Value))
    {
      dwarf_value *value = static_cast<dwarf_value *> (entry);
      retval = entry_factory->value_to_gdb_value (value, subobj_type,
                                                  subobj_offset);
    }
  else if (entry->is_entry (dwarf_entry::Undefined))
    {
      retval = allocate_value (subobj_type);
      mark_value_bytes_optimized_out (retval, subobj_offset,
                                      TYPE_LENGTH (subobj_type));
    }
  else if (entry->is_entry (dwarf_entry::Memory))
    {
      dwarf_memory *memory_entry = static_cast<dwarf_memory*> (entry);
      struct type *ptr_type = builtin_type (gdbarch)->builtin_data_ptr;
      CORE_ADDR address = memory_entry->get_offset ();
      unsigned int address_space = memory_entry->get_address_space ();

      address = aspace_address_to_flat_address (address, address_space);

      if (subobj_type->code () == TYPE_CODE_FUNC
          || subobj_type->code () == TYPE_CODE_METHOD)
    ptr_type = builtin_type (gdbarch)->builtin_func_ptr;

      address = value_as_address (value_from_pointer (ptr_type, address));
      retval = value_at_lazy (subobj_type, address + subobj_offset);
      set_value_stack (retval, memory_entry->in_stack ());
    }
  else if (entry->is_entry (dwarf_entry::Register))
    {
      dwarf_register* register_entry = static_cast<dwarf_register*> (entry);
      unsigned int regnum = register_entry->get_regnum ();
      int gdb_regnum = dwarf_reg_to_regnum_or_error (gdbarch, regnum);
      struct frame_info* frame = get_context_frame ();

      if (register_entry->is_on_entry ())
        frame = get_prev_frame_always (frame);

      if (frame == NULL)
        internal_error (__FILE__, __LINE__, _("invalid frame information"));

      retval = value_from_register (type, gdb_regnum, frame,
                                    register_entry->get_offset ());

      if (value_optimized_out (retval))
    {
      /* This means the register has undefined value / was not saved.
         As we're computing the location of some variable etc. in the
         program, not a value for inspecting a register ($pc, $sp, etc.),
         return a generic optimized out value instead, so that we show
         <optimized out> instead of <not saved>.  */
      struct value *temp = allocate_value (subobj_type);
      value_contents_copy (temp, 0, retval, 0, TYPE_LENGTH (subobj_type));
      retval = temp;
    }
    }
  else if (entry->is_entry (dwarf_entry::Implicit))
    {
      dwarf_implicit *implicit_entry = static_cast<dwarf_implicit*> (entry);
      size_t subtype_len = TYPE_LENGTH (subobj_type);
      size_t type_len = TYPE_LENGTH (type);

      if (subobj_offset + subtype_len > type_len)
        invalid_synthetic_pointer ();

      retval = allocate_value (subobj_type);

      /* The given offset is relative to the actual object.  */
      if (gdbarch_byte_order (gdbarch) == BFD_ENDIAN_BIG)
    subobj_offset += implicit_entry->get_size () - type_len;

      memcpy ((void *)value_contents_raw (retval),
              (void *)(implicit_entry->get_contents () + subobj_offset),
              subtype_len);
    }
  else if (entry->is_entry (dwarf_entry::ImplicitPtr))
    {
      implicit_pointer_closure *closure
          = new implicit_pointer_closure (entry, get_per_cu (),
                                          this->per_objfile);
      const struct lval_funcs *funcs = get_closure_callbacks ();

      closure->incref ();

      /* Complain if the expression is larger than the size of the
         outer type.  */
      if (this->addr_size > 8 * TYPE_LENGTH (type))
    invalid_synthetic_pointer ();

      retval = allocate_computed_value (subobj_type, funcs, closure);
      set_value_offset (retval, subobj_offset);
    }
  else if (entry->is_entry (dwarf_entry::Composite))
    {
      composite_closure *closure;
      const struct lval_funcs *funcs;
      dwarf_composite *composite_entry
          = static_cast<dwarf_composite *> (entry);
      size_t pieces_num = composite_entry->get_pieces_num ();
      ULONGEST bit_size = 0;

      for (unsigned int i = 0; i < pieces_num; i++)
    bit_size += composite_entry->get_bit_size_at (i);

      /* If compilation unit information is not available
         we are in a CFI context.  */
      if (get_per_cu () == NULL)
    closure = new composite_closure (entry, get_context_frame ());
      else
    closure
        = new composite_closure (entry, get_frame_id (get_context_frame ()));
 
      closure->incref ();

      funcs = get_closure_callbacks ();

      /* Complain if the expression is larger than the size of the
         outer type.  */
      if (bit_size > 8 * TYPE_LENGTH (type))
        invalid_synthetic_pointer ();

      retval = allocate_computed_value (subobj_type, funcs, closure);
      set_value_offset (retval, subobj_offset);
  }

  return retval;
}

bool
dwarf_expr_context::dwarf_entry_equal_op (dwarf_entry *arg1, dwarf_entry *arg2)
{
  dwarf_value *arg1_value, *arg2_value;
  struct value *arg1_gdb_value, *arg2_gdb_value;
  arg1_value = entry_factory->entry_to_value (arg1, address_type ());
  arg2_value = entry_factory->entry_to_value (arg2, address_type ());
  arg1_gdb_value
      = entry_factory->value_to_gdb_value (arg1_value,
                                           arg1_value->get_type ());
  arg2_gdb_value
      = entry_factory->value_to_gdb_value (arg2_value,
                                           arg2_value->get_type ());

  return value_equal (arg1_gdb_value, arg2_gdb_value);
}

bool
dwarf_expr_context::dwarf_entry_less_op (dwarf_entry *arg1, dwarf_entry *arg2)
{
  dwarf_value *arg1_value, *arg2_value;
  struct value *arg1_gdb_value, *arg2_gdb_value;
  arg1_value = entry_factory->entry_to_value (arg1, address_type ());
  arg2_value = entry_factory->entry_to_value (arg2, address_type ());
  arg1_gdb_value
      = entry_factory->value_to_gdb_value (arg1_value,
                                           arg1_value->get_type ());
  arg2_gdb_value
      = entry_factory->value_to_gdb_value (arg2_value,
                                           arg2_value->get_type ());

  return value_less (arg1_gdb_value, arg2_gdb_value);
}

/* Return true if the expression stack is empty.  */

bool
dwarf_expr_context::stack_empty_p () const
{
  return stack.empty ();
}

/* Fetch the last stack element and convert it to the appropriate struct value
   form of a TYPE type. AS_LVAL defines if the fetched value is expected to be
   a value or a location description. SUBOBJ_TYPE information is used for more
   precise description of the source variable type information. SUBOBJ_OFFSET
   information defines an offset into the DWARF entry contents.  */

value *
dwarf_expr_context::fetch_value (bool as_lval, struct type *type,
                                 struct type *subobj_type,
                                 LONGEST subobj_offset)
{
  dwarf_entry *entry = fetch (0);

  if (type == nullptr)
    type = address_type ();

  if (!as_lval)
    entry = entry_factory->entry_to_value (entry, address_type ());
  else
    entry = entry_factory->entry_to_location (entry);

  return dwarf_entry_to_gdb_value (entry, type, subobj_type, subobj_offset);
}

/* Add a new piece to the dwarf_expr_context's piece list.  */

dwarf_entry *
dwarf_expr_context::add_piece (ULONGEST bit_size, ULONGEST bit_offset)
{
  dwarf_location *piece_entry;
  dwarf_composite *composite_entry;

  if (stack_empty_p ())
    piece_entry = entry_factory->create_undefined ();
  else
    {
      piece_entry = entry_factory->entry_to_location (fetch (0));

      if (piece_entry->is_entry (dwarf_entry::Composite))
    {
      composite_entry = static_cast<dwarf_composite *> (piece_entry);

      if (!composite_entry->is_completed ())
        piece_entry = entry_factory->create_undefined ();
    }
      else if (piece_entry->is_entry (dwarf_entry::Undefined))
    pop ();
    }

  if (!piece_entry->is_entry (dwarf_entry::Undefined))
    {
      piece_entry->add_bit_offset (bit_offset);
      pop ();
    }

   /* If stack is empty then it is a start of a new composite.
      Later this will check if the composite is finished or not.  */
   if (stack_empty_p () || !fetch (0)->is_entry (dwarf_entry::Composite))
    composite_entry = entry_factory->create_composite ();
  else
    {
      composite_entry = static_cast<dwarf_composite *> (fetch (0));

      if (composite_entry->is_completed ())
    composite_entry = entry_factory->create_composite ();
      else
    {
      composite_entry = static_cast<dwarf_composite *> (fetch (0));
      pop ();
    }
    }

   composite_entry->add_piece (piece_entry, bit_size);
   return composite_entry;
 }

dwarf_entry *
dwarf_expr_context::create_select_composite (ULONGEST piece_bit_size,
                                             ULONGEST pieces_count)
{
  dwarf_location *zero, *one;
  dwarf_value *mask;
  struct type *mask_type;
  dwarf_composite *composite_entry;
  gdb::byte_vector mask_buf;
  gdb_byte* mask_buf_data;
  ULONGEST i, mask_size;

  if (stack_empty_p () || piece_bit_size == 0 || pieces_count == 0)
    ill_formed_expression ();

  mask = entry_factory->entry_to_value (fetch (0), address_type ());
  mask_type = mask->get_type ();
  mask_size = TYPE_LENGTH (mask_type);
  dwarf_require_integral (mask_type);
  pop ();

  if (mask_size * 8 < pieces_count)
    ill_formed_expression ();

  mask_buf.resize (mask_size);
  mask_buf_data = mask_buf.data ();

  copy_bitwise (mask_buf_data, 0, mask->get_contents (), 0, mask_size * 8,
                type_byte_order (mask_type) == BFD_ENDIAN_BIG);

  if (stack_empty_p ())
    ill_formed_expression ();

  one = entry_factory->entry_to_location (fetch (0));
  pop ();

  if (stack_empty_p ())
    ill_formed_expression ();

  zero = entry_factory->entry_to_location (fetch (0));
  pop ();

  composite_entry = entry_factory->create_composite ();

  for (i = 0; i < pieces_count; i++)
    {
      dwarf_location *piece;

      if ((mask_buf_data[i / 8] >> (i % 8)) & 1)
    piece = static_cast<dwarf_location *> (entry_factory->copy_entry (one));
      else
    piece = static_cast<dwarf_location *> (entry_factory->copy_entry (zero));

      piece->add_bit_offset (i * piece_bit_size);
      composite_entry->add_piece (piece, piece_bit_size);
    }

  return composite_entry;
}

dwarf_entry *
dwarf_expr_context::create_extend_composite (ULONGEST piece_bit_size,
                                             ULONGEST pieces_count)
{
  dwarf_location *location;
  dwarf_composite *composite_entry;
  ULONGEST i;

  if (stack_empty_p () || piece_bit_size == 0 || pieces_count == 0)
    ill_formed_expression ();

  location = entry_factory->entry_to_location (fetch (0));
  composite_entry = entry_factory->create_composite ();

  for (i = 0; i < pieces_count; i++)
    {
      dwarf_entry *piece = entry_factory->copy_entry (location);
      composite_entry->add_piece (static_cast<dwarf_location *> (piece),
                                  piece_bit_size);
    }

  return composite_entry;
}

/* Evaluate the expression at ADDR (LEN bytes long).  */

void
dwarf_expr_context::eval (const gdb_byte *addr, size_t len)
{
  int old_recursion_depth = this->recursion_depth;

  execute_stack_op (addr, addr + len);

  /* RECURSION_DEPTH becomes invalid if an exception was thrown here.  */

  gdb_assert (this->recursion_depth == old_recursion_depth);
}

/* Helper to read a uleb128 value or throw an error.  */

const gdb_byte *
safe_read_uleb128 (const gdb_byte *buf, const gdb_byte *buf_end,
		   uint64_t *r)
{
  buf = gdb_read_uleb128 (buf, buf_end, r);
  if (buf == NULL)
    error (_("DWARF expression error: ran off end of buffer reading uleb128 value"));
  return buf;
}

/* Helper to read a sleb128 value or throw an error.  */

const gdb_byte *
safe_read_sleb128 (const gdb_byte *buf, const gdb_byte *buf_end,
		   int64_t *r)
{
  buf = gdb_read_sleb128 (buf, buf_end, r);
  if (buf == NULL)
    error (_("DWARF expression error: ran off end of buffer reading sleb128 value"));
  return buf;
}

const gdb_byte *
safe_skip_leb128 (const gdb_byte *buf, const gdb_byte *buf_end)
{
  buf = gdb_skip_leb128 (buf, buf_end);
  if (buf == NULL)
    error (_("DWARF expression error: ran off end of buffer reading leb128 value"));
  return buf;
}


/* Check that the current operator is either at the end of an
   expression, or that it is followed by a composition operator or by
   DW_OP_GNU_uninit (which should terminate the expression).  */

void
dwarf_expr_require_composition (const gdb_byte *op_ptr, const gdb_byte *op_end,
				const char *op_name)
{
  if (op_ptr != op_end && *op_ptr != DW_OP_piece && *op_ptr != DW_OP_bit_piece
      && *op_ptr != DW_OP_GNU_uninit)
    error (_("DWARF-2 expression error: `%s' operations must be "
	     "used either alone or in conjunction with DW_OP_piece "
	     "or DW_OP_bit_piece."),
	   op_name);
}

/* Return true iff the types T1 and T2 are "the same".  This only does
   checks that might reasonably be needed to compare DWARF base
   types.  */

static int
base_types_equal_p (struct type *t1, struct type *t2)
{
  if (t1->code () != t2->code ())
    return 0;
  if (TYPE_UNSIGNED (t1) != TYPE_UNSIGNED (t2))
    return 0;
  return TYPE_LENGTH (t1) == TYPE_LENGTH (t2);
}

/* If <BUF..BUF_END] contains DW_FORM_block* with single DW_OP_reg* return the
   DWARF register number.  Otherwise return -1.  */

int
dwarf_block_to_dwarf_reg (const gdb_byte *buf, const gdb_byte *buf_end)
{
  uint64_t dwarf_reg;

  if (buf_end <= buf)
    return -1;
  if (*buf >= DW_OP_reg0 && *buf <= DW_OP_reg31)
    {
      if (buf_end - buf != 1)
	return -1;
      return *buf - DW_OP_reg0;
    }

  if (*buf == DW_OP_regval_type || *buf == DW_OP_GNU_regval_type)
    {
      buf++;
      buf = gdb_read_uleb128 (buf, buf_end, &dwarf_reg);
      if (buf == NULL)
	return -1;
      buf = gdb_skip_leb128 (buf, buf_end);
      if (buf == NULL)
	return -1;
    }
  else if (*buf == DW_OP_regx)
    {
      buf++;
      buf = gdb_read_uleb128 (buf, buf_end, &dwarf_reg);
      if (buf == NULL)
	return -1;
    }
  else
    return -1;
  if (buf != buf_end || (int) dwarf_reg != dwarf_reg)
    return -1;
  return dwarf_reg;
}

/* If <BUF..BUF_END] contains DW_FORM_block* with just DW_OP_breg*(0) and
   DW_OP_deref* return the DWARF register number.  Otherwise return -1.
   DEREF_SIZE_RETURN contains -1 for DW_OP_deref; otherwise it contains the
   size from DW_OP_deref_size.  */

int
dwarf_block_to_dwarf_reg_deref (const gdb_byte *buf, const gdb_byte *buf_end,
				CORE_ADDR *deref_size_return)
{
  uint64_t dwarf_reg;
  int64_t offset;

  if (buf_end <= buf)
    return -1;

  if (*buf >= DW_OP_breg0 && *buf <= DW_OP_breg31)
    {
      dwarf_reg = *buf - DW_OP_breg0;
      buf++;
      if (buf >= buf_end)
	return -1;
    }
  else if (*buf == DW_OP_bregx)
    {
      buf++;
      buf = gdb_read_uleb128 (buf, buf_end, &dwarf_reg);
      if (buf == NULL)
	return -1;
      if ((int) dwarf_reg != dwarf_reg)
       return -1;
    }
  else
    return -1;

  buf = gdb_read_sleb128 (buf, buf_end, &offset);
  if (buf == NULL)
    return -1;
  if (offset != 0)
    return -1;

  if (*buf == DW_OP_deref)
    {
      buf++;
      *deref_size_return = -1;
    }
  else if (*buf == DW_OP_deref_size)
    {
      buf++;
      if (buf >= buf_end)
       return -1;
      *deref_size_return = *buf++;
    }
  else
    return -1;

  if (buf != buf_end)
    return -1;

  return dwarf_reg;
}

/* If <BUF..BUF_END] contains DW_FORM_block* with single DW_OP_fbreg(X) fill
   in FB_OFFSET_RETURN with the X offset and return 1.  Otherwise return 0.  */

int
dwarf_block_to_fb_offset (const gdb_byte *buf, const gdb_byte *buf_end,
			  CORE_ADDR *fb_offset_return)
{
  int64_t fb_offset;

  if (buf_end <= buf)
    return 0;

  if (*buf != DW_OP_fbreg)
    return 0;
  buf++;

  buf = gdb_read_sleb128 (buf, buf_end, &fb_offset);
  if (buf == NULL)
    return 0;
  *fb_offset_return = fb_offset;
  if (buf != buf_end || fb_offset != (LONGEST) *fb_offset_return)
    return 0;

  return 1;
}

/* If <BUF..BUF_END] contains DW_FORM_block* with single DW_OP_bregSP(X) fill
   in SP_OFFSET_RETURN with the X offset and return 1.  Otherwise return 0.
   The matched SP register number depends on GDBARCH.  */

int
dwarf_block_to_sp_offset (struct gdbarch *gdbarch, const gdb_byte *buf,
			  const gdb_byte *buf_end, CORE_ADDR *sp_offset_return)
{
  uint64_t dwarf_reg;
  int64_t sp_offset;

  if (buf_end <= buf)
    return 0;
  if (*buf >= DW_OP_breg0 && *buf <= DW_OP_breg31)
    {
      dwarf_reg = *buf - DW_OP_breg0;
      buf++;
    }
  else
    {
      if (*buf != DW_OP_bregx)
       return 0;
      buf++;
      buf = gdb_read_uleb128 (buf, buf_end, &dwarf_reg);
      if (buf == NULL)
	return 0;
    }

  if (dwarf_reg_to_regnum (gdbarch, dwarf_reg)
      != gdbarch_sp_regnum (gdbarch))
    return 0;

  buf = gdb_read_sleb128 (buf, buf_end, &sp_offset);
  if (buf == NULL)
    return 0;
  *sp_offset_return = sp_offset;
  if (buf != buf_end || sp_offset != (LONGEST) *sp_offset_return)
    return 0;

  return 1;
}

/* Read or write a pieced value V.  If FROM != NULL, operate in "write
   mode": copy FROM into the pieces comprising V.  If FROM == NULL,
   operate in "read mode": fetch the contents of the (lazy) value V by
   composing it from its pieces.  */

void
rw_closure_value (struct value *v, struct value *from)
{
  unsigned int i, pieces_num;
  LONGEST bit_offset = 0, max_bit_offset;
  ULONGEST bits_to_skip;
  gdb_byte *v_contents;
  const gdb_byte *from_contents;
  computed_closure *closure = ((computed_closure*)value_computed_closure (v));
  bool big_endian = type_byte_order (value_type (v)) == BFD_ENDIAN_BIG;

  if (!closure->is_closure (computed_closure::ImplicitPtr)
      && !closure->is_closure (computed_closure::Composite))
    internal_error (__FILE__, __LINE__, _("invalid closure type"));

  if (from != NULL)
    {
      from_contents = value_contents (from);
      v_contents = NULL;
    }
  else
    {
      if (value_type (v) != value_enclosing_type (v))
    internal_error (__FILE__, __LINE__,
                    _("Should not be able to create a lazy value with "
                      "an enclosing type"));
      v_contents = value_contents_raw (v);
      from_contents = NULL;
    }

  bits_to_skip = 8 * value_offset (v);
  if (value_bitsize (v))
    {
      bits_to_skip += (8 * value_offset (value_parent (v)) + value_bitpos (v));
      if (from != NULL && big_endian)
    {
      /* Use the least significant bits of FROM.  */
      max_bit_offset = 8 * TYPE_LENGTH (value_type (from));
      bit_offset = max_bit_offset - value_bitsize (v);
    }
      else
    max_bit_offset = value_bitsize (v);
    }
  else
    max_bit_offset = 8 * TYPE_LENGTH (value_type (v));

  if (closure->is_closure (computed_closure::ImplicitPtr))
    {
      if (from != NULL)
    mark_value_bits_optimized_out (v, bits_to_skip,
                                   8 * TYPE_LENGTH (value_type (v)));
      return;
    }

  composite_closure *comp_closure
      = static_cast<composite_closure *> (closure);
  struct frame_info *frame = comp_closure->get_frame ();

  if (frame == NULL)
    frame = frame_find_by_id (comp_closure->get_frame_id ());

  if (!closure->get_entry ()->is_entry (dwarf_entry::Composite))
    internal_error (__FILE__, __LINE__, _("invalid location type"));

  dwarf_composite *composite_entry
      = static_cast<dwarf_composite *> (closure->get_entry ());

  /* Advance to the first non-skipped piece.  */
  pieces_num = composite_entry->get_pieces_num ();

  for (i = 0; i < pieces_num; i++)
    {
      ULONGEST bit_size = composite_entry->get_bit_size_at (i);

      if (bits_to_skip < bit_size)
    break;

      bits_to_skip -= bit_size;
    }

  for (; i < pieces_num && bit_offset < max_bit_offset; i++)
    {
      const dwarf_location *location = composite_entry->get_piece_at (i);
      ULONGEST bit_size = composite_entry->get_bit_size_at (i);
      size_t this_bit_size = bit_size - bits_to_skip;
      int optimized, unavailable;

      if (this_bit_size > max_bit_offset - bit_offset)
    this_bit_size = max_bit_offset - bit_offset;

      if (from == NULL)
    {
      read_from_location (location, frame, bits_to_skip, v_contents,
                          bit_offset, this_bit_size, big_endian,
                          &optimized, &unavailable);

      if (optimized)
        mark_value_bits_optimized_out (v, bit_offset, this_bit_size);
      if (unavailable)
        mark_value_bits_unavailable (v, bit_offset, this_bit_size);
    }
      else
    {
      write_to_location (location, frame, bits_to_skip, from_contents,
                         bit_offset, this_bit_size, big_endian,
                         &optimized, &unavailable);

      if (optimized)
        throw_error (OPTIMIZED_OUT_ERROR,
                     _("Can't do read-modify-write to "
                     "update bitfield; containing word "
                     "has been optimized out"));
      if (unavailable)
        throw_error (NOT_AVAILABLE_ERROR,
                     _("Can't do read-modify-write to "
                     "update bitfield; containing word "
                     "is unavailable"));
    }

      bit_offset += this_bit_size;
      bits_to_skip = 0;
    }
}

void *
copy_value_closure (const struct value *v)
{
  computed_closure *closure = ((computed_closure*)value_computed_closure (v));

  if (closure == nullptr)
    internal_error (__FILE__, __LINE__, _("invalid closure type"));

  closure->incref ();
  return closure;
}

void
free_value_closure (struct value *v)
{
  computed_closure *closure = ((computed_closure*)value_computed_closure (v));

  if (closure == nullptr)
    internal_error (__FILE__, __LINE__, _("invalid closure type"));

  closure->decref ();

  if (closure->refcount () == 0)
    delete closure;
}

/* FIXME: This needs to be removed when CORE_ADDR supports address spaces.  */
CORE_ADDR
aspace_address_to_flat_address (CORE_ADDR address, unsigned int address_space)
{
  return address | ((ULONGEST) address_space) << ROCM_ASPACE_BIT_OFFSET;
}

/* The engine for the expression evaluator.  Using the context in this
   object, evaluate the expression between OP_PTR and OP_END.  */

void
dwarf_expr_context::execute_stack_op (const gdb_byte *op_ptr,
				      const gdb_byte *op_end)
{
  enum bfd_endian byte_order = gdbarch_byte_order (this->gdbarch);
  /* Old-style "untyped" DWARF values need special treatment in a
     couple of places, specifically DW_OP_mod and DW_OP_shr.  We need
     a special type for these values so we can distinguish them from
     values that have an explicit type, because explicitly-typed
     values do not need special treatment.  This special type must be
     different (in the `==' sense) from any base type coming from the
     CU.  */
  struct type *address_type = this->address_type ();

  if (this->recursion_depth > this->max_recursion_depth)
    error (_("DWARF-2 expression error: Loop detected (%d)."),
	   this->recursion_depth);
  this->recursion_depth++;

  while (op_ptr < op_end)
    {
      enum dwarf_location_atom op = (enum dwarf_location_atom) *op_ptr++;
      ULONGEST result;
      uint64_t uoffset, reg;
      int64_t offset;
      struct dwarf_entry *result_entry = NULL;

      /* The DWARF expression might have a bug causing an infinite
	 loop.  In that case, quitting is the only way out.  */
      QUIT;

      switch (op)
	{
	case DW_OP_lit0:
	case DW_OP_lit1:
	case DW_OP_lit2:
	case DW_OP_lit3:
	case DW_OP_lit4:
	case DW_OP_lit5:
	case DW_OP_lit6:
	case DW_OP_lit7:
	case DW_OP_lit8:
	case DW_OP_lit9:
	case DW_OP_lit10:
	case DW_OP_lit11:
	case DW_OP_lit12:
	case DW_OP_lit13:
	case DW_OP_lit14:
	case DW_OP_lit15:
	case DW_OP_lit16:
	case DW_OP_lit17:
	case DW_OP_lit18:
	case DW_OP_lit19:
	case DW_OP_lit20:
	case DW_OP_lit21:
	case DW_OP_lit22:
	case DW_OP_lit23:
	case DW_OP_lit24:
	case DW_OP_lit25:
	case DW_OP_lit26:
	case DW_OP_lit27:
	case DW_OP_lit28:
	case DW_OP_lit29:
	case DW_OP_lit30:
	case DW_OP_lit31:
	  result = op - DW_OP_lit0;
	  result_entry = entry_factory->create_value (result, address_type);
	  break;

	case DW_OP_addr:
	  result = extract_unsigned_integer (op_ptr, this->addr_size, byte_order);
	  op_ptr += this->addr_size;
	  /* Some versions of GCC emit DW_OP_addr before
	     DW_OP_GNU_push_tls_address.  In this case the value is an
	     index, not an address.  We don't support things like
	     branching between the address and the TLS op.  */
	  if (op_ptr >= op_end || *op_ptr != DW_OP_GNU_push_tls_address)
	    {
	      result += this->per_objfile->objfile->text_section_offset ();
	      result_entry = entry_factory->create_memory (result);
	    }
	  else
            /* This is a special case where the value is expected to be
               created instead of memory location.  */
	    result_entry = entry_factory->create_value (result, address_type);
	  break;

	case DW_OP_addrx:
	case DW_OP_GNU_addr_index:
	  op_ptr = safe_read_uleb128 (op_ptr, op_end, &uoffset);
	  result = this->get_addr_index (uoffset);
	  result += this->per_objfile->objfile->text_section_offset ();
	  result_entry = entry_factory->create_memory (result);
	  break;
	case DW_OP_GNU_const_index:
	  op_ptr = safe_read_uleb128 (op_ptr, op_end, &uoffset);
	  result = this->get_addr_index (uoffset);
	  result_entry = entry_factory->create_value (result, address_type);
	  break;

	case DW_OP_const1u:
	  result = extract_unsigned_integer (op_ptr, 1, byte_order);
	  result_entry = entry_factory->create_value (result, address_type);
	  op_ptr += 1;
	  break;
	case DW_OP_const1s:
	  result = extract_signed_integer (op_ptr, 1, byte_order);
	  result_entry = entry_factory->create_value (result, address_type);
	  op_ptr += 1;
	  break;
	case DW_OP_const2u:
	  result = extract_unsigned_integer (op_ptr, 2, byte_order);
	  result_entry = entry_factory->create_value (result, address_type);
	  op_ptr += 2;
	  break;
	case DW_OP_const2s:
	  result = extract_signed_integer (op_ptr, 2, byte_order);
	  result_entry = entry_factory->create_value (result, address_type);
	  op_ptr += 2;
	  break;
	case DW_OP_const4u:
	  result = extract_unsigned_integer (op_ptr, 4, byte_order);
	  result_entry = entry_factory->create_value (result, address_type);
	  op_ptr += 4;
	  break;
	case DW_OP_const4s:
	  result = extract_signed_integer (op_ptr, 4, byte_order);
	  result_entry = entry_factory->create_value (result, address_type);
	  op_ptr += 4;
	  break;
	case DW_OP_const8u:
	  result = extract_unsigned_integer (op_ptr, 8, byte_order);
	  result_entry = entry_factory->create_value (result, address_type);
	  op_ptr += 8;
	  break;
	case DW_OP_const8s:
	  result = extract_signed_integer (op_ptr, 8, byte_order);
	  result_entry = entry_factory->create_value (result, address_type);
	  op_ptr += 8;
	  break;
	case DW_OP_constu:
	  op_ptr = safe_read_uleb128 (op_ptr, op_end, &uoffset);
	  result = uoffset;
	  result_entry = entry_factory->create_value (result, address_type);
	  break;
	case DW_OP_consts:
	  op_ptr = safe_read_sleb128 (op_ptr, op_end, &offset);
	  result = offset;
	  result_entry = entry_factory->create_value (result, address_type);
	  break;

	/* The DW_OP_reg operations are required to occur alone in
	   location expressions.  */
	case DW_OP_reg0:
	case DW_OP_reg1:
	case DW_OP_reg2:
	case DW_OP_reg3:
	case DW_OP_reg4:
	case DW_OP_reg5:
	case DW_OP_reg6:
	case DW_OP_reg7:
	case DW_OP_reg8:
	case DW_OP_reg9:
	case DW_OP_reg10:
	case DW_OP_reg11:
	case DW_OP_reg12:
	case DW_OP_reg13:
	case DW_OP_reg14:
	case DW_OP_reg15:
	case DW_OP_reg16:
	case DW_OP_reg17:
	case DW_OP_reg18:
	case DW_OP_reg19:
	case DW_OP_reg20:
	case DW_OP_reg21:
	case DW_OP_reg22:
	case DW_OP_reg23:
	case DW_OP_reg24:
	case DW_OP_reg25:
	case DW_OP_reg26:
	case DW_OP_reg27:
	case DW_OP_reg28:
	case DW_OP_reg29:
	case DW_OP_reg30:
	case DW_OP_reg31:
	  result = op - DW_OP_reg0;
	  result_entry = entry_factory->create_register (result);
	  eval_needs_frame ();
	  break;

	case DW_OP_regx:
	  op_ptr = safe_read_uleb128 (op_ptr, op_end, &reg);
	  result_entry = entry_factory->create_register (reg);
	  eval_needs_frame ();
	  break;

	case DW_OP_implicit_value:
	  {
	    uint64_t len;

	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &len);
	    if (op_ptr + len > op_end)
	      error (_("DW_OP_implicit_value: too few bytes available."));
	    result_entry = entry_factory->create_implicit (op_ptr, len);
	    op_ptr += len;
	  }
	  break;

	case DW_OP_stack_value:
	  {
	    dwarf_value *value
	        = entry_factory->entry_to_value (fetch (0), address_type);
	    pop ();

	    result_entry
	        = entry_factory->create_implicit (value->get_contents (),
	                                          TYPE_LENGTH (value->get_type ()));
	  }
	  break;

	case DW_OP_implicit_pointer:
	case DW_OP_GNU_implicit_pointer:
	  {
	    sect_offset die_offset;
	    int64_t len;

	    if (this->ref_addr_size == -1)
	      error (_("DWARF-2 expression error: DW_OP_implicit_pointer "
	             "is not allowed in frame context"));

	    /* The referred-to DIE of sect_offset kind.  */
	    die_offset = (sect_offset) extract_unsigned_integer (op_ptr,
                                                           this->ref_addr_size,
	                                                         byte_order);
	    op_ptr += this->ref_addr_size;

	    /* The byte offset into the data.  */
	    op_ptr = safe_read_sleb128 (op_ptr, op_end, &len);

	    result_entry
	        = entry_factory->create_implicit_pointer (die_offset, len);
	  }
	  break;

	case DW_OP_breg0:
	case DW_OP_breg1:
	case DW_OP_breg2:
	case DW_OP_breg3:
	case DW_OP_breg4:
	case DW_OP_breg5:
	case DW_OP_breg6:
	case DW_OP_breg7:
	case DW_OP_breg8:
	case DW_OP_breg9:
	case DW_OP_breg10:
	case DW_OP_breg11:
	case DW_OP_breg12:
	case DW_OP_breg13:
	case DW_OP_breg14:
	case DW_OP_breg15:
	case DW_OP_breg16:
	case DW_OP_breg17:
	case DW_OP_breg18:
	case DW_OP_breg19:
	case DW_OP_breg20:
	case DW_OP_breg21:
	case DW_OP_breg22:
	case DW_OP_breg23:
	case DW_OP_breg24:
	case DW_OP_breg25:
	case DW_OP_breg26:
	case DW_OP_breg27:
	case DW_OP_breg28:
	case DW_OP_breg29:
	case DW_OP_breg30:
	case DW_OP_breg31:
	  {
	    dwarf_location *location;

	    op_ptr = safe_read_sleb128 (op_ptr, op_end, &offset);

	    location = entry_factory->create_register (op - DW_OP_breg0);
	    result_entry = dwarf_entry_deref (location, address_type);
	    location = entry_factory->entry_to_location (result_entry);
	    location->add_bit_offset (offset * 8);
	    result_entry = location;
	  }
	  break;
	case DW_OP_bregx:
	  {
	    dwarf_location *location;

	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &reg);
	    op_ptr = safe_read_sleb128 (op_ptr, op_end, &offset);

	    location = entry_factory->create_register (op - DW_OP_breg0);
	    result_entry = dwarf_entry_deref (location, address_type);
	    location = entry_factory->entry_to_location (result_entry);
	    location->add_bit_offset (offset * 8);
	    result_entry = location;
	  }
	  break;
	case DW_OP_fbreg:
	  {
	    dwarf_memory *memory_entry;
	    const gdb_byte *datastart;
	    size_t datalen;

	    op_ptr = safe_read_sleb128 (op_ptr, op_end, &offset);

	    /* Rather than create a whole new context, we simply
	       backup the current stack locally and install a new empty stack,
	       then reset it afterwards, effectively erasing whatever the
	       recursive call put there.  */
	    std::vector<dwarf_entry *> saved_stack = std::move (stack);
	    stack.clear ();

	    /* FIXME: cagney/2003-03-26: This code should be using
               get_frame_base_address(), and then implement a dwarf2
               specific this_base method.  */
	    this->get_frame_base (&datastart, &datalen);
	    eval (datastart, datalen);
	    result_entry = fetch (0);

	    if (result_entry->is_entry (dwarf_entry::Register))
	      result_entry = dwarf_entry_deref (result_entry, address_type);

	    result_entry = entry_factory->entry_to_location (result_entry);

	    /* If we get anything else then memory location here,
	       the DWARF standard defines the expression as ill formed.  */
	    if (!result_entry->is_entry (dwarf_entry::Memory))
	      ill_formed_expression ();

	    memory_entry = static_cast<dwarf_memory *> (result_entry);
	    memory_entry->add_bit_offset (offset * 8);
	    memory_entry->set_stack (true);
	    result_entry = memory_entry;

	    /* Restore the content of the original stack.  */
	    stack = std::move (saved_stack);
	  }
	  break;

	case DW_OP_dup:
	  result_entry = entry_factory->copy_entry (fetch (0));
	  break;

	case DW_OP_drop:
	  pop ();
	  goto no_push;

	case DW_OP_pick:
	  offset = *op_ptr++;
	  result_entry = fetch (offset);
	  break;
	  
	case DW_OP_swap:
	  {
	    if (stack.size () < 2)
	       error (_("Not enough elements for "
	              "DW_OP_swap.  Need 2, have %zu."), stack.size ());

	    dwarf_entry *temp = stack[stack.size () - 1];
	    stack[stack.size () - 1] = stack[stack.size () - 2];
	    stack[stack.size () - 2] = temp;
	    goto no_push;
	  }

	case DW_OP_over:
	  result_entry =  entry_factory->copy_entry (fetch (1));
	  break;

	case DW_OP_rot:
	  {
	    if (stack.size () < 3)
	       error (_("Not enough elements for "
			"DW_OP_rot.  Need 3, have %zu."),
		      stack.size ());

	    dwarf_entry *temp = stack[stack.size () - 1];
	    stack[stack.size () - 1] = stack[stack.size () - 2];
	    stack[stack.size () - 2] = stack[stack.size () - 3];
	    stack[stack.size () - 3] = temp;
	    goto no_push;
	  }

	case DW_OP_deref:
	case DW_OP_deref_size:
	case DW_OP_deref_type:
	case DW_OP_GNU_deref_type:
	  {
	    int addr_size = (op == DW_OP_deref ? this->addr_size : *op_ptr++);
	    struct type *type = address_type;

	    if (op == DW_OP_deref_type || op == DW_OP_GNU_deref_type)
	      {
	        op_ptr = safe_read_uleb128 (op_ptr, op_end, &uoffset);
	        cu_offset type_die_cu_off = (cu_offset) uoffset;
	        type = get_base_type (type_die_cu_off, 0);
	        addr_size = TYPE_LENGTH (type);
	      }

	    result_entry = dwarf_entry_deref (fetch (0), type, addr_size);
	    pop ();
	  }
	  break;

	case DW_OP_abs:
	case DW_OP_neg:
	case DW_OP_not:
	case DW_OP_plus_uconst:
	  {
	    /* Unary operations.  */
	    dwarf_value *arg
	        = entry_factory->entry_to_value (fetch (0), address_type);
	    pop ();

	    switch (op)
	  {
	    case DW_OP_abs:
	      {
	        struct value *arg_value
	            = entry_factory->value_to_gdb_value (arg, arg->get_type ());

	        if (value_less (arg_value, value_zero (arg->get_type (), not_lval)))
	          arg = entry_factory->value_negation_op (arg);
	      }
	    break;
	    case DW_OP_neg:
	      arg = entry_factory->value_negation_op (arg);
	      break;
	    case DW_OP_not:
	      dwarf_require_integral (arg->get_type ());
	      arg = entry_factory->value_complement_op (arg);
	      break;
	    case DW_OP_plus_uconst:
	      dwarf_require_integral (arg->get_type ());
	      op_ptr = safe_read_uleb128 (op_ptr, op_end, &reg);
	      result = arg->to_long () + reg;
	      arg = entry_factory->create_value (result, address_type);
	      break;
	  }
	    result_entry = arg;
	  }
	  break;

	case DW_OP_and:
	case DW_OP_div:
	case DW_OP_minus:
	case DW_OP_mod:
	case DW_OP_mul:
	case DW_OP_or:
	case DW_OP_plus:
	case DW_OP_shl:
	case DW_OP_shr:
	case DW_OP_shra:
	case DW_OP_xor:
	case DW_OP_le:
	case DW_OP_ge:
	case DW_OP_eq:
	case DW_OP_lt:
	case DW_OP_gt:
	case DW_OP_ne:
	  {
	    /* Binary operations.  */
	    dwarf_value *arg1, *arg2, *op_result;

	    arg2 = entry_factory->entry_to_value (fetch (0), address_type);
	    pop ();

	    arg1 = entry_factory->entry_to_value (fetch (0), address_type);
	    pop ();

	    if (! base_types_equal_p (arg1->get_type (), arg2->get_type ()))
	      error (_("Incompatible types on DWARF stack"));

	    switch (op)
	  {
	    case DW_OP_and:
	      dwarf_require_integral (arg1->get_type ());
	      dwarf_require_integral (arg2->get_type ());
	      op_result = entry_factory->value_binary_op (arg1, arg2,
	                                                  BINOP_BITWISE_AND);
	      break;
	    case DW_OP_div:
	      op_result = entry_factory->value_binary_op (arg1, arg2, BINOP_DIV);
	      break;
	    case DW_OP_minus:
	      op_result = entry_factory->value_binary_op (arg1, arg2, BINOP_SUB);
	      break;
	    case DW_OP_mod:
	      {
	        int cast_back = 0;
	        struct type *orig_type = arg1->get_type ();

	        /* We have to special-case "old-style" untyped values
	           these must have mod computed using unsigned math.  */
	        if (orig_type == address_type)
	      {
	        struct type *utype = get_unsigned_type (this->gdbarch, orig_type);

	        cast_back = 1;
	        arg1 = entry_factory->value_cast_op (arg1, utype);
	        arg2 = entry_factory->value_cast_op (arg2, utype);
	      }
	        /* Note that value_binop doesn't handle float or
	           decimal float here.  This seems unimportant.  */
	        op_result = entry_factory->value_binary_op (arg1, arg2, BINOP_MOD);
	        if (cast_back)
	      op_result = entry_factory->value_cast_op (op_result, orig_type);
	        break;
	      }
	    case DW_OP_mul:
	      op_result = entry_factory->value_binary_op (arg1, arg2, BINOP_MUL);
	      break;
	    case DW_OP_or:
	      dwarf_require_integral (arg1->get_type ());
	      dwarf_require_integral (arg2->get_type ());
	      op_result = entry_factory->value_binary_op (arg1, arg2,
	                                                  BINOP_BITWISE_IOR);
	      break;
	    case DW_OP_plus:
	      op_result = entry_factory->value_binary_op (arg1, arg2, BINOP_ADD);
	      break;
	    case DW_OP_shl:
	      dwarf_require_integral (arg1->get_type ());
	      dwarf_require_integral (arg2->get_type ());
	      op_result = entry_factory->value_binary_op (arg1, arg2, BINOP_LSH);
	      break;
	    case DW_OP_shr:
	      dwarf_require_integral (arg1->get_type ());
	      dwarf_require_integral (arg2->get_type ());
	      if (!TYPE_UNSIGNED (arg1->get_type ()))
	        {
	          struct type *utype
	              = get_unsigned_type (this->gdbarch, arg1->get_type ());

	          arg1 = entry_factory->value_cast_op (arg1, utype);
	        }

	      op_result = entry_factory->value_binary_op (arg1, arg2, BINOP_RSH);
	      /* Make sure we wind up with the same type we started
	         with.  */
	      if (op_result->get_type () != arg2->get_type ())
	        op_result = entry_factory->value_cast_op (op_result,
	                                                  arg2->get_type ());
	      break;
	    case DW_OP_shra:
	      dwarf_require_integral (arg1->get_type ());
	      dwarf_require_integral (arg2->get_type ());
	      if (TYPE_UNSIGNED (arg1->get_type ()))
	        {
	          struct type *stype = get_signed_type (this->gdbarch,
	                                                arg1->get_type ());

	          arg1 = entry_factory->value_cast_op (arg1, stype);
	        }

	      op_result = entry_factory->value_binary_op (arg1, arg2, BINOP_RSH);
	      /* Make sure we wind up with the same type we started  with.  */
	      if (op_result->get_type () != arg2->get_type ())
	        op_result = entry_factory->value_cast_op (op_result,
	                                                  arg2->get_type ());
	      break;
	    case DW_OP_xor:
	      dwarf_require_integral (arg1->get_type ());
	      dwarf_require_integral (arg2->get_type ());
	      op_result = entry_factory->value_binary_op (arg1, arg2,
	                                                  BINOP_BITWISE_XOR);
	      break;
	    case DW_OP_le:
	     /* A <= B is !(B < A).  */
	      result = ! dwarf_entry_less_op (arg2, arg1);
	      op_result = entry_factory->create_value (result, address_type);
	      break;
	    case DW_OP_ge:
	      /* A >= B is !(A < B).  */
	      result = ! dwarf_entry_less_op (arg1, arg2);
	      op_result = entry_factory->create_value (result, address_type);
	      break;
	    case DW_OP_eq:
	      result = dwarf_entry_equal_op (arg1, arg2);
	      op_result = entry_factory->create_value (result, address_type);
	      break;
	    case DW_OP_lt:
	      result = dwarf_entry_less_op (arg1, arg2);
	      op_result = entry_factory->create_value (result, address_type);
	      break;
	    case DW_OP_gt:
	      /* A > B is B < A.  */
	      result = dwarf_entry_less_op (arg2, arg1);
	      op_result = entry_factory->create_value (result, address_type);
	      break;
	    case DW_OP_ne:
	      result = ! dwarf_entry_equal_op (arg1, arg2);
	      op_result = entry_factory->create_value (result, address_type);
	      break;
	    default:
	      internal_error (__FILE__, __LINE__, ("Can't be reached."));
	  }
	    result_entry = op_result;
	  }
	  break;

	case DW_OP_call_frame_cfa:
	  result = this->get_frame_cfa ();
	  result_entry = entry_factory->create_memory (result, 0, 0, true);
	  break;

	case DW_OP_GNU_push_tls_address:
	case DW_OP_form_tls_address:
	  /* Variable is at a constant offset in the thread-local
	  storage block into the objfile for the current thread and
	  the dynamic linker module containing this expression.  Here
	  we return returns the offset from that base.  The top of the
	  stack has the offset from the beginning of the thread
	  control block at which the variable is located.  Nothing
	  should follow this operator, so the top of stack would be
	  returned.  */
	  result
	      = entry_factory->entry_to_value (fetch (0), address_type)->to_long ();
	  pop ();
	  result = this->get_tls_address (result);
	  result_entry = entry_factory->create_memory (result);
	  break;

	case DW_OP_skip:
	  offset = extract_signed_integer (op_ptr, 2, byte_order);
	  op_ptr += 2;
	  op_ptr += offset;
	  goto no_push;

	case DW_OP_bra:
	  {
	    dwarf_value *dwarf_value
	        = entry_factory->entry_to_value (fetch (0), address_type);

	    offset = extract_signed_integer (op_ptr, 2, byte_order);
	    op_ptr += 2;
	    dwarf_require_integral (dwarf_value->get_type ());
	    if (dwarf_value->to_long () != 0)
	      op_ptr += offset;
	    pop ();
	  }
	  goto no_push;

	case DW_OP_nop:
	  goto no_push;

	case DW_OP_piece:
	  {
	    uint64_t size;

	    /* Record the piece.  */
	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &size);
	    result_entry = add_piece (8 * size, 0);
	    break;
	  }

	case DW_OP_bit_piece:
	  {
	    uint64_t size, uleb_offset;

	    /* Record the piece.  */
	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &size);
	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &uleb_offset);
	    result_entry = add_piece (size, uleb_offset);
	    break;
	  }

	case DW_OP_GNU_uninit:
	  if (op_ptr != op_end)
	    error (_("DWARF-2 expression error: DW_OP_GNU_uninit must always "
	             "be the very last op."));

	  result_entry = fetch (0);
	  if (!result_entry->is_entry (dwarf_entry::Location))
	    ill_formed_expression ();

	  static_cast<dwarf_location *> (result_entry)->set_initialised (false);
	  goto no_push;

	case DW_OP_call2:
	  {
	    cu_offset cu_off
	      = (cu_offset) extract_unsigned_integer (op_ptr, 2, byte_order);
	    op_ptr += 2;
	    this->dwarf_call (cu_off);
	  }
	  goto no_push;

	case DW_OP_call4:
	  {
	    cu_offset cu_off
	      = (cu_offset) extract_unsigned_integer (op_ptr, 4, byte_order);
	    op_ptr += 4;
	    this->dwarf_call (cu_off);
	  }
	  goto no_push;

	case DW_OP_GNU_variable_value:
	  {
	    struct value *value;
	    struct type *type;
	    sect_offset die_offset
	      = (sect_offset) extract_unsigned_integer (op_ptr,
	                                                this->ref_addr_size,
	                                                byte_order);
	    op_ptr += this->ref_addr_size;
	    value = this->dwarf_variable_value (die_offset);
	    type = value_type (value);

	    result_entry = gdb_value_to_dwarf_entry (value);

	    eval_needs_frame ();

	    if (result_entry->is_entry (dwarf_entry::Undefined))
	      error_value_optimized_out ();
	    else
	      result_entry = dwarf_entry_deref (result_entry, type);
	  }
	  break;
	
	case DW_OP_entry_value:
	case DW_OP_GNU_entry_value:
	  {
	    uint64_t len;
	    CORE_ADDR deref_size;
	    union call_site_parameter_u kind_u;

	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &len);
	    if (op_ptr + len > op_end)
	      error (_("DW_OP_entry_value: too few bytes available."));

	    kind_u.dwarf_reg = dwarf_block_to_dwarf_reg (op_ptr, op_ptr + len);
	    if (kind_u.dwarf_reg != -1)
	      {
		op_ptr += len;
		this->push_dwarf_reg_entry_value (CALL_SITE_PARAMETER_DWARF_REG,
						  kind_u,
						  -1 /* deref_size */);
		goto no_push;
	      }

	    kind_u.dwarf_reg = dwarf_block_to_dwarf_reg_deref (op_ptr,
							       op_ptr + len,
							       &deref_size);
	    if (kind_u.dwarf_reg != -1)
	      {
		if (deref_size == -1)
		  deref_size = this->addr_size;
		op_ptr += len;
		this->push_dwarf_reg_entry_value (CALL_SITE_PARAMETER_DWARF_REG,
						  kind_u, deref_size);
		goto no_push;
	      }

	    error (_("DWARF-2 expression error: DW_OP_entry_value is "
		     "supported only for single DW_OP_reg* "
		     "or for DW_OP_breg*(0)+DW_OP_deref*"));
	  }

	case DW_OP_GNU_parameter_ref:
	  {
	    union call_site_parameter_u kind_u;

	    kind_u.param_cu_off
	      = (cu_offset) extract_unsigned_integer (op_ptr, 4, byte_order);
	    op_ptr += 4;
	    this->push_dwarf_reg_entry_value (CALL_SITE_PARAMETER_PARAM_OFFSET,
					      kind_u,
					      -1 /* deref_size */);
	  }
	  goto no_push;

	case DW_OP_const_type:
	case DW_OP_GNU_const_type:
	  {
	    int n;
	    const gdb_byte *data;
	    struct type *type;

	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &uoffset);
	    cu_offset type_die_cu_off = (cu_offset) uoffset;

	    n = *op_ptr++;
	    data = op_ptr;
	    op_ptr += n;

	    type = get_base_type (type_die_cu_off, n);
	    result_entry
          = entry_factory->create_value (data, TYPE_LENGTH (type), type);
	  }
	  break;

	case DW_OP_regval_type:
	case DW_OP_GNU_regval_type:
	  {
	    struct type *type;
	    dwarf_register *register_descr;

	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &reg);
	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &uoffset);
	    cu_offset type_die_cu_off = (cu_offset) uoffset;
	    type = get_base_type (type_die_cu_off, 0);

	    register_descr = entry_factory->create_register (reg);
	    result_entry = dwarf_entry_deref_reinterpret (register_descr, type);
	  }
	  break;

	case DW_OP_convert:
	case DW_OP_GNU_convert:
	case DW_OP_reinterpret:
	case DW_OP_GNU_reinterpret:
	  {
	    struct type *type;
	    dwarf_value *dwarf_value
	        = entry_factory->entry_to_value (fetch (0), address_type);

	    pop ();

	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &uoffset);
	    cu_offset type_die_cu_off = (cu_offset) uoffset;

	    if (to_underlying (type_die_cu_off) == 0)
	      type = address_type;
	    else
	      type = get_base_type (type_die_cu_off, 0);

	    if (op == DW_OP_convert || op == DW_OP_GNU_convert)
	      dwarf_value = entry_factory->value_cast_op (dwarf_value, type);
	    else if (type == dwarf_value->get_type ())
	      {
	      /* Nothing.  */
	      }
	    else if (TYPE_LENGTH (type)
	         != TYPE_LENGTH (dwarf_value->get_type ()))
	      error (_("DW_OP_reinterpret has wrong size"));
	    else
	      dwarf_value
	          = entry_factory->create_value (dwarf_value->get_contents (),
	                                         TYPE_LENGTH (type), type);

	    result_entry = dwarf_value;
	    break;
	  }

	case DW_OP_push_object_address:
	  /* Return the address of the object we are currently observing.  */
	  result = this->get_object_address ();
	  result_entry = entry_factory->create_value (result, address_type);
	  break;

	case DW_OP_LLVM_form_aspace_address:
	  {
	    dwarf_value *aspace_value, *address_value;

	    aspace_value
	        = entry_factory->entry_to_value (fetch (0), address_type);
	    pop ();

	    address_value
	        = entry_factory->entry_to_value (fetch (0), address_type);
	    pop ();

	    result_entry
	        = entry_factory->create_memory (address_value->to_long (), 0,
	                                        aspace_value->to_long ());
	  }
	  break;

	case DW_OP_LLVM_offset:
	  {
	    dwarf_value *offset_value;
	    dwarf_location *location;

	    offset_value
	        = entry_factory->entry_to_value (fetch (0), address_type);
	    pop ();

	    dwarf_require_integral (offset_value->get_type ());

	    location
	        = entry_factory->entry_to_location (fetch (0));
	    pop ();

	    location->add_bit_offset (offset_value->to_long () * 8);
	    result_entry = location;
	  }
	  break;

	case DW_OP_LLVM_offset_constu:
	  {
	    dwarf_location *location;

	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &uoffset);
	    result = uoffset;

	    location
	        = entry_factory->entry_to_location (fetch (0));
	    pop ();

	    location->add_bit_offset (result * 8);
	    result_entry = location;
	  }
	  break;

	case DW_OP_LLVM_bit_offset:
	  {
	    dwarf_value *offset_value;
	    dwarf_location *location;

	    offset_value
	        = entry_factory->entry_to_value (fetch (0), address_type);
	    pop ();

	    dwarf_require_integral (offset_value->get_type ());

	    location
	        = entry_factory->entry_to_location (fetch (0));
	    pop ();

	    location->add_bit_offset (offset_value->to_long ());
	    result_entry = location;
	  }
	  break;

	case DW_OP_LLVM_call_frame_entry_reg:
	  op_ptr = safe_read_uleb128 (op_ptr, op_end, &reg);

	  result_entry = entry_factory->create_register (reg, true);
	  eval_needs_frame ();
	  break;

	case DW_OP_LLVM_undefined:
	  result_entry = entry_factory->create_undefined ();
	  break;

	case DW_OP_LLVM_piece_end:
	  {
	    dwarf_composite *composite_entry;
	    dwarf_entry *entry = fetch (0);

	    if (!entry->is_entry (dwarf_entry::Composite))
	      ill_formed_expression ();

	    composite_entry = static_cast<dwarf_composite *> (entry);

	    if (composite_entry->is_completed ())
	      ill_formed_expression ();

	    composite_entry->set_completed (true);
	    goto no_push;
	  }

	case DW_OP_LLVM_select_bit_piece:
	  {
	    uint64_t piece_bit_size, pieces_count;

	    /* Record the piece.  */
	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &piece_bit_size);
	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &pieces_count);
	    result_entry
	        = create_select_composite (piece_bit_size, pieces_count);
	  }
	  break;

	case DW_OP_LLVM_extend:
	  {
	    uint64_t piece_bit_size, pieces_count;

	    /* Record the piece.  */
	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &piece_bit_size);
	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &pieces_count);
	    result_entry
	        = create_extend_composite (piece_bit_size, pieces_count);
	  }
	  break;

	case DW_OP_LLVM_aspace_bregx:
	  {
	    dwarf_location *location;
	    dwarf_memory *memory_entry;
	    dwarf_value *aspace_value;

	    op_ptr = safe_read_uleb128 (op_ptr, op_end, &reg);
	    op_ptr = safe_read_sleb128 (op_ptr, op_end, &offset);

	    location = entry_factory->create_register (op - DW_OP_breg0);
	    result_entry = dwarf_entry_deref (location, address_type);
	    location = entry_factory->entry_to_location (result_entry);

	    if (!location->is_entry (dwarf_entry::Memory))
	      ill_formed_expression ();

	    memory_entry = static_cast<dwarf_memory *> (location);
	    memory_entry->add_bit_offset (offset * 8);

	    aspace_value
	        = entry_factory->entry_to_value (fetch (0), address_type);
	    pop ();

	    dwarf_require_integral (aspace_value->get_type ());
	    memory_entry->set_address_space (aspace_value->to_long ());

	    result_entry = memory_entry;
	  }
	  break;

	default:
	  error (_("Unhandled dwarf expression opcode 0x%x"), op);
	}

      /* Most things push a result value.  */
      gdb_assert (result_entry != NULL);
      push (result_entry);
    no_push:
      ;
    }

  this->recursion_depth--;
  gdb_assert (this->recursion_depth >= 0);
}

void _initialize_dwarf2expr ();
void
_initialize_dwarf2expr ()
{
  dwarf_arch_cookie
    = gdbarch_data_register_post_init (dwarf_gdbarch_types_init);
}
