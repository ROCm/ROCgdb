/* Target-dependent code for the AMDGPU architectures.

   Copyright (C) 2019-2024 Free Software Foundation, Inc.
   Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.

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


#include "amd-dbgapi-target.h"
#include "amdgpu-tdep.h"
#include "arch-utils.h"
#include "disasm.h"
#include "dwarf2/frame.h"
#include "frame-unwind.h"
#include "gdbarch.h"
#include "gdbsupport/selftest.h"
#include "gdbtypes.h"
#include "inferior.h"
#include "language.h"
#include "producer.h"
#include "reggroups.h"
#include "extract-store-integer.h"

/* Bit mask of the address space information in the core address.  */
constexpr CORE_ADDR AMDGPU_ADDRESS_SPACE_MASK = 0xff00000000000000;

/* Bit offset from the start of the core address
   that represent the address space information.  */
constexpr unsigned int AMDGPU_ADDRESS_SPACE_BIT_OFFSET = 56;

/* See amdgpu-tdep.h.  */

bool
is_amdgpu_arch (struct gdbarch *arch)
{
  gdb_assert (arch != nullptr);
  return gdbarch_bfd_arch_info (arch)->arch == bfd_arch_amdgcn;
}

/* See amdgpu-tdep.h.  */

amdgpu_gdbarch_tdep *
get_amdgpu_gdbarch_tdep (gdbarch *arch)
{
  return gdbarch_tdep<amdgpu_gdbarch_tdep> (arch);
}

/* Return the name of register REGNUM.  */

static const char *
amdgpu_register_name (struct gdbarch *gdbarch, int regnum)
{
  /* The list of registers reported by amd-dbgapi for a given architecture
     contains some duplicate names.  For instance, there is an "exec" register
     for waves in the wave32 mode and one for the waves in the wave64 mode.
     However, at most one register with a given name is actually allocated for
     a specific wave.  If INFERIOR_PTID represents a GPU wave, we query
     amd-dbgapi to know whether the requested register actually exists for the
     current wave, so there won't be duplicates in the the register names we
     report for that wave.

     But there are two known cases where INFERIOR_PTID doesn't represent a GPU
     wave:

      - The user does "set arch amdgcn:gfxNNN" followed with "maint print
	registers"
      - The "register_name" selftest

     In these cases, we can't query amd-dbgapi to know whether we should hide
     the register or not.  The "register_name" selftest checks that there aren't
     duplicates in the register names returned by the gdbarch, so if we simply
     return all register names, that test will fail.  The other simple option is
     to never return a register name, which is what we do here.  */
  if (!ptid_is_gpu (inferior_ptid))
    return "";

  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (inferior_ptid);
  amdgpu_gdbarch_tdep *tdep = get_amdgpu_gdbarch_tdep (gdbarch);

  amd_dbgapi_register_exists_t register_exists;
  if (amd_dbgapi_wave_register_exists (wave_id, tdep->register_ids[regnum],
				       &register_exists)
	!= AMD_DBGAPI_STATUS_SUCCESS
      || register_exists != AMD_DBGAPI_REGISTER_PRESENT)
    return "";

  return tdep->register_names[regnum].c_str ();
}

/* Return the internal register number for the DWARF register number DWARF_REG.

   Return -1 if there's no internal register mapping to DWARF_REG.  */

static int
amdgpu_dwarf_reg_to_regnum (struct gdbarch *gdbarch, int dwarf_reg)
{
  amdgpu_gdbarch_tdep *tdep = get_amdgpu_gdbarch_tdep (gdbarch);

  if (dwarf_reg < tdep->dwarf_regnum_to_gdb_regnum.size ())
    return tdep->dwarf_regnum_to_gdb_regnum[dwarf_reg];

  return -1;
}

/* Up to 32 registers can be used for argument passing purposes.  */
constexpr unsigned int AMDGCN_MAX_NUM_REGS_FOR_ARGS_RET = 32;

/* See https://llvm.org/docs/AMDGPUUsage.html#register-identifier */
constexpr unsigned int AMDGCN_VGPR0_WAVE32_REGNUM = 1536;
constexpr unsigned int AMDGCN_VGPR0_WAVE64_REGNUM = 2560;

/* VGPR registers are 32 bits wide.  */
constexpr int AMDGCN_VGPR_LEN = 4;

/* Return the register number of VGPR0 for the platform, which is the first
   register in which function arguments and return values are passed.  */
static int
first_regnum_for_arg_or_return_value (gdbarch *gdbarch, ptid_t ptid)
{
  const amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (ptid);

  size_t lanecount;
  if (amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_LANE_COUNT,
				sizeof (lanecount), &lanecount)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("Failed to fetch the number of lanes for current wave"));

  const unsigned int dwarf_register_number = [lanecount] ()
    {
      switch (lanecount)
	{
	case 32:
	  return AMDGCN_VGPR0_WAVE32_REGNUM;
	case 64:
	   return AMDGCN_VGPR0_WAVE64_REGNUM;
	default:
	   error (_("Unsupported wave length"));
	}
    }();

  const int regnum = amdgpu_dwarf_reg_to_regnum (gdbarch,
						 dwarf_register_number);
  gdb_assert (regnum != -1);
  return regnum;
}

namespace {

/* Describe how an argument should be placed in registers for argument passing
   or value return purposes.  */

class amdgcn_arg_placement
{
public:

  /* Map a portion of a value into a register.  */
  struct part_placement
  {
    part_placement (int offset, int size, int regno, bool sign_extend)
      : offset { offset }, size { size }, regno { regno }
    , sign_extend { sign_extend }
    {}

    /* Offset (in bytes) from the start of the containing structure.  */
    int offset;

    /* Size (in bytes) of the part to map to the register.  */
    int size;

    /* Register number, relative to the first register for argument
       passing (VGPR0).  Count starts at 0.  */
    int regno;

    /* If size is less than a register size (32 bits), indicate if the
       most significant bits should be sign extended (if true) or
       0 extended (if false).  */
    bool sign_extend;
  };

  /* Compute how a value of type TYPE should be placed in registers,
     assuming an unlimited number of registers.  If PACKED is true,
     aggregates are treated as a block, otherwise they are decomposed
     recursively into their elements.  */
  amdgcn_arg_placement (struct type *type, bool packed)
  {
    alloc_for_type (check_typedef (type), 0, packed);
  }

  const std::vector<part_placement> &allocation () const
  {
    return m_allocation;
  }

protected:
  /* List describing in which registers each part of a value should be
     placed.  */
  std::vector<part_placement> m_allocation;

  /* Allocate the next available register.

     It will contain data located at OFFSET bytes (from the beginning of the
     top most encompassing aggregate) and occupying SIZE bytes.
     If SIGN_EXTEND is true and SIZE is less than the size of the register,
     the value should be sign-extended to fill the entire register.  If
     SIGN_EXTEND is false and SIZE is less than the size of the register,
     the value should be 0-extended.  If the size of the part equals the
     size of a register, SIGN_EXTEND is ignored.  */
  void alloc_reg_for_part (int offset, int size, bool sign_extend)
  {
    m_allocation.emplace_back (offset, size, m_allocation.size (),
			       sign_extend);
  }

  /* Allocate registers for a value of type TYPE, which is placed at OFFSET
     bytes from the start of the top most encompassing aggregate.  If PACKED,
     aggregate values should not be decomposed into their members.  */
  void alloc_for_type (struct type *type, int offset, bool packed);

  /* Allocate registers for a struct type.  */
  void alloc_for_struct (struct type *type, int offset);

  /* Allocate registers for an array type.  */
  void alloc_for_array (struct type *type, int offset);
};

void
amdgcn_arg_placement::alloc_for_type (struct type *type, int offset,
				      bool packed)
{
  type = check_typedef (type);
  if (!packed)
    switch (type->code ())
      {
      case TYPE_CODE_ARRAY:
	alloc_for_array (type, offset);
	return;
      case TYPE_CODE_STRUCT:
	alloc_for_struct (type, offset);
	return;
      }

  const int typelen = type->length ();
  /* Non-aggregate non packed type whose size is under 1 register might
     have to be sign extended.  */
  if (!packed && is_integral_type (type) && typelen < AMDGCN_VGPR_LEN)
    {
      /* For the purpose of calling convention, an enum is treated as its
	 underlying type.  */
      if (type->code () == TYPE_CODE_ENUM)
	type = check_typedef (type->target_type ());

      bool sign_extend = false;
      if ((type->code () == TYPE_CODE_CHAR && !type->has_no_signedness ())
	  || type->code () == TYPE_CODE_INT)
	sign_extend = !type->is_unsigned ();

      alloc_reg_for_part (offset, typelen, sign_extend);
    }

  /* Non aggregate or packed type.  */
  else
    {
      int type_offset = 0;
      while (type_offset < typelen)
	{
	  alloc_reg_for_part (offset + type_offset,
			      std::min (AMDGCN_VGPR_LEN,
					typelen - type_offset),
			      false);
	  type_offset += AMDGCN_VGPR_LEN;
	}
    }
}

void
amdgcn_arg_placement::alloc_for_struct (struct type *type, int offset)
{
  /* We normally allocate registers for the members of the struct.  An empty
     struct, or a struct only containing static fields, should still be
     allocated 1 register even if it has no members.  For this purpose, keep
     track of the presence of non static fields so we can adjust things at the
     end if no allocatable members have been seen.  */
  bool has_non_static_fields = false;

  int field_num = 0;
  while (field_num < type->num_fields ())
    {
      /* Allocation units (which contains bitfields) are packed.  */
      if (type->field (field_num).is_packed ())
	{
	  has_non_static_fields = true;
	  /* The first element of the pack must be byte aligned.  */
	  gdb_assert
	    (type->field (field_num).loc_bitpos () % HOST_CHAR_BIT == 0);
	  int bf_offset
	    = type->field (field_num).loc_bitpos () / HOST_CHAR_BIT;

	  int pack_end = field_num;
	  int pack_size = 0;
	  while (pack_end < type->num_fields ()
		 && type->field (pack_end).is_packed ())
	    {
	      pack_size += type->field (pack_end).bitsize ();
	      pack_end++;
	    }

	  /* From here on, the size is expressed in bytes.  */
	  pack_size = (pack_size + (HOST_CHAR_BIT - 1)) / HOST_CHAR_BIT;

	  /* Place the entire allocation unit packed in adjacent registers.  */
	  while (pack_size > 0)
	    {
	      alloc_reg_for_part (offset + bf_offset,
				  std::min (AMDGCN_VGPR_LEN, pack_size),
				  false);
	      bf_offset += std::min (AMDGCN_VGPR_LEN, pack_size);
	      pack_size -= AMDGCN_VGPR_LEN;
	    }

	  field_num = pack_end;
	}

      else
	{
	  const field &field = type->field (field_num);
	  struct type *field_type = check_typedef (field.type ());

	  /* Ignore static fields.  */
	  if (!type->field (field_num).is_static ())
	    {
	      has_non_static_fields = true;
	      gdb_assert (field.loc_bitpos () % HOST_CHAR_BIT == 0);
	      const int field_offset = field.loc_bitpos () / HOST_CHAR_BIT;
	      alloc_for_type (field_type, offset + field_offset, false);
	    }
	  field_num++;
	}
    }

  if (!has_non_static_fields)
    {
      if (type->length () != 1)
	warning (_("Empty struct should have a length of 1 byte.  "
		   "Assuming a size of 1 for ABI purposes."));

      alloc_reg_for_part (offset, 1, false);
    }
}

void
amdgcn_arg_placement::alloc_for_array (struct type *type, int offset)
{
  const int typelen = type->length ();
  if (type->is_vector () && typelen <= 2)
    alloc_reg_for_part (offset, 2, false);
  else
    {
      /* Check how one element of the array is mapped to registers.  */
      const amdgcn_arg_placement elem_placement (type->target_type (),
						 false);

      /* Repeat the same mapping pattern for each element of the array.  */
      const int element_type_length = type->target_type ()->length ();
      const unsigned element_align = type_align (type->target_type ());
      const int padding
	= ((element_align - (element_type_length % element_align))
	   % element_align);

      int array_offset = 0;
      while (array_offset < typelen)
	{
	  for (const auto & part_allocation : elem_placement.allocation ())
	    alloc_reg_for_part (offset + array_offset + part_allocation.offset,
				part_allocation.size,
				part_allocation.sign_extend);

	  array_offset += element_type_length + padding;
	}
    }
}

} // anonymous namespace

/* Checks if the type TYPE contains a flexible array member.  */

static bool
has_flexible_array_member (type *type)
{
  type = check_typedef (type);
  if (type->code () != TYPE_CODE_STRUCT)
    return false;

  /* A struct with a flexible array must have at least one other named field.
   */
  if (type->num_fields () < 2)
    return false;

  /* A flexible array member, if present, has to be the last element of the
     record.  */
  const field &last_member = type->field (type->num_fields () - 1);
  struct type *last_member_type = check_typedef (last_member.type ());
  if (last_member_type->code () != TYPE_CODE_ARRAY)
    return false;

  /* For a flexible array, we have a default lower bound set to 0, and an
     unknown upper bound (i.e. missing upper bound).  */
  return
    last_member_type->index_type ()->bounds ()->high.kind () == PROP_UNDEFINED;
}

/* Helper function for amdgpu_return_value.  */

static void
amdgcn_return_value_load_store (gdbarch *gdbarch, regcache *regcache,
				const amdgcn_arg_placement &alloc,
				gdb_byte *readbuf, const gdb_byte *writebuf)
{
  gdb_assert (regcache != nullptr);

  const int base_regno
    = first_regnum_for_arg_or_return_value (gdbarch, regcache->ptid ());
  const int lanenumber
    = current_inferior ()->find_thread
	(regcache->ptid ())->current_simd_lane ();

  for (const auto &piece : alloc.allocation ())
    {
      if (piece.size == 0)
	continue;

      if (readbuf != nullptr)
	regcache->raw_read_part
	  (base_regno + piece.regno, lanenumber * AMDGCN_VGPR_LEN,
	   piece.size, readbuf + piece.offset);

      if (writebuf != nullptr)
	{
	  gdb_byte regval[AMDGCN_VGPR_LEN] = {0x0, 0x0, 0x0, 0x0};

	  int i = 0;
	  for (; i < piece.size; ++i)
	    regval[i] = writebuf[piece.offset + i];
	  if (piece.sign_extend && writebuf[piece.offset + i - 1] & 0x80)
	    for (; i < AMDGCN_VGPR_LEN; ++i)
	      regval[i] = 0xff;

	  regcache->raw_write_part
	    (base_regno + piece.regno, lanenumber * AMDGCN_VGPR_LEN,
	     AMDGCN_VGPR_LEN, regval);
	}
    }
}

/* Handle return values from AMDGCN.

   Based on the de facto ABI hipcc uses.  This is described in LLVM
   clang/lib/CodeGen/TargetInfo.cpp in the following method:
   - ABIArgInfo AMDGPUABIInfo::classifyReturnType(QualType RetTy) const

   Partial description of the calling convention is available at [1].

   [1] https://llvm.org/docs/AMDGPUUsage.html#non-kernel-functions
*/
static enum return_value_convention
amdgpu_return_value_as_value (gdbarch *arch, value *function, type *valtype,
			      regcache *regcache, value **read_value,
			      const gdb_byte *writebuf)
{
  gdb_assert (gdbarch_bfd_arch_info (arch)->arch == bfd_arch_amdgcn);
  gdb_byte *readbuf = nullptr;

  valtype = check_typedef (valtype);

  if (read_value != nullptr)
    {
      *read_value = value::allocate (valtype);
      readbuf = (*read_value)->contents_raw ().data ();
    }

  /* Non-trivial objects are not returned by value.  */
  if (!language_pass_by_reference (valtype).trivially_copyable)
    return RETURN_VALUE_STRUCT_CONVENTION;

  /* Struct with flexible array are never returned by value.  */
  if (has_flexible_array_member (valtype))
    return RETURN_VALUE_STRUCT_CONVENTION;

  /* Nothing particular to do for empty structs.  We still use
     RETURN_VALUE_REGISTER_CONVENTION so GDB can display the (empty) value.  */
  if (valtype->code () == TYPE_CODE_STRUCT)
    {
      bool has_non_static_fields = false;
      for (int i = 0; i < valtype->num_fields (); ++i)
	has_non_static_fields
	  = has_non_static_fields || !valtype->field (i).is_static ();

      if (!has_non_static_fields)
	return RETURN_VALUE_REGISTER_CONVENTION;
    }

  /* Types of size under 8 bytes are returned packed in v0-1.  */
  if (valtype->length () <= 2 * AMDGCN_VGPR_LEN)
    {
      if (regcache != nullptr)
	{
	  /* Pack aggregates, but not scalar types so sign extension can be
	     done if necessary.  */
	  const bool pack = (valtype->code () == TYPE_CODE_STRUCT
			     ||valtype->code () == TYPE_CODE_ARRAY);
	  const amdgcn_arg_placement alloc (valtype, pack);

	  amdgcn_return_value_load_store (arch, regcache,
					  alloc, readbuf, writebuf);
	}
      return RETURN_VALUE_REGISTER_CONVENTION;
    }

  const amdgcn_arg_placement alloc (valtype, false);
  if (alloc.allocation ().size () <= AMDGCN_MAX_NUM_REGS_FOR_ARGS_RET)
    {
      if (regcache != nullptr)
	amdgcn_return_value_load_store (arch, regcache, alloc,
					readbuf, writebuf);
      return RETURN_VALUE_REGISTER_CONVENTION;
    }

  /* The value is passed in a way GDB cannot handle.  Ignoring it.  */
  return RETURN_VALUE_STRUCT_CONVENTION;
}

/* A hierarchy of classes to represent an amd-dbgapi register type.  */

struct amd_dbgapi_register_type
{
  enum class kind
    {
      INTEGER,
      FLOAT,
      DOUBLE,
      VECTOR,
      CODE_PTR,
      FLAGS,
      ENUM,
    };

  amd_dbgapi_register_type (kind kind, std::string lookup_name)
    : m_kind (kind), m_lookup_name (std::move (lookup_name))
  {}

  virtual ~amd_dbgapi_register_type () = default;

  /* Return the type's kind.  */
  kind kind () const
  { return m_kind; }

  /* Name to use for this type in the existing type map.  */
  const std::string &lookup_name () const
  { return m_lookup_name; }

private:
  enum kind m_kind;
  std::string m_lookup_name;
};

using amd_dbgapi_register_type_up = std::unique_ptr<amd_dbgapi_register_type>;

struct amd_dbgapi_register_type_integer : public amd_dbgapi_register_type
{
  amd_dbgapi_register_type_integer (bool is_unsigned, unsigned int bit_size)
    : amd_dbgapi_register_type
	(kind::INTEGER,
	 string_printf ("%sint%d", is_unsigned ? "u" : "", bit_size)),
      m_is_unsigned (is_unsigned),
      m_bit_size (bit_size)
  {}

  bool is_unsigned () const
  { return m_is_unsigned; }

  unsigned int bit_size () const
  { return m_bit_size; }

private:
  bool m_is_unsigned;
  unsigned int m_bit_size;
};

struct amd_dbgapi_register_type_float : public amd_dbgapi_register_type
{
  amd_dbgapi_register_type_float ()
    : amd_dbgapi_register_type (kind::FLOAT, "float")
  {}
};

struct amd_dbgapi_register_type_double : public amd_dbgapi_register_type
{
  amd_dbgapi_register_type_double ()
    : amd_dbgapi_register_type (kind::DOUBLE, "double")
  {}
};

struct amd_dbgapi_register_type_vector : public amd_dbgapi_register_type
{
  amd_dbgapi_register_type_vector (const amd_dbgapi_register_type &element_type,
				   unsigned int count)
    : amd_dbgapi_register_type (kind::VECTOR,
				make_lookup_name (element_type, count)),
      m_element_type (element_type),
      m_count (count)
  {}

  const amd_dbgapi_register_type &element_type () const
  { return m_element_type; }

  unsigned int count () const
  { return m_count; }

  static std::string make_lookup_name
    (const amd_dbgapi_register_type &element_type, unsigned int count)
  {
    return string_printf ("%s[%d]", element_type.lookup_name ().c_str (),
			  count);
  }

private:
  const amd_dbgapi_register_type &m_element_type;
  unsigned int m_count;
};

struct amd_dbgapi_register_type_code_ptr : public amd_dbgapi_register_type
{
  amd_dbgapi_register_type_code_ptr ()
    : amd_dbgapi_register_type (kind::CODE_PTR, "void (*)()")
  {}
};

struct amd_dbgapi_register_type_flags : public amd_dbgapi_register_type
{
  struct field
  {
    std::string name;
    unsigned int bit_pos_start;
    unsigned int bit_pos_end;
    const amd_dbgapi_register_type *type;
  };

  using container_type = std::vector<field>;
  using const_iterator_type = container_type::const_iterator;

  amd_dbgapi_register_type_flags (unsigned int bit_size, std::string_view name)
    : amd_dbgapi_register_type (kind::FLAGS,
				make_lookup_name (bit_size, name)),
      m_bit_size (bit_size),
      m_name (std::move (name))
  {}

  unsigned int bit_size () const
  { return m_bit_size; }

  void add_field (std::string name, unsigned int bit_pos_start,
		  unsigned int bit_pos_end,
		  const amd_dbgapi_register_type *type)
  {
    m_fields.push_back (field {std::move (name), bit_pos_start,
			       bit_pos_end, type});
  }

  container_type::size_type size () const
  { return m_fields.size (); }

  const field &operator[] (container_type::size_type pos) const
  { return m_fields[pos]; }

  const_iterator_type begin () const
  { return m_fields.begin (); }

  const_iterator_type end () const
  { return m_fields.end (); }

  const std::string &name () const
  { return m_name; }

  static std::string make_lookup_name (int bits, std::string_view name)
  {
    std::string res = string_printf ("flags%d_t ", bits);
    res.append (name.data (), name.size ());
    return res;
  }

private:
  unsigned int m_bit_size;
  container_type m_fields;
  std::string m_name;
};

using amd_dbgapi_register_type_flags_up
  = std::unique_ptr<amd_dbgapi_register_type_flags>;

struct amd_dbgapi_register_type_enum : public amd_dbgapi_register_type
{
  struct enumerator
  {
    std::string name;
    ULONGEST value;
  };

  using container_type = std::vector<enumerator>;
  using const_iterator_type = container_type::const_iterator;

  amd_dbgapi_register_type_enum (std::string_view name)
    : amd_dbgapi_register_type (kind::ENUM, make_lookup_name (name)),
      m_name (name.data (), name.length ())
  {}

  void set_bit_size (int bit_size)
  { m_bit_size = bit_size; }

  unsigned int bit_size () const
  { return m_bit_size; }

  void add_enumerator (std::string name, ULONGEST value)
  { m_enumerators.push_back (enumerator {std::move (name), value}); }

  container_type::size_type size () const
  { return m_enumerators.size (); }

  const enumerator &operator[] (container_type::size_type pos) const
  { return m_enumerators[pos]; }

  const_iterator_type begin () const
  { return m_enumerators.begin (); }

  const_iterator_type end () const
  { return m_enumerators.end (); }

  const std::string &name () const
  { return m_name; }

  static std::string make_lookup_name (std::string_view name)
  {
    std::string res = "enum ";
    res.append (name.data (), name.length ());
    return res;
  }

private:
  unsigned int m_bit_size = 32;
  container_type m_enumerators;
  std::string m_name;
};

using amd_dbgapi_register_type_enum_up
  = std::unique_ptr<amd_dbgapi_register_type_enum>;

/* Map type lookup names to types.  */
using amd_dbgapi_register_type_map
  = std::unordered_map<std::string, amd_dbgapi_register_type_up>;

/* Parse S as a ULONGEST, raise an error on overflow.  */

static ULONGEST
try_strtoulst (std::string_view s)
{
  errno = 0;
  ULONGEST value = strtoulst (s.data (), nullptr, 0);
  if (errno != 0)
    error (_("Failed to parse integer."));

  return value;
};

/* Shared regex bits.  */
#define IDENTIFIER "[A-Za-z0-9_.]+"
#define WS "[ \t]+"
#define WSOPT "[ \t]*"

static const amd_dbgapi_register_type &
parse_amd_dbgapi_register_type (std::string_view type_name,
				amd_dbgapi_register_type_map &type_map);


/* parse_amd_dbgapi_register_type helper for enum types.  */

static void
parse_amd_dbgapi_register_type_enum_fields
  (amd_dbgapi_register_type_enum &enum_type, std::string_view fields)
{
  compiled_regex regex (/* name */
			"^(" IDENTIFIER ")"
			WSOPT "=" WSOPT
			/* value */
			"([0-9]+)"
			WSOPT "(," WSOPT ")?",
			REG_EXTENDED,
			_("Error in AMDGPU enum register type regex"));
  regmatch_t matches[4];

  while (!fields.empty ())
    {
      int res = regex.exec (fields.data (), ARRAY_SIZE (matches), matches, 0);
      if (res == REG_NOMATCH)
	error (_("Failed to parse enum fields"));

      auto sv_from_match = [fields] (const regmatch_t &m)
	{ return fields.substr (m.rm_so, m.rm_eo - m.rm_so); };

      std::string_view name = sv_from_match (matches[1]);
      std::string_view value_str = sv_from_match (matches[2]);
      ULONGEST value = try_strtoulst (value_str);

      if (value > std::numeric_limits<uint32_t>::max ())
	enum_type.set_bit_size (64);

      enum_type.add_enumerator (std::string (name), value);

      fields = fields.substr (matches[0].rm_eo);
    }
}

/* parse_amd_dbgapi_register_type helper for flags types.  */

static void
parse_amd_dbgapi_register_type_flags_fields
  (amd_dbgapi_register_type_flags &flags_type,
   int bits, std::string_view name, std::string_view fields,
   amd_dbgapi_register_type_map &type_map)
{
  gdb_assert (bits == 32 || bits == 64);

  std::string regex_str
    = string_printf (/* type */
		     "^(bool|uint%d_t|enum" WS IDENTIFIER WSOPT "(\\{[^}]*})?)"
		     WS
		     /* name */
		     "(" IDENTIFIER ")" WSOPT
		     /* bit position */
		     "@([0-9]+)(-[0-9]+)?" WSOPT ";" WSOPT,
		     bits);
  compiled_regex regex (regex_str.c_str (), REG_EXTENDED,
			_("Error in AMDGPU register type flags fields regex"));
  regmatch_t matches[6];

  while (!fields.empty ())
    {
      int res = regex.exec (fields.data (), ARRAY_SIZE (matches), matches, 0);
      if (res == REG_NOMATCH)
	error (_("Failed to parse flags type fields string"));

      auto sv_from_match = [fields] (const regmatch_t &m)
	{ return fields.substr (m.rm_so, m.rm_eo - m.rm_so); };

      std::string_view field_type_str = sv_from_match (matches[1]);
      std::string_view field_name = sv_from_match (matches[3]);
      std::string_view pos_begin_str = sv_from_match (matches[4]);
      ULONGEST pos_begin = try_strtoulst (pos_begin_str);

      if (field_type_str == "bool")
	flags_type.add_field (std::string (field_name), pos_begin, pos_begin,
			      nullptr);
      else
	{
	  if (matches[5].rm_so == -1)
	    error (_("Missing end bit position"));

	  std::string_view pos_end_str = sv_from_match (matches[5]);
	  ULONGEST pos_end = try_strtoulst (pos_end_str.substr (1));
	  const amd_dbgapi_register_type &field_type
	    = parse_amd_dbgapi_register_type (field_type_str, type_map);
	  flags_type.add_field (std::string (field_name), pos_begin, pos_end,
				&field_type);
	}

      fields = fields.substr (matches[0].rm_eo);
    }
}

/* parse_amd_dbgapi_register_type helper for scalars.  */

static const amd_dbgapi_register_type &
parse_amd_dbgapi_register_type_scalar (std::string_view name,
				       amd_dbgapi_register_type_map &type_map)
{
  std::string name_str (name);
  auto it = type_map.find (name_str);
  if (it != type_map.end ())
    {
      enum amd_dbgapi_register_type::kind kind = it->second->kind ();
      if (kind != amd_dbgapi_register_type::kind::INTEGER
	  && kind != amd_dbgapi_register_type::kind::FLOAT
	  && kind != amd_dbgapi_register_type::kind::DOUBLE
	  && kind != amd_dbgapi_register_type::kind::CODE_PTR)
	error (_("type mismatch"));

      return *it->second;
    }

  amd_dbgapi_register_type_up type;
  if (name == "int32_t")
    type.reset (new amd_dbgapi_register_type_integer (false, 32));
  else if (name == "uint32_t")
    type.reset (new amd_dbgapi_register_type_integer (true, 32));
  else if (name == "int64_t")
    type.reset (new amd_dbgapi_register_type_integer (false, 64));
  else if (name == "uint64_t")
    type.reset (new amd_dbgapi_register_type_integer (true, 64));
  else if (name == "float")
    type.reset (new amd_dbgapi_register_type_float ());
  else if (name == "double")
    type.reset (new amd_dbgapi_register_type_double ());
  else if (name == "void (*)()")
    type.reset (new amd_dbgapi_register_type_code_ptr ());
  else
    error (_("unknown type %s"), name_str.c_str ());

  auto insertion_pair = type_map.emplace (name, std::move (type));
  return *insertion_pair.first->second;
}

/* Parse an amd-dbgapi register type string into an amd_dbgapi_register_type
   object.

   See the documentation of AMD_DBGAPI_REGISTER_INFO_TYPE in amd-dbgapi.h for
   details about the format.  */

static const amd_dbgapi_register_type &
parse_amd_dbgapi_register_type (std::string_view type_str,
				amd_dbgapi_register_type_map &type_map)
{
  size_t pos_open_bracket = type_str.find_last_of ('[');
  auto sv_from_match = [type_str] (const regmatch_t &m)
    { return type_str.substr (m.rm_so, m.rm_eo - m.rm_so); };

  if (pos_open_bracket != std::string_view::npos)
    {
      /* Vector types.  */
      std::string_view element_type_str
	= type_str.substr (0, pos_open_bracket);
      const amd_dbgapi_register_type &element_type
	= parse_amd_dbgapi_register_type (element_type_str, type_map);

      size_t pos_close_bracket = type_str.find_last_of (']');
      gdb_assert (pos_close_bracket != std::string_view::npos);
      std::string_view count_str_view
	= type_str.substr (pos_open_bracket + 1,
			    pos_close_bracket - pos_open_bracket);
      std::string count_str (count_str_view);
      unsigned int count = std::stoul (count_str);

      std::string lookup_name
	= amd_dbgapi_register_type_vector::make_lookup_name (element_type, count);
      auto existing_type_it = type_map.find (lookup_name);
      if (existing_type_it != type_map.end ())
	{
	  gdb_assert (existing_type_it->second->kind ()
		      == amd_dbgapi_register_type::kind::VECTOR);
	  return *existing_type_it->second;
	}

      amd_dbgapi_register_type_up type
	(new amd_dbgapi_register_type_vector (element_type, count));
      auto insertion_pair
	= type_map.emplace (type->lookup_name (), std::move (type));
      return *insertion_pair.first->second;
    }

  if (type_str.find ("flags32_t") == 0 || type_str.find ("flags64_t") == 0)
    {
      /* Split 'type_str' into 4 tokens: "(type) (name) ({ (fields) })".  */
      compiled_regex regex ("^(flags32_t|flags64_t)"
			    WS "(" IDENTIFIER ")" WSOPT
			    "(\\{" WSOPT "(.*)})?",
			    REG_EXTENDED,
			    _("Error in AMDGPU register type regex"));

      regmatch_t matches[5];
      int res = regex.exec (type_str.data (), ARRAY_SIZE (matches), matches, 0);
      if (res == REG_NOMATCH)
	error (_("Failed to parse flags type string"));

      std::string_view flags_keyword = sv_from_match (matches[1]);
      unsigned int bit_size = flags_keyword == "flags32_t" ? 32 : 64;
      std::string_view name = sv_from_match (matches[2]);
      std::string lookup_name
	= amd_dbgapi_register_type_flags::make_lookup_name (bit_size, name);
      auto existing_type_it = type_map.find (lookup_name);

      if (matches[3].rm_so == -1)
	{
	  /* No braces, lookup existing type.  */
	  if (existing_type_it == type_map.end ())
	    error (_("reference to unknown type %s."),
		   std::string (name).c_str ());

	  if (existing_type_it->second->kind ()
	      != amd_dbgapi_register_type::kind::FLAGS)
	    error (_("type mismatch"));

	  return *existing_type_it->second;
	}
      else
	{
	  /* With braces, it's a definition.  */
	  if (existing_type_it != type_map.end ())
	    error (_("re-definition of type %s."),
		   std::string (name).c_str ());

	  amd_dbgapi_register_type_flags_up flags_type
	    (new amd_dbgapi_register_type_flags (bit_size, name));
	  std::string_view fields_without_braces = sv_from_match (matches[4]);

	  parse_amd_dbgapi_register_type_flags_fields
	    (*flags_type, bit_size, name, fields_without_braces, type_map);

	  auto insertion_pair
	    = type_map.emplace (flags_type->lookup_name (),
				std::move (flags_type));
	  return *insertion_pair.first->second;
	}
    }

  if (type_str.find ("enum") == 0)
    {
      compiled_regex regex ("^enum" WS "(" IDENTIFIER ")" WSOPT "(\\{" WSOPT "([^}]*)})?",
			    REG_EXTENDED,
			    _("Error in AMDGPU register type enum regex"));

      /* Split 'type_name' into 3 tokens: "(name) ( { (fields) } )".  */
      regmatch_t matches[4];
      int res = regex.exec (type_str.data (), ARRAY_SIZE (matches), matches, 0);
      if (res == REG_NOMATCH)
	error (_("Failed to parse flags type string"));

      std::string_view name = sv_from_match (matches[1]);

      std::string lookup_name
	= amd_dbgapi_register_type_enum::make_lookup_name (name);
      auto existing_type_it = type_map.find (lookup_name);

      if (matches[2].rm_so == -1)
	{
	  /* No braces, lookup existing type.  */
	  if (existing_type_it == type_map.end ())
	    error (_("reference to unknown type %s"),
		   std::string (name).c_str ());

	  if (existing_type_it->second->kind ()
	      != amd_dbgapi_register_type::kind::ENUM)
	    error (_("type mismatch"));

	  return *existing_type_it->second;
	}
      else
	{
	  /* With braces, it's a definition.  */
	  if (existing_type_it != type_map.end ())
	    error (_("re-definition of type %s"),
		   std::string (name).c_str ());

	  amd_dbgapi_register_type_enum_up enum_type
	    (new amd_dbgapi_register_type_enum (name));
	  std::string_view fields_without_braces = sv_from_match (matches[3]);

	  parse_amd_dbgapi_register_type_enum_fields
	    (*enum_type, fields_without_braces);

	  auto insertion_pair
	    = type_map.emplace (enum_type->lookup_name (),
				std::move (enum_type));
	  return *insertion_pair.first->second;
	}
    }

  return parse_amd_dbgapi_register_type_scalar (type_str, type_map);
}

/* Convert an amd_dbgapi_register_type object to a GDB type.  */

static type *
amd_dbgapi_register_type_to_gdb_type (const amd_dbgapi_register_type &type,
				      struct gdbarch *gdbarch)
{
  switch (type.kind ())
    {
    case amd_dbgapi_register_type::kind::INTEGER:
      {
	const auto &integer_type
	  = gdb::checked_static_cast<const amd_dbgapi_register_type_integer &>
	      (type);
	switch (integer_type.bit_size ())
	  {
	  case 32:
	    if (integer_type.is_unsigned ())
	      return builtin_type (gdbarch)->builtin_uint32;
	    else
	      return builtin_type (gdbarch)->builtin_int32;

	  case 64:
	    if (integer_type.is_unsigned ())
	      return builtin_type (gdbarch)->builtin_uint64;
	    else
	      return builtin_type (gdbarch)->builtin_int64;

	  default:
	    gdb_assert_not_reached ("invalid bit size");
	  }
      }

    case amd_dbgapi_register_type::kind::VECTOR:
      {
	const auto &vector_type
	  = gdb::checked_static_cast<const amd_dbgapi_register_type_vector &>
	      (type);
	struct type *element_type
	  = amd_dbgapi_register_type_to_gdb_type (vector_type.element_type (),
						  gdbarch);
	return init_vector_type (element_type, vector_type.count ());
      }

    case amd_dbgapi_register_type::kind::FLOAT:
      return builtin_type (gdbarch)->builtin_float;

    case amd_dbgapi_register_type::kind::DOUBLE:
      return builtin_type (gdbarch)->builtin_double;

    case amd_dbgapi_register_type::kind::CODE_PTR:
      return builtin_type (gdbarch)->builtin_func_ptr;

    case amd_dbgapi_register_type::kind::FLAGS:
      {
	const auto &flags_type
	  = gdb::checked_static_cast<const amd_dbgapi_register_type_flags &>
	      (type);
	struct type *gdb_type
	  = arch_flags_type (gdbarch, flags_type.name ().c_str (),
			     flags_type.bit_size ());

	for (const auto &field : flags_type)
	  {
	    if (field.type == nullptr)
	      {
		gdb_assert (field.bit_pos_start == field.bit_pos_end);
		append_flags_type_flag (gdb_type, field.bit_pos_start,
					field.name.c_str ());
	      }
	    else
	      {
		struct type *field_type
		  = amd_dbgapi_register_type_to_gdb_type (*field.type, gdbarch);
		gdb_assert (field_type != nullptr);
		append_flags_type_field
		  (gdb_type, field.bit_pos_start,
		   field.bit_pos_end - field.bit_pos_start + 1,
		   field_type, field.name.c_str ());
	      }
	  }

	return gdb_type;
      }

    case amd_dbgapi_register_type::kind::ENUM:
      {
	const auto &enum_type
	  = gdb::checked_static_cast<const amd_dbgapi_register_type_enum &>
	      (type);
	struct type *gdb_type
	  = (type_allocator (gdbarch)
	     .new_type (TYPE_CODE_ENUM, enum_type.bit_size (),
			enum_type.name ().c_str ()));

	gdb_type->alloc_fields (enum_type.size ());
	gdb_type->set_is_unsigned (true);

	for (size_t i = 0; i < enum_type.size (); ++i)
	  {
	    const auto &field = enum_type[i];
	    gdb_type->field (i).set_name (xstrdup (field.name.c_str ()));
	    gdb_type->field (i).set_loc_enumval (field.value);
	  }

	return gdb_type;
      }

    default:
      gdb_assert_not_reached ("unhandled amd_dbgapi_register_type kind");
    }
}

static type *
amdgpu_register_type (struct gdbarch *gdbarch, int regnum)
{
  amdgpu_gdbarch_tdep *tdep = get_amdgpu_gdbarch_tdep (gdbarch);

  if (tdep->register_types[regnum] == nullptr)
    {
      /* This is done lazily (not at gdbarch initialization time), because it
	 requires access to builtin_type, which can't be used while the gdbarch
	 is not fully initialized.  */
      char *bytes;
      amd_dbgapi_status_t status
	= amd_dbgapi_register_get_info (tdep->register_ids[regnum],
					AMD_DBGAPI_REGISTER_INFO_TYPE,
					sizeof (bytes), &bytes);
      if (status != AMD_DBGAPI_STATUS_SUCCESS)
	error (_("Failed to get register type from amd-dbgapi"));

      gdb::unique_xmalloc_ptr<char> bytes_holder (bytes);
      amd_dbgapi_register_type_map type_map;
      const amd_dbgapi_register_type &register_type
	= parse_amd_dbgapi_register_type (bytes, type_map);
      tdep->register_types[regnum]
	= amd_dbgapi_register_type_to_gdb_type (register_type, gdbarch);
      gdb_assert (tdep->register_types[regnum] != nullptr);
    }

  return tdep->register_types[regnum];
}

static int
amdgpu_register_reggroup_p (struct gdbarch *gdbarch, int regnum,
			    const reggroup *group)
{
  amdgpu_gdbarch_tdep *tdep = get_amdgpu_gdbarch_tdep (gdbarch);

  auto it = tdep->register_class_map.find (group->name ());
  if (it == tdep->register_class_map.end ())
    return group == all_reggroup;

  amd_dbgapi_register_class_state_t state;
  if (amd_dbgapi_register_is_in_register_class (it->second,
						tdep->register_ids[regnum],
						&state)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return group == all_reggroup;

  return (state == AMD_DBGAPI_REGISTER_CLASS_STATE_MEMBER
	  || group == all_reggroup);
}

static int
amdgpu_breakpoint_kind_from_pc (struct gdbarch *gdbarch, CORE_ADDR *)
{
  return get_amdgpu_gdbarch_tdep (gdbarch)->breakpoint_instruction_size;
}

static const gdb_byte *
amdgpu_sw_breakpoint_from_kind (struct gdbarch *gdbarch, int kind, int *size)
{
  *size = kind;
  return get_amdgpu_gdbarch_tdep (gdbarch)->breakpoint_instruction_bytes.get ();
}

struct amdgpu_frame_cache
{
  CORE_ADDR base;
  CORE_ADDR pc;
};

static amdgpu_frame_cache *
amdgpu_frame_cache (const frame_info_ptr &this_frame, void **this_cache)
{
  if (*this_cache != nullptr)
    return (struct amdgpu_frame_cache *) *this_cache;

  struct amdgpu_frame_cache *cache
    = FRAME_OBSTACK_ZALLOC (struct amdgpu_frame_cache);
  (*this_cache) = cache;

  cache->pc = get_frame_func (this_frame);
  cache->base = 0;

  return cache;
}

static void
amdgpu_frame_this_id (const frame_info_ptr &this_frame, void **this_cache,
		      frame_id *this_id)
{
  struct amdgpu_frame_cache *cache
    = amdgpu_frame_cache (this_frame, this_cache);

  if (get_frame_type (this_frame) == INLINE_FRAME)
    (*this_id) = frame_id_build (cache->base, cache->pc);
  else
    (*this_id) = outer_frame_id;

  frame_debug_printf ("this_frame=%d, type=%d, this_id=%s",
		      frame_relative_level (this_frame),
		      get_frame_type (this_frame),
		      this_id->to_string ().c_str ());
}

static frame_id
amdgpu_dummy_id (struct gdbarch *gdbarch, const frame_info_ptr &this_frame)
{
  return frame_id_build (0, get_frame_pc (this_frame));
}

static struct value *
amdgpu_frame_prev_register (const frame_info_ptr &this_frame, void **this_cache,
			    int regnum)
{
  return frame_unwind_got_register (this_frame, regnum, regnum);
}

static const frame_unwind amdgpu_frame_unwind = {
  "amdgpu",
  NORMAL_FRAME,
  default_frame_unwind_stop_reason,
  amdgpu_frame_this_id,
  amdgpu_frame_prev_register,
  nullptr,
  default_frame_sniffer,
  nullptr,
  nullptr,
};

static int
print_insn_amdgpu (bfd_vma memaddr, struct disassemble_info *info)
{
  gdb_disassemble_info *di
    = static_cast<gdb_disassemble_info *> (info->application_data);

  /* Try to read at most INSTRUCTION_SIZE bytes.  */

  amd_dbgapi_size_t instruction_size = gdbarch_max_insn_length (di->arch ());
  gdb::byte_vector buffer (instruction_size);

  /* read_memory_func doesn't support partial reads, so if the read
     fails, try one byte less, on and on until we manage to read
     something.  A case where this would happen is if we're trying to
     read the last instruction at the end of a file section and that
     instruction is smaller than the largest instruction.  */
  while (instruction_size > 0)
    {
      int ret = info->read_memory_func (memaddr, buffer.data (),
					instruction_size, info);
      if (ret == 0)
	break;

      --instruction_size;
    }

  if (instruction_size == 0)
    {
      info->memory_error_func (-1, memaddr, info);
      return -1;
    }

  amd_dbgapi_architecture_id_t architecture_id;
  amd_dbgapi_status_t status
    = amd_dbgapi_get_architecture (gdbarch_bfd_arch_info (di->arch ())->mach,
				   &architecture_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    return -1;

  auto symbolizer = [] (amd_dbgapi_symbolizer_id_t symbolizer_id,
			amd_dbgapi_global_address_t address,
			char **symbol_text) -> amd_dbgapi_status_t
  {
    gdb_disassemble_info *disasm_info
      = reinterpret_cast<gdb_disassemble_info *> (symbolizer_id);
    gdb_printing_disassembler *disasm
      = dynamic_cast<gdb_printing_disassembler *> (disasm_info);
    gdb_assert (disasm != nullptr);

    string_file string (disasm->stream ()->can_emit_style_escape ());
    print_address (disasm->arch (), address, &string);
    *symbol_text = xstrdup (string.c_str ());

    return AMD_DBGAPI_STATUS_SUCCESS;
  };
  auto symbolizer_id = reinterpret_cast<amd_dbgapi_symbolizer_id_t> (di);
  char *instruction_text = nullptr;
  status = amd_dbgapi_disassemble_instruction (architecture_id, memaddr,
					       &instruction_size,
					       buffer.data (),
					       &instruction_text,
					       symbolizer_id,
					       symbolizer);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    {
      size_t alignment;
      status = amd_dbgapi_architecture_get_info
	(architecture_id,
	 AMD_DBGAPI_ARCHITECTURE_INFO_MINIMUM_INSTRUCTION_ALIGNMENT,
	 sizeof (alignment), &alignment);
      if (status != AMD_DBGAPI_STATUS_SUCCESS)
	error (_("amd_dbgapi_architecture_get_info failed"));

      info->fprintf_func (di, "<illegal instruction>");

      /* Skip to the next valid instruction address.  */
      return align_up (memaddr + 1, alignment) - memaddr;
    }

  /* Print the instruction.  */
  info->fprintf_func (di, "%s", instruction_text);

  /* Free the memory allocated by the amd-dbgapi.  */
  xfree (instruction_text);

  return static_cast<int> (instruction_size);
}

/* Convert address space and segment address into a core address.  */
static CORE_ADDR
amdgpu_segment_address_to_core_address (arch_addr_space_id address_space_id,
					CORE_ADDR address)
{
  return (address & ~AMDGPU_ADDRESS_SPACE_MASK)
	 | (((CORE_ADDR) address_space_id) << AMDGPU_ADDRESS_SPACE_BIT_OFFSET);
}

/* Convert an integer to an address of a given segment address.  */
static CORE_ADDR
amdgpu_integer_to_address (struct gdbarch *gdbarch,
			   struct type *type, const gdb_byte *buf,
			   arch_addr_space_id address_space_id)
{
  return amdgpu_segment_address_to_core_address (address_space_id,
						 unpack_long (type, buf));
}

/* See amdgpu-tdep.h.  */

arch_addr_space_id
amdgpu_address_space_id_from_core_address (CORE_ADDR addr)
{
  return (addr & AMDGPU_ADDRESS_SPACE_MASK) >> AMDGPU_ADDRESS_SPACE_BIT_OFFSET;
}

/* See amdgpu-tdep.h.  */

CORE_ADDR
amdgpu_segment_address_from_core_address (CORE_ADDR addr)
{
  return addr & ~AMDGPU_ADDRESS_SPACE_MASK;
}

/* Address class to address space mapping.

   TODO: This is just a quick fix to make the address class hand
	 written test work.

	 Unfortunately, there are only two bits currently in the type
	 instance flags used for address classes, and this is not
	 enough to describe the OpenCL address classes.

	 Another problem for even a basic compiler address class
	 support is the fact that both private_lane and private_wave
	 address spaces can be mapped to the same private address
	 class.

	 We will have to ignore that problem for now and assume that a
	 private address class is always mapped to a private_lane
	 address space.

	 This implementation will not be backward compatible when the
	 compiler starts generating the address class type information.  */

constexpr unsigned int DWARF_GLOBAL_ADDR_CLASS = 0;
constexpr unsigned int DWARF_GENERIC_ADDR_CLASS = 1;
constexpr unsigned int DWARF_LOCAL_ADDR_CLASS = 3;
constexpr unsigned int DWARF_PRIVATE_LANE_ADDR_CLASS = 5;

/* Map DWARF2_ADDR_CLASS address class to type instance flags.  */
static type_instance_flags
amdgpu_address_class_type_flags (int byte_size, int dwarf2_addr_class)
{
  if (dwarf2_addr_class == DWARF_GENERIC_ADDR_CLASS)
    return TYPE_INSTANCE_FLAG_ADDRESS_CLASS_1
	   | TYPE_INSTANCE_FLAG_ADDRESS_CLASS_2;
  else if (dwarf2_addr_class == DWARF_LOCAL_ADDR_CLASS)
    return TYPE_INSTANCE_FLAG_ADDRESS_CLASS_1;
  else if (dwarf2_addr_class == DWARF_PRIVATE_LANE_ADDR_CLASS)
    return TYPE_INSTANCE_FLAG_ADDRESS_CLASS_2;

  return 0;
}

/* Map TYPE_FLAGS type instance flags to address class.  */
static unsigned int
amdgpu_type_flags_to_addr_class (type_instance_flags type_flags)
{
  if ((type_flags & TYPE_INSTANCE_FLAG_ADDRESS_CLASS_1)
      && (type_flags & TYPE_INSTANCE_FLAG_ADDRESS_CLASS_2))
    return DWARF_GENERIC_ADDR_CLASS;
  else if (type_flags & TYPE_INSTANCE_FLAG_ADDRESS_CLASS_1)
    return DWARF_LOCAL_ADDR_CLASS;
  else if (type_flags & TYPE_INSTANCE_FLAG_ADDRESS_CLASS_2)
    return DWARF_PRIVATE_LANE_ADDR_CLASS;

  /* With current limitations, we can only assume
     that the address class is global.  */
  return DWARF_GLOBAL_ADDR_CLASS;
}

/* Map TYPE_FLAGS type instance flags to address class name.

   TODO: The idea of a target resolving a language address class name
	 just seems wrong.

	 At the moment, we assume that the language is OpenCL and
	 provide matching address class names.  */
static const char*
amdgpu_address_class_type_flags_to_name (struct gdbarch *gdbarch,
					 type_instance_flags type_flags)
{
  unsigned int addr_class = amdgpu_type_flags_to_addr_class (type_flags);

  if (addr_class == DWARF_GENERIC_ADDR_CLASS)
    return "generic";
  if (addr_class == DWARF_LOCAL_ADDR_CLASS)
    return "local";
  else if (addr_class == DWARF_PRIVATE_LANE_ADDR_CLASS)
    return "private";
  else
    return "";
}

/* Form a core address from a TYPE pointer type information and BUF
   pointer value buffer.

   TODO: The address class information belongs to a type, which means
	 that an address class is a language based concept, so either
	 the language should define the address class numbers and
	 names or some kind of a language enumeration needs to be
	 passed in.

	 At the moment, we assume that the language is OpenCL and we
	 apply 1-1 mapping between its address classes and the AMDGPU
	 address spaces.  */
static CORE_ADDR
amdgpu_pointer_to_address (struct gdbarch *gdbarch,
			   struct type *type, const gdb_byte *buf)
{
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  CORE_ADDR address
    = extract_unsigned_integer (buf, type->length (), byte_order);
  unsigned int address_class
    = amdgpu_type_flags_to_addr_class (type->instance_flags ());

  /* Address might be in a converted format already, so even if the
     class is global, the address might have the address space part
     in it.  This happens in cases like 'p &local_array'."  */
  if (address_class == DWARF_GLOBAL_ADDR_CLASS)
    return address;

  /* In the current implementation, we shouldn't have a case where we
     have both type address class information as well as address
     space information in a core address.  */
  gdb_assert (!amdgpu_address_space_id_from_core_address (address));

  address = amdgpu_segment_address_from_core_address (address);

  return amdgpu_segment_address_to_core_address (address_class, address);
}

static CORE_ADDR
amdgpu_skip_prologue (struct gdbarch *gdbarch, CORE_ADDR start_pc)
{
  CORE_ADDR func_addr;

  /* See if we can determine the end of the prologue via the symbol table.
     If so, then return either PC, or the PC after the prologue, whichever
     is greater.  */
  if (find_pc_partial_function (start_pc, nullptr, &func_addr, nullptr))
    {
      CORE_ADDR post_prologue_pc
	= skip_prologue_using_sal (gdbarch, func_addr);
      struct compunit_symtab *cust = find_pc_compunit_symtab (func_addr);

      /* Clang always emits a line note before the prologue and another
	 one after.  We trust clang to emit usable line notes.  */
      if (post_prologue_pc != 0
	  && cust != nullptr
	  && cust->producer () != nullptr
	  && producer_is_llvm (cust->producer ()))
	return std::max (start_pc, post_prologue_pc);
    }

  return start_pc;
}

static gdb::array_view<const arch_addr_space>
amdgpu_address_spaces (struct gdbarch *gdbarch)
{
  amdgpu_gdbarch_tdep *tdep = get_amdgpu_gdbarch_tdep (gdbarch);
  return tdep->address_spaces;
}

static location_scope
amdgpu_address_scope (struct gdbarch *gdbarch, CORE_ADDR address)
{
  amd_dbgapi_segment_address_dependency_t segment_address_dependency;

  uint64_t dwarf_address_space
    = (uint64_t) amdgpu_address_space_id_from_core_address (address);

  amd_dbgapi_segment_address_t segment_address
    = amdgpu_segment_address_from_core_address (address);

  amd_dbgapi_architecture_id_t architecture_id;
  if (amd_dbgapi_get_architecture
      (gdbarch_bfd_arch_info (gdbarch)->mach, &architecture_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_get_architecture failed"));

  amd_dbgapi_address_space_id_t address_space_id;
  if (amd_dbgapi_dwarf_address_space_to_address_space (architecture_id,
						       dwarf_address_space,
						       &address_space_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_dwarf_address_space_to_address_space failed"));

  if (amd_dbgapi_address_dependency (address_space_id,
				     segment_address,
				     &segment_address_dependency)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_address_dependency failed"));

  switch (segment_address_dependency)
    {
    case AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_LANE:
      return LOCATION_SCOPE_LANE;

    case AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_WAVE:
      return LOCATION_SCOPE_THREAD;

    case AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_PROCESS:
      return LOCATION_SCOPE_INFERIOR;

    case AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_WORKGROUP:
    case AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_AGENT:
      /* GDB currently doesn't model workgroups or agents as first
	 class citizens, so the mapping here isn't perfect.  */
      return LOCATION_SCOPE_INFERIOR;

    default:
      error (_("unhandled segment address dependency kind"));
    }
}

/* Queries the Debug API for aliasing address ranges in the default
   address space (global memory) for a given address range in a
   specific address space.

   Based on the address space, an address range can alias to multiple
   smaller address ranges in the default address space.  In this case,
   watching a given address range means watching all aliasing address
   ranges instead.

   In practice, this means that watching an address range in a non
   default address space might take more hardware watchpoints then
   watching the same address range in the default address space.  */

static std::vector<addr_range>
amdgpu_get_watchable_aliases (struct gdbarch *gdbarch,
			      ptid_t ptid, int simd_lane,
			      addr_range range)
{
  CORE_ADDR addr = range.addr;
  arch_addr_space_id addr_space_id
    = amdgpu_address_space_id_from_core_address (addr);
  size_t size = range.size;

  /* Addresses from the default address space are always watchable.  */
  if (addr_space_id == ARCH_ADDR_SPACE_ID_DEFAULT)
    return {range};

  if (size == 0)
    error (_("Watchpoint length must be non-zero number."));

  /* We should never get a valid ptid here that is not a gpu
     thread.  */
  gdb_assert (ptid == null_ptid || ptid_is_gpu (ptid));

  std::vector<addr_range> aliases;
  amd_dbgapi_architecture_id_t architecture_id;
  if (amd_dbgapi_get_architecture (gdbarch_bfd_arch_info (gdbarch)->mach,
				   &architecture_id)
	!= AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_get_architecture failed"));

  /* Get a dbgapi address space id for the original address space.  */
  amd_dbgapi_address_space_id_t dbgapi_from_addr_space_id;
  if (amd_dbgapi_dwarf_address_space_to_address_space
	(architecture_id, (uint64_t) addr_space_id,
	 &dbgapi_from_addr_space_id)
	!= AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_dwarf_address_space_to_address_space failed"));

  /* Get the dbgapi address space id for the default address space.  */
  amd_dbgapi_address_space_id_t dbgapi_to_addr_space_id;
  gdb_assert (amd_dbgapi_dwarf_address_space_to_address_space
	       (architecture_id, (uint64_t) ARCH_ADDR_SPACE_ID_DEFAULT,
		&dbgapi_to_addr_space_id)
	       == AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_wave_id_t wave_id = AMD_DBGAPI_WAVE_NONE;

  if (ptid != null_ptid)
    wave_id = get_amd_dbgapi_wave_id (ptid);

  /* Aliasing address range might not have the same layout.  Because
     of that, when ever there is a gap between the aliasing addresses,
     create a new range.  */
  while (true)
    {
      amd_dbgapi_size_t converted_size;
      amd_dbgapi_segment_address_t to_offset;

      /* Try to convert the address to an address in the
	 default address space.  */
      if (amd_dbgapi_convert_address_space
	    (wave_id, simd_lane, dbgapi_from_addr_space_id,
	     (amd_dbgapi_segment_address_t) addr,
	     dbgapi_to_addr_space_id, &to_offset, &converted_size)
	    != AMD_DBGAPI_STATUS_SUCCESS)
	return {};

      if (converted_size == 0)
	return {};

      if (size > converted_size)
	{
	  aliases.emplace_back (to_offset, converted_size);
	  addr += converted_size;
	  size -= converted_size;
	}
      else
	{
	  aliases.emplace_back (to_offset, size);
	  break;
	}

    };

  return aliases;
}

static simd_lanes_mask_t
amdgpu_active_lanes_mask (struct gdbarch *gdbarch, thread_info *tp)
{
  static_assert (sizeof (simd_lanes_mask_t) >= sizeof (uint64_t));

  uint64_t exec_mask;
  if (wave_get_info (tp, AMD_DBGAPI_WAVE_INFO_EXEC_MASK, exec_mask)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return 0;

  return exec_mask;
}

static int
amdgpu_supported_lanes_count (struct gdbarch *gdbarch, thread_info *tp)
{
  size_t count;
  if (wave_get_info (tp, AMD_DBGAPI_WAVE_INFO_LANE_COUNT, count)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return 0;

  return count;
}

/* A client debugger can check if the lane is part of a valid
   work-group by checking that the lane is in the range of the
   associated work-group within the grid, accounting for partial
   work-groups.  */

static int
amdgpu_used_lanes_count (struct gdbarch *gdbarch, thread_info *tp)
{
  amd_dbgapi_dispatch_id_t dispatch_id;
  if (wave_get_info (tp, AMD_DBGAPI_WAVE_INFO_DISPATCH, dispatch_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    {
      /* The dispatch associated with a wave is not available.  A wave
	 may not have an associated dispatch if attaching to a process
	 with already existing waves.  In that case, all we can do is
	 claim that all lanes are used.  */
      return amdgpu_supported_lanes_count (gdbarch, tp);
    }

  uint32_t grid_sizes[3];
  dispatch_get_info_throw (dispatch_id,
			   AMD_DBGAPI_DISPATCH_INFO_GRID_SIZES,
			   grid_sizes);

  uint16_t work_group_sizes[3];
  dispatch_get_info_throw (dispatch_id,
			   AMD_DBGAPI_DISPATCH_INFO_WORKGROUP_SIZES,
			   work_group_sizes);

  uint32_t group_ids[3];
  wave_get_info_throw (tp, AMD_DBGAPI_WAVE_INFO_WORKGROUP_COORD, group_ids);

  uint32_t wave_in_group;
  wave_get_info_throw (tp, AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP,
		       wave_in_group);

  size_t lane_count;
  wave_get_info_throw (tp, AMD_DBGAPI_WAVE_INFO_LANE_COUNT, lane_count);

  size_t work_group_item_sizes[3];
  for (int i = 0; i < 3; i++)
    {
      size_t item_start = group_ids[i] * work_group_sizes[i];
      size_t item_end = item_start + work_group_sizes[i];
      if (item_end > grid_sizes[i])
	item_end = grid_sizes[i];
      work_group_item_sizes[i] = item_end - item_start;
    }

  size_t work_items = (work_group_item_sizes[0]
		       * work_group_item_sizes[1]
		       * work_group_item_sizes[2]);

  size_t work_items_left = work_items - wave_in_group * lane_count;
  return std::min (work_items_left, lane_count);
}

static bool
amdgpu_supports_arch_info (const struct bfd_arch_info *info)
{
  amd_dbgapi_architecture_id_t architecture_id;
  amd_dbgapi_status_t status
    = amd_dbgapi_get_architecture (info->mach, &architecture_id);

  gdb_assert (status != AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
  return status == AMD_DBGAPI_STATUS_SUCCESS;
}

static struct gdbarch *
amdgpu_gdbarch_init (struct gdbarch_info info, struct gdbarch_list *arches)
{
  /* If there is already a candidate, use it.  */
  arches = gdbarch_list_lookup_by_info (arches, &info);
  if (arches != nullptr)
    return arches->gdbarch;

  /* Allocate space for the new architecture.  */
  gdbarch_up gdbarch_u
    (gdbarch_alloc (&info, gdbarch_tdep_up (new amdgpu_gdbarch_tdep)));
  gdbarch *gdbarch = gdbarch_u.get ();
  amdgpu_gdbarch_tdep *tdep = gdbarch_tdep<amdgpu_gdbarch_tdep> (gdbarch);

  /* Data types.  */
  set_gdbarch_char_signed (gdbarch, 0);
  set_gdbarch_ptr_bit (gdbarch, 64);
  set_gdbarch_addr_bit (gdbarch, 64);
  set_gdbarch_short_bit (gdbarch, 16);
  set_gdbarch_int_bit (gdbarch, 32);
  set_gdbarch_long_bit (gdbarch, 64);
  set_gdbarch_long_long_bit (gdbarch, 64);
  set_gdbarch_float_bit (gdbarch, 32);
  set_gdbarch_double_bit (gdbarch, 64);
  set_gdbarch_long_double_bit (gdbarch, 128);
  set_gdbarch_half_format (gdbarch, floatformats_ieee_half);
  set_gdbarch_float_format (gdbarch, floatformats_ieee_single);
  set_gdbarch_double_format (gdbarch, floatformats_ieee_double);
  set_gdbarch_long_double_format (gdbarch, floatformats_ieee_double);

/* Address space handling.  */
  set_gdbarch_integer_to_address (gdbarch, amdgpu_integer_to_address);
  set_gdbarch_address_space_id_from_core_address
    (gdbarch, amdgpu_address_space_id_from_core_address);
  set_gdbarch_segment_address_from_core_address
    (gdbarch, amdgpu_segment_address_from_core_address);
  set_gdbarch_segment_address_to_core_address
    (gdbarch, amdgpu_segment_address_to_core_address);
  set_gdbarch_address_spaces (gdbarch, amdgpu_address_spaces);
  set_gdbarch_address_scope (gdbarch, amdgpu_address_scope);
  set_gdbarch_get_watchable_aliases (gdbarch, amdgpu_get_watchable_aliases);

  /* Frame interpretation.  */
  set_gdbarch_skip_prologue (gdbarch, amdgpu_skip_prologue);
  set_gdbarch_inner_than (gdbarch, core_addr_greaterthan);
  dwarf2_append_unwinders (gdbarch);
  frame_unwind_append_unwinder (gdbarch, &amdgpu_frame_unwind);
  set_gdbarch_dummy_id (gdbarch, amdgpu_dummy_id);

  set_gdbarch_pointer_to_address (gdbarch, amdgpu_pointer_to_address);
  set_gdbarch_address_class_type_flags
    (gdbarch, amdgpu_address_class_type_flags);
  set_gdbarch_address_class_type_flags_to_name
    (gdbarch, amdgpu_address_class_type_flags_to_name);

  /* Registers and memory.  */
  amd_dbgapi_architecture_id_t architecture_id;
  amd_dbgapi_status_t status
    = amd_dbgapi_get_architecture (gdbarch_bfd_arch_info (gdbarch)->mach,
				   &architecture_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    {
      warning (_("Failed to get architecture from amd-dbgapi"));
      return nullptr;
    }


  /* Add register groups.  */
  size_t register_class_count;
  amd_dbgapi_register_class_id_t *register_class_ids;
  status = amd_dbgapi_architecture_register_class_list (architecture_id,
							&register_class_count,
							&register_class_ids);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    {
      warning (_("Failed to get register class list from amd-dbgapi"));
      return nullptr;
    }

  gdb::unique_xmalloc_ptr<amd_dbgapi_register_class_id_t>
    register_class_ids_holder (register_class_ids);

  for (size_t i = 0; i < register_class_count; ++i)
    {
      char *bytes;
      status = amd_dbgapi_architecture_register_class_get_info
	(register_class_ids[i], AMD_DBGAPI_REGISTER_CLASS_INFO_NAME,
	 sizeof (bytes), &bytes);
      if (status != AMD_DBGAPI_STATUS_SUCCESS)
	{
	  warning (_("Failed to get register class name from amd-dbgapi"));
	  return nullptr;
	}

      gdb::unique_xmalloc_ptr<char> name (bytes);

      auto inserted = tdep->register_class_map.emplace (name.get (),
							register_class_ids[i]);
      gdb_assert (inserted.second);

      /* Avoid creating a user reggroup with the same name as some built-in
	 reggroup, such as "general", "system", "vector", etc.  */
      if (reggroup_find (gdbarch, name.get ()) != nullptr)
	continue;

      /* Allocate the reggroup in the gdbarch.  */
      reggroup_add
	(gdbarch, reggroup_gdbarch_new (gdbarch, name.get (), USER_REGGROUP));
    }

  /* Add registers. */
  size_t register_count;
  amd_dbgapi_register_id_t *register_ids;
  status = amd_dbgapi_architecture_register_list (architecture_id,
						  &register_count,
						  &register_ids);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    {
      warning (_("Failed to get register list from amd-dbgapi"));
      return nullptr;
    }

  gdb::unique_xmalloc_ptr<amd_dbgapi_register_id_t> register_ids_holder
    (register_ids);

  tdep->register_ids.insert (tdep->register_ids.end (), &register_ids[0],
			     &register_ids[register_count]);

  tdep->register_properties.resize (register_count,
				    AMD_DBGAPI_REGISTER_PROPERTY_NONE);
  for (size_t regnum = 0; regnum < register_count; ++regnum)
    {
      auto &register_properties = tdep->register_properties[regnum];
      if (amd_dbgapi_register_get_info (register_ids[regnum],
					AMD_DBGAPI_REGISTER_INFO_PROPERTIES,
					sizeof (register_properties),
					&register_properties)
	  != AMD_DBGAPI_STATUS_SUCCESS)
	{
	  warning (_("Failed to get register properties from amd-dbgapi"));
	  return nullptr;
	}
    }

  set_gdbarch_num_regs (gdbarch, register_count);
  set_gdbarch_num_pseudo_regs (gdbarch, 0);

  tdep->register_names.resize (register_count);
  tdep->register_types.resize (register_count);
  for (size_t i = 0; i < register_count; ++i)
    {
      /* Set amd-dbgapi register id -> gdb regnum mapping.  */
      tdep->regnum_map.emplace (tdep->register_ids[i], i);

      /* Get register name.  */
      char *bytes;
      status = amd_dbgapi_register_get_info (tdep->register_ids[i],
					     AMD_DBGAPI_REGISTER_INFO_NAME,
					     sizeof (bytes), &bytes);
      if (status == AMD_DBGAPI_STATUS_SUCCESS)
	{
	  tdep->register_names[i] = bytes;
	  xfree (bytes);
	}

      /* Get register DWARF number.  */
      uint64_t dwarf_num;
      status = amd_dbgapi_register_get_info (tdep->register_ids[i],
					     AMD_DBGAPI_REGISTER_INFO_DWARF,
					     sizeof (dwarf_num), &dwarf_num);
      if (status == AMD_DBGAPI_STATUS_SUCCESS)
	{
	  if (dwarf_num >= tdep->dwarf_regnum_to_gdb_regnum.size ())
	    tdep->dwarf_regnum_to_gdb_regnum.resize (dwarf_num + 1, -1);

	  tdep->dwarf_regnum_to_gdb_regnum[dwarf_num] = i;
	}
    }

  amd_dbgapi_register_id_t pc_register_id;
  status = amd_dbgapi_architecture_get_info
    (architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_PC_REGISTER,
     sizeof (pc_register_id), &pc_register_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    {
      warning (_("Failed to get PC register from amd-dbgapi"));
      return nullptr;
    }

  set_gdbarch_pc_regnum (gdbarch, tdep->regnum_map[pc_register_id]);
  set_gdbarch_ps_regnum (gdbarch, -1);
  set_gdbarch_sp_regnum (gdbarch, -1);
  set_gdbarch_fp0_regnum (gdbarch, -1);

  set_gdbarch_dwarf2_reg_to_regnum (gdbarch, amdgpu_dwarf_reg_to_regnum);

  set_gdbarch_return_value_as_value (gdbarch, amdgpu_return_value_as_value);

  /* Register representation.  */
  set_gdbarch_register_name (gdbarch, amdgpu_register_name);
  set_gdbarch_register_type (gdbarch, amdgpu_register_type);
  set_gdbarch_register_reggroup_p (gdbarch, amdgpu_register_reggroup_p);

  /* Disassembly.  */
  set_gdbarch_print_insn (gdbarch, print_insn_amdgpu);

 /* Instructions.  */
  amd_dbgapi_size_t max_insn_length = 0;
  status = amd_dbgapi_architecture_get_info
    (architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_LARGEST_INSTRUCTION_SIZE,
     sizeof (max_insn_length), &max_insn_length);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_architecture_get_info failed"));

  set_gdbarch_max_insn_length (gdbarch, max_insn_length);

  /* Lane debugging.  */
  set_gdbarch_active_lanes_mask (gdbarch, amdgpu_active_lanes_mask);
  set_gdbarch_supported_lanes_count (gdbarch, amdgpu_supported_lanes_count);
  set_gdbarch_used_lanes_count (gdbarch, amdgpu_used_lanes_count);

  status = amd_dbgapi_architecture_get_info
    (architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE,
     sizeof (tdep->breakpoint_instruction_size),
     &tdep->breakpoint_instruction_size);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_architecture_get_info failed"));

  gdb_byte *breakpoint_instruction_bytes;
  status = amd_dbgapi_architecture_get_info
    (architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION,
     sizeof (breakpoint_instruction_bytes), &breakpoint_instruction_bytes);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_architecture_get_info failed"));

  tdep->breakpoint_instruction_bytes.reset (breakpoint_instruction_bytes);

  set_gdbarch_breakpoint_kind_from_pc (gdbarch,
				       amdgpu_breakpoint_kind_from_pc);
  set_gdbarch_sw_breakpoint_from_kind (gdbarch,
				       amdgpu_sw_breakpoint_from_kind);

  amd_dbgapi_size_t pc_adjust;
  status = amd_dbgapi_architecture_get_info
    (architecture_id,
     AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_PC_ADJUST,
     sizeof (pc_adjust), &pc_adjust);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_architecture_get_info failed"));

  set_gdbarch_decr_pc_after_break (gdbarch, pc_adjust);

  /* Get info about address spaces.  */
  size_t address_space_count;
  amd_dbgapi_address_space_id_t *address_spaces;

  if (amd_dbgapi_architecture_address_space_list (architecture_id,
						  &address_space_count,
						  &address_spaces)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_architecture_address_space_list failed"));

  gdb::unique_xmalloc_ptr<amd_dbgapi_address_space_id_t[]> address_spaces_holder
    (address_spaces);

  for (size_t i = 0; i < address_space_count; ++i)
    {
      char *address_space_name;

      if (amd_dbgapi_address_space_get_info
	    (address_spaces[i], AMD_DBGAPI_ADDRESS_SPACE_INFO_NAME,
	     sizeof (address_space_name), &address_space_name)
	  != AMD_DBGAPI_STATUS_SUCCESS)
	error (_("amd_dbgapi_address_space_get_info (name) failed"));

      gdb_assert (address_space_name != nullptr);
      gdb::unique_xmalloc_ptr<char> address_space_name_holder
	(address_space_name);

      arch_addr_space_id address_space_dwarf_num;
      if (amd_dbgapi_address_space_get_info
	    (address_spaces[i], AMD_DBGAPI_ADDRESS_SPACE_INFO_DWARF,
	     sizeof (address_space_dwarf_num), &address_space_dwarf_num)
	  != AMD_DBGAPI_STATUS_SUCCESS)
	error (_("amd_dbgapi_address_space_get_info (dwarf) failed"));

      tdep->address_spaces.emplace_back (address_space_dwarf_num,
					 std::move (address_space_name_holder));
    }

  return gdbarch_u.release ();
}

#if defined GDB_SELF_TEST

static void
amdgpu_register_type_parse_test ()
{
  {
    /* A type that exercises flags and enums, in particular looking up an
       existing enum type by name. */
    const char *flags_type_str =
      "flags32_t mode { \
	 enum fp_round { \
	   NEAREST_EVEN = 0, \
	   PLUS_INF  = 1, \
	   MINUS_INF = 2, \
	   ZERO      = 3 \
	 } FP_ROUND.32 @0-1; \
	 enum fp_round FP_ROUND.64_16 @2-3; \
	 enum fp_denorm { \
	   FLUSH_SRC_DST = 0, \
	   FLUSH_DST     = 1, \
	   FLUSH_SRC     = 2, \
	   FLUSH_NONE    = 3 \
	 } FP_DENORM.32 @4-5; \
	 enum fp_denorm FP_DENORM.64_16 @6-7; \
	 bool DX10_CLAMP @8; \
	 bool IEEE @9; \
	 bool LOD_CLAMPED @10; \
	 bool DEBUG_EN @11; \
	 bool EXCP_EN.INVALID @12; \
	 bool EXCP_EN.DENORM @13; \
	 bool EXCP_EN.DIV0 @14; \
	 bool EXCP_EN.OVERFLOW @15; \
	 bool EXCP_EN.UNDERFLOW @16; \
	 bool EXCP_EN.INEXACT @17; \
	 bool EXCP_EN.INT_DIV0 @18; \
	 bool EXCP_EN.ADDR_WATCH @19; \
	 bool FP16_OVFL @23; \
	 bool POPS_PACKER0 @24; \
	 bool POPS_PACKER1 @25; \
	 bool DISABLE_PERF @26; \
	 bool GPR_IDX_EN @27; \
	 bool VSKIP @28; \
	 uint32_t CSP @29-31; \
       }";
    amd_dbgapi_register_type_map type_map;
    const amd_dbgapi_register_type &type
      = parse_amd_dbgapi_register_type (flags_type_str, type_map);

    gdb_assert (type.kind () == amd_dbgapi_register_type::kind::FLAGS);

    const auto &f
      = gdb::checked_static_cast<const amd_dbgapi_register_type_flags &> (type);
    gdb_assert (f.size () == 23);

    /* Check the two "FP_ROUND" fields.  */
    auto check_fp_round_field
      = [] (const char *name, const amd_dbgapi_register_type_flags::field &field)
	{
	  gdb_assert (field.name == name);
	  gdb_assert (field.type->kind ()
		      == amd_dbgapi_register_type::kind::ENUM);

	  const auto &e
	    = gdb::checked_static_cast<const amd_dbgapi_register_type_enum &>
	      (*field.type);
	  gdb_assert (e.size () == 4);
	  gdb_assert (e[0].name == "NEAREST_EVEN");
	  gdb_assert (e[0].value == 0);
	  gdb_assert (e[3].name == "ZERO");
	  gdb_assert (e[3].value == 3);
	};

    check_fp_round_field ("FP_ROUND.32", f[0]);
    check_fp_round_field ("FP_ROUND.64_16", f[1]);

    /* Check the "CSP" field.  */
    gdb_assert (f[22].name == "CSP");
    gdb_assert (f[22].type->kind () == amd_dbgapi_register_type::kind::INTEGER);

    const auto &i
      = gdb::checked_static_cast<const amd_dbgapi_register_type_integer &>
	  (*f[22].type);
    gdb_assert (i.bit_size () == 32);
    gdb_assert (i.is_unsigned ());
  }

  {
    /* Test the vector type.  */
    const char *vector_type_str = "int32_t[64]";
    amd_dbgapi_register_type_map type_map;
    const amd_dbgapi_register_type &type
      = parse_amd_dbgapi_register_type (vector_type_str, type_map);

    gdb_assert (type.kind () == amd_dbgapi_register_type::kind::VECTOR);

    const auto &v
      = gdb::checked_static_cast<const amd_dbgapi_register_type_vector &>
	  (type);
    gdb_assert (v.count () == 64);

    const auto &et = v.element_type ();
    gdb_assert (et.kind () == amd_dbgapi_register_type::kind::INTEGER);

    const auto &i
      = gdb::checked_static_cast<const amd_dbgapi_register_type_integer &> (et);
    gdb_assert (i.bit_size () == 32);
    gdb_assert (!i.is_unsigned ());
  }
}

#endif

void _initialize_amdgpu_tdep ();

void
_initialize_amdgpu_tdep ()
{
  gdbarch_register (bfd_arch_amdgcn, amdgpu_gdbarch_init, NULL,
		    amdgpu_supports_arch_info);
#if defined GDB_SELF_TEST
  selftests::register_test ("amdgpu-register-type-parse-flags-fields",
			    amdgpu_register_type_parse_test);
#endif
}
