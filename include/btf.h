/* btf.h - BTF support header file
   Copyright (C) 2026 Free Software Foundation, Inc.

   This file is part of GNU Binutils.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston,
   MA 02110-1301, USA.  */

#ifndef BTF_H
#define BTF_H

/* BTF magic number to identify header, endianness.  */
#define BTF_MAGIC	0xeb9f
/* Data format version number.  */
#define BTF_VERSION 	1

struct btf_header
{
  uint16_t magic;	/* Magic number (BTF_MAGIC).  */
  uint8_t  version;	/* Data format version (BTF_VERSION).  */
  uint8_t  flags;	/* Flags. Currently unused.  */
  uint32_t hdr_len;	/* Length of this header (sizeof (struct btf_header)).  */

  /* Following offsets are relative to the end of this header.  */
  uint32_t type_off;	/* Offset of type section, in bytes.  */
  uint32_t type_len;	/* Length of type section, in bytes.  */
  uint32_t str_off;	/* Offset of string section, in bytes.  */
  uint32_t str_len;	/* Length of string section, in bytes.  */
};

/* Maximum type identifier.  */
#define BTF_MAX_TYPE	0x000fffff
/* Maximum offset into the string section.  */
#define BTF_MAX_NAME_OFFSET	0x00ffffff
/* Maximum number of struct, union, enum members or func args.  */
#define BTF_MAX_VLEN	0xffff

/* Type ID 0 represents the void type.  */
#define BTF_VOID_TYPEID 0
/* Initial type ID for regular types.  */
#define BTF_INIT_TYPEID 1

struct btf_type
{
  uint32_t name_off; 	/* Offset in string section of type name.  */
  uint32_t info;	/* Encoded kind, variant length, kind flag:
			   - bits  0-15: vlen
			   - bits 16-23: unused
			   - bits 24-28: kind
			   - bits 29-30: unused
			   - bit     31: kind_flag
			   See accessor macros below.  */

  /* SIZE is used by INT, ENUM, STRUCT, UNION, DATASEC kinds.
     TYPE is used by PTR, TYPEDEF, VOLATILE, CONST, RESTRICT, FUNC,
     FUNC_PROTO and VAR kinds.  */
  union
  {
    uint32_t size;	/* Size of the entire type, in bytes.  */
    uint32_t type;	/* A type_id referring to another type.  */
  };
};

/* The following macros access the information encoded in btf_type.info.  */
/* Type kind. See below.  */
#define BTF_INFO_KIND(info)	(((info) >> 24) & 0x1f)
/* Number of entries of variable length data following certain type kinds.
   For example, number of structure members, number of function parameters.  */
#define BTF_INFO_VLEN(info)	((info) & 0xffff)
/* For BTF_KIND_FWD, 1 if forward to union, 0 if forward to struct.
   For BTF_KIND_STRUCT and BTF_KIND_UNION, 1 if the struct/union contains
   a bitfield.  */
#define BTF_INFO_KFLAG(info)	((info) >> 31)

/* Encoding for struct btf_type.info.  */
#define BTF_TYPE_INFO(kind, kflag, vlen) \
  ((((kflag) ? 1 : 0 ) << 31) | ((kind & 0x1f) << 24) | ((vlen) & 0xffff))

#define BTF_KIND_UNKN		0	/* Unknown or invalid.  */
#define BTF_KIND_INT		1	/* Integer.  */
#define BTF_KIND_PTR		2	/* Pointer.  */
#define BTF_KIND_ARRAY		3	/* Array.  */
#define BTF_KIND_STRUCT		4	/* Struct.  */
#define BTF_KIND_UNION		5	/* Union.  */
#define BTF_KIND_ENUM		6	/* Enumeration.  */
#define BTF_KIND_FWD		7	/* Forward.  */
#define BTF_KIND_TYPEDEF	8	/* Typedef.  */
#define BTF_KIND_VOLATILE	9	/* Referenced type is volatile.  */
#define BTF_KIND_CONST		10	/* Referenced type is const.  */
#define BTF_KIND_RESTRICT	11	/* Restrict.  */
#define BTF_KIND_FUNC		12	/* Subprogram.  */
#define BTF_KIND_FUNC_PROTO	13	/* Function Prototype.  */
#define BTF_KIND_VAR		14	/* Variable.  */
#define BTF_KIND_DATASEC	15	/* Section such as .bss or .data.  */
#define BTF_KIND_FLOAT		16	/* Floating point.  */
#define BTF_KIND_DECL_TAG	17	/* Declaration tag.  */
#define BTF_KIND_TYPE_TAG	18	/* Type tag.  */
#define BTF_KIND_ENUM64 	19	/* Enumeration up to 64 bits.  */
#define BTF_KIND_MAX		BTF_KIND_ENUM64
#define NR_BTF_KINDS		(BTF_KIND_MAX + 1)

/* For some BTF_KINDs, struct btf_type is immediately followed by
   additional data describing the type.  */

/* BTF_KIND_INT is followed by a 32-bit word, with the following
   bit arrangement.  */
#define BTF_INT_ENCODING(VAL)	(((VAL) & 0x0f000000) >> 24)
#define BTF_INT_OFFSET(VAL)	(((VAL) & 0x00ff0000) >> 16)
#define BTF_INT_BITS(VAL)	((VAL)  & 0x000000ff)

#define BTF_INT_DATA(encoding, offset, bits) \
  ((((encoding) & 0x0f) << 24) | (((offset) & 0xff) << 16) | ((bits) & 0xff))

/* BTF_INT_ENCODING holds the following attribute flags.  */
#define BTF_INT_SIGNED 	(1 << 0)
#define BTF_INT_CHAR 	(1 << 1)
#define BTF_INT_BOOL	(1 << 2)

/* BTF_KIND_ENUM is followed by VLEN struct btf_enum entries,
   which describe the enumerators. */
struct btf_enum
{
  uint32_t name_off;	/* Offset in string section of enumerator name.  */
  int32_t  val;		/* Enumerator value.  */
};

/* BTF_KF_ENUM_ holds the flags for kflags in BTF_KIND_ENUM{,64}.  */
#define BTF_KF_ENUM_UNSIGNED	(0)
#define BTF_KF_ENUM_SIGNED 	(1 << 0)

/* BTF_KIND_ARRAY is followed by a single struct btf_array.  */
struct btf_array
{
  uint32_t type;	/* Type of array elements.  */
  uint32_t index_type;	/* Type of array index.  */
  uint32_t nelems;	/* Number of elements. 0 for unsized/variable length.  */
};

/* BTF_KIND_STRUCT and BTF_KIND_UNION are followed by VLEN
   struct btf_member.  */
struct btf_member
{
  uint32_t name_off;	/* Offset in string section of member name.  */
  uint32_t type;	/* Type of member.  */
  uint32_t offset;	/* If the type info kind_flag is set, this contains
			   both the member bitfield size and bit offset,
			   according to the macros below. If kind_flag is not
			   set, offset contains only the bit offset (from the
			   beginning of the struct).  */
};

/* If struct or union type info kind_flag is set, used to access member
   bitfield size from btf_member.offset.  */
#define BTF_MEMBER_BITFIELD_SIZE(VAL) 	((VAL) >> 24)
/* If struct or union type info kind_flag is set, used to access member
   bit offset from btf_member.offset.  */
#define BTF_MEMBER_BITFIELD_OFFSET(VAL)	((VAL) & 0x00ffffff)

/* BTF_KIND_FUNC_PROTO is followed by VLEN struct btf_param entries, which
   describe the types of the function parameters.  */
struct btf_param
{
  uint32_t name_off;	/* Offset in string section of parameter name.  */
  uint32_t type;	/* Type of parameter.  */
};

/* BTF_KIND_FUNC records encode linkage information in the VLEN bits
   of the type record.  These are the supported values.  */
enum btf_func_linkage
{
  BTF_FUNC_STATIC = 0,
  BTF_FUNC_GLOBAL = 1,
  BTF_FUNC_EXTERN = 2,
};

/* BTF_KIND_VAR records encode linkage information in a single
   trailing struct btf_var.  These are the supported values.  */
enum btf_var_linkage
{
  BTF_VAR_STATIC = 0,
  BTF_VAR_GLOBAL_ALLOCATED = 1,
  BTF_VAR_GLOBAL_EXTERN = 2,
};

/* BTF_KIND_VAR is followed by a single struct btf_var, which describes
   information about the variable.  */
struct btf_var
{
  uint32_t linkage;	/* 0=static, 1=global, 2=extern.  */
};

/* BTF_KIND_DATASEC is followed by VLEN struct btf_var_secinfo entries,
   which describe all BTF_KIND_VAR or extern BTF_KIND_FUNC types contained
   in the section.  */
struct btf_var_secinfo
{
  uint32_t type;	/* Type of BTF_KIND_VAR or BTF_KIND_FUNC item.  */
  uint32_t offset;	/* In-section offset (in bytes) of item.  */
  uint32_t size;	/* Size (in bytes) of item.  */
};

/* BTF_KIND_ENUM64 is followed by VLEN struct btf_enum64 entries,
   which describe the 64 bits enumerators.  */
struct btf_enum64
{
  uint32_t name_off;	/* Offset in string section of enumerator name.  */
  uint32_t val_lo32;	/* lower 32-bit value for a 64-bit value Enumerator */
  uint32_t val_hi32;	/* high 32-bit value for a 64-bit value Enumerator */
};

/* BTF_KIND_DECL_TAG is followed by a single struct btf_decl_tag, which
   describes the item to which the tag applies:
   - If component_idx == (uint32_t) -1, then the tag applies to item referred
     to by the type_id.
   - Otherwise, the tag applies to the struct or union member, or function
     argument of the type referred to by type_id with the 0-based index
     given by component_idx.  */
struct btf_decl_tag
{
  uint32_t component_idx;
};

#define BTF_ENUM64_VALUE(ENUM64) \
  ((int64_t) ((((uint64_t) (ENUM64)->val_hi32) << 32) | (ENUM64)->val_lo32))

/* Type for callback to be passed to btf_map below.

   ABFD is the bfd containing the BTF section.
   BUF is a pointer to the BTF section content.
   STRTAB_OFFSET is the offset of the BTF string table in BUF.
   TYPE_OFFSET is the offset of T in BUF.
   TYPE_ID is the ID of the type being handled.
   T is the type to be handled.  */

typedef void (*btf_type_map_cb) (bfd *abfd, bfd_byte *buf,
				 bfd_vma strtab_offset, bfd_vma type_offset,
				 uint32_t type_id, struct btf_type *entry);

/* Iterate over the entries in a BTF section.

   ABFD is the bfd containing the BTF section.
   SEC is the section containing BTF data.

   This function returns `false' if the section doesn't contain valid
   BTF data or if there is an out of memory condition.  */

bool btf_map (bfd *abfd, asection *sec, btf_type_map_cb cb);

/* Get the total size in bytes of a given BTF entry, including its
   payload.  */

bfd_vma btf_type_size (struct btf_type *entry);

/* Decoding functions.  The functions returning pointers to structures
   allocate memory, and it is up to the user to free that memory when
   no longer used. */

struct btf_header *btf_read_header (bfd *abfd, bfd_byte *buf);
struct btf_type *btf_read_type (bfd *abfd, bfd_byte *buf);
struct btf_member *btf_read_member (bfd *abfd, bfd_byte *buf);
struct btf_decl_tag *btf_read_decl_tag (bfd *abfd, bfd_byte *buf);
struct btf_var_secinfo *btf_read_var_secinfo (bfd *abfd, bfd_byte *buf);
struct btf_var *btf_read_var (bfd *abfd, bfd_byte *buf);
struct btf_param *btf_read_param (bfd *abfd, bfd_byte *buf);
struct btf_enum64 *btf_read_enum64 (bfd *abfd, bfd_byte *buf);
struct btf_enum *btf_read_enum (bfd *abdf, bfd_byte *buf);
struct btf_array *btf_read_array (bfd *abfd, bfd_byte *buf);
uint32_t btf_read_integral (bfd *abfd, bfd_byte *buf);

#endif /* ! BTF_H */
