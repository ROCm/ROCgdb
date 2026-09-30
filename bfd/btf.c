/* btf.c -- display BTF contents of a BFD binary file
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
   Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston, MA
   02110-1301, USA.  */

#include "sysdep.h"
#include "bfd.h"
#include "libbfd.h"
#include "btf.h"

struct btf_header *
btf_read_header (bfd *abfd, bfd_byte *buf)
{
  struct btf_header *header = bfd_zmalloc (sizeof (struct btf_header));

  header->magic = bfd_get_16 (abfd, buf);
  buf += 2;
  header->version = bfd_get_8 (abfd, buf);
  buf += 1;
  header->flags = bfd_get_8 (abfd, buf);
  buf += 1;
  header->hdr_len = bfd_get_32 (abfd, buf);
  buf += 4;
  header->type_off = bfd_get_32 (abfd, buf);
  buf += 4;
  header->type_len = bfd_get_32 (abfd, buf);
  buf += 4;
  header->str_off = bfd_get_32 (abfd, buf);
  buf += 4;
  header->str_len = bfd_get_32 (abfd, buf);

  return header;
}

struct btf_type *
btf_read_type (bfd *abfd, bfd_byte *buf)
{
  struct btf_type *typ = bfd_zmalloc (sizeof (struct btf_type));

  typ->name_off = bfd_get_32 (abfd, buf);
  buf += 4;
  typ->info = bfd_get_32 (abfd, buf);
  buf += 4;
  typ->size = bfd_get_32 (abfd, buf);

  return typ;
}

struct btf_member *
btf_read_member (bfd *abfd, bfd_byte *buf)
{
  struct btf_member *member = bfd_zmalloc (sizeof (struct btf_member));

  member->name_off = bfd_get_32 (abfd, buf);
  buf += 4;
  member->type = bfd_get_32 (abfd, buf);
  buf += 4;
  member->offset = bfd_get_32 (abfd, buf);

  return member;
}

struct btf_decl_tag *
btf_read_decl_tag (bfd *abfd, bfd_byte *buf)
{
  struct btf_decl_tag *decl_tag = bfd_zmalloc (sizeof (struct btf_decl_tag));

  decl_tag->component_idx = bfd_get_32 (abfd, buf);
  return decl_tag;
}

struct btf_var_secinfo *
btf_read_var_secinfo (bfd *abfd, bfd_byte *buf)
{
  struct btf_var_secinfo *var_secinfo = bfd_zmalloc (sizeof (struct btf_var_secinfo));

  var_secinfo->type = bfd_get_32 (abfd, buf);
  buf += 4;
  var_secinfo->offset = bfd_get_32 (abfd, buf);
  buf += 4;
  var_secinfo->size = bfd_get_32 (abfd, buf);

  return var_secinfo;
}

struct btf_var *
btf_read_var (bfd *abfd, bfd_byte *buf)
{
  struct btf_var *var = bfd_zmalloc (sizeof (struct btf_var));

  var->linkage = bfd_get_32 (abfd, buf);
  return var;
}

struct btf_param *
btf_read_param (bfd *abfd, bfd_byte *buf)
{
  struct btf_param *param = bfd_zmalloc (sizeof (struct btf_param));

  param->name_off = bfd_get_32 (abfd, buf);
  buf += 4;
  param->type = bfd_get_32 (abfd, buf);

  return param;
}

struct btf_enum64 *
btf_read_enum64 (bfd *abfd, bfd_byte *buf)
{
  struct btf_enum64 *enum64 = bfd_zmalloc (sizeof (struct btf_enum64));

  enum64->name_off = bfd_get_32 (abfd, buf);
  buf += 4;
  enum64->val_lo32 = bfd_get_32 (abfd, buf);
  buf += 4;
  enum64->val_hi32 = bfd_get_32 (abfd, buf);

  return enum64;
}

struct btf_enum *
btf_read_enum (bfd *abfd, bfd_byte *buf)
{
  struct btf_enum *anenum = bfd_zmalloc (sizeof (struct btf_enum));

  anenum->name_off = bfd_get_32 (abfd, buf);
  buf += 4;
  anenum->val = bfd_get_32 (abfd, buf);

  return anenum;
}

struct btf_array *
btf_read_array (bfd *abfd, bfd_byte *buf)
{
  struct btf_array *array = bfd_zmalloc (sizeof (struct btf_array));

  array->type = bfd_get_32 (abfd, buf);
  buf += 4;
  array->index_type = bfd_get_32 (abfd, buf);
  buf += 4;
  array->nelems = bfd_get_32 (abfd, buf);

  return array;
}

uint32_t
btf_read_integral (bfd *abfd, bfd_byte *buf)
{
  return bfd_get_32 (abfd, buf);
}

bfd_vma
btf_type_size (struct btf_type *entry)
{
  bfd_vma size = sizeof (struct btf_type);
  uint16_t vlen = BTF_INFO_VLEN (entry->info);

  switch (BTF_INFO_KIND (entry->info))
    {
    case BTF_KIND_INT:
      size += 4;
      break;
    case BTF_KIND_ARRAY:
      size += sizeof (struct btf_array);
      break;
    case BTF_KIND_ENUM:
      size += sizeof (struct btf_enum) * vlen;
      break;
    case BTF_KIND_ENUM64:
      size += sizeof (struct btf_enum64) * vlen;
      break;
    case BTF_KIND_FUNC_PROTO:
      size += sizeof (struct btf_param) * vlen;
      break;
    case BTF_KIND_VAR:
      size += sizeof (struct btf_var);
      break;
    case BTF_KIND_UNION:
      /* Fallthrough.  */
    case BTF_KIND_STRUCT:
      size += sizeof (struct btf_member) * vlen;
      break;
    case BTF_KIND_DATASEC:
      size += sizeof (struct btf_var_secinfo) * vlen;
      break;
    case BTF_KIND_DECL_TAG:
      size += sizeof (struct btf_decl_tag);
      break;
    default:
      /* No entry specific additional data follows this entry.  */
      break;
    }

  return size;
}

bool
btf_map (bfd *abfd, asection *sec, btf_type_map_cb cb)
{
  bfd_byte *btfdata = NULL;
  struct btf_header *header = NULL;
  bfd_size_type section_size = bfd_section_size (sec);

  if (!bfd_malloc_and_get_section (abfd, sec, &btfdata))
    goto error;

  header = btf_read_header (abfd, btfdata);
  if (header->magic != 0xeb9f)
    goto error;

  if (header->version != 1)
    goto error;

  /* Make sure the BTF regions specified by the header (entries and
     string table) fall within the section.  A corrupted header may
     lead to a buffer overflow below.  */

  if (header->hdr_len + header->str_off + header->str_len > section_size
      || header->hdr_len + header->type_off + header->type_len > section_size)
    goto error;

  /* Iterate over all the entries in the section and invoke the user
     provided callback.  */
  {
    bfd_byte *str_base = btfdata + header->hdr_len + header->str_off;
    bfd_byte *type_base = btfdata + header->hdr_len + header->type_off;
    bfd_byte *type_off = type_base;
    uint32_t type_id = 1;

    while (type_off < type_base + header->type_len)
      {
	struct btf_type *t = btf_read_type (abfd, type_off);

	(*cb) (abfd, btfdata,
	       str_base - btfdata,
	       type_off - btfdata,
	       type_id++, t);
	type_off += btf_type_size (t);
	free (t);
      }
  }

  free (btfdata);
  return true;

 error:
  free (btfdata);
  return false;
}
