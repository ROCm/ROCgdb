/* Shared helpers for SEH .pdata/.xdata COFF object file format for
   multiple architectures.
   Copyright (C) 2009-2026 Free Software Foundation, Inc.

   This file is part of GAS.

   GAS is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GAS is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GAS; see the file COPYING.  If not, write to the Free
   Software Foundation, 51 Franklin Street - Fifth Floor, Boston, MA
   02110-1301, USA.  */

#include "obj-coff-seh.h"

/* Private segment collection list.  */
struct seh_seg_list {
  segT seg;
  subsegT subseg;
  char *seg_name;
};

static struct seh_context *seh_ctx_cur = NULL;

static htab_t seh_hash;

static struct seh_seg_list *x_segcur = NULL;
static struct seh_seg_list *p_segcur = NULL;

/* Build based on segment the derived .pdata/.xdata
   segment name containing origin segment's postfix name part.  */
static char *
get_pxdata_name (segT seg, const char *base_name)
{
  const char *name,*dollar, *dot;
  char *sname;

  name = bfd_section_name (seg);

  dollar = strchr (name, '$');
  dot = strchr (name + 1, '.');

  if (!dollar && !dot)
    name = "";
  else if (!dollar)
    name = dot;
  else if (!dot)
    name = dollar;
  else if (dot < dollar)
    name = dot;
  else
    name = dollar;

  sname = notes_concat (base_name, name, (const char *) NULL);

  return sname;
}

/* Allocate a seh_seg_list structure.  */
static struct seh_seg_list *
alloc_pxdata_item (segT seg, subsegT subseg, char *name)
{
  struct seh_seg_list *r;

  r = notes_alloc (sizeof (struct seh_seg_list) + strlen (name));
  r->seg = seg;
  r->subseg = subseg;
  r->seg_name = name;
  return r;
}

/* Generate pdata/xdata segment with same linkonce properties
   of based segment.  */
static segT
make_pxdata_seg (segT cseg, char *name)
{
  segT save_seg = now_seg;
  subsegT save_subseg = now_subseg;
  segT r;
  flagword flags;

  r = subseg_new (name, 0);
  /* Check if code segment is marked as linked once.  */
  flags = (bfd_section_flags (cseg)
	   & (SEC_LINK_ONCE | SEC_LINK_DUPLICATES_DISCARD
	      | SEC_LINK_DUPLICATES_ONE_ONLY | SEC_LINK_DUPLICATES_SAME_SIZE
	      | SEC_LINK_DUPLICATES_SAME_CONTENTS));

  /* Add standard section flags.  */
  flags |= SEC_ALLOC | SEC_LOAD | SEC_READONLY | SEC_DATA;

  /* Apply possibly linked once flags to new generated segment, too.  */
  if (!bfd_set_section_flags (r, flags))
    as_bad (_("bfd_set_section_flags: %s"),
	    bfd_errmsg (bfd_get_error ()));

  /* Restore to previous segment.  */
  subseg_set (save_seg, save_subseg);
  return r;
}

static void
seh_hash_insert (const char *name, struct seh_seg_list *item)
{
  str_hash_insert (seh_hash, name, item, 1);
}

static struct seh_seg_list *
seh_hash_find (char *name)
{
  return str_hash_find (seh_hash, name);
}

static struct seh_seg_list *
seh_hash_find_or_make (segT cseg, const char *base_name)
{
  struct seh_seg_list *item;
  char *name;

  /* Initialize seh_hash once.  */
  if (!seh_hash)
    seh_hash = str_htab_create ();

  name = get_pxdata_name (cseg, base_name);

  item = seh_hash_find (name);
  if (!item)
    {
      item = alloc_pxdata_item (make_pxdata_seg (cseg, name), 0, name);

      seh_hash_insert (item->seg_name, item);
    }
  else
    notes_free (name);

  return item;
}

/* Check if current segment has same name.  */
static int
seh_validate_seg (const char *directive)
{
  const char *cseg_name, *nseg_name;
  if (seh_ctx_cur->code_seg == now_seg)
    return 1;
  cseg_name = bfd_section_name (seh_ctx_cur->code_seg);
  nseg_name = bfd_section_name (now_seg);
  as_bad (_("%s used in segment '%s' instead of expected '%s'"),
	  directive, nseg_name, cseg_name);
  ignore_rest_of_line ();
  return 0;
}

/* Switch back to the code section, whatever that may be.  */
static void
switch_xdata (subsegT subseg, segT code_seg)
{
  x_segcur = seh_hash_find_or_make (code_seg, ".xdata");

  subseg_set (x_segcur->seg, subseg);
}

static void
switch_pdata (segT code_seg)
{
  p_segcur = seh_hash_find_or_make (code_seg, ".pdata");

  subseg_set (p_segcur->seg, p_segcur->subseg);
}

/* Verify that we're in the context of a seh_proc.  */

static int
verify_context (const char *directive)
{
  if (seh_ctx_cur == NULL)
    {
      as_bad (_("%s used outside of .seh_proc block"), directive);
      ignore_rest_of_line ();
      return 0;
    }
  return 1;
}

/* Skip whitespace and a comma.  Error if the comma is not seen.  */

static int
skip_whitespace_and_comma (int required)
{
  SKIP_WHITESPACE ();
  if (*input_line_pointer == ',')
    {
      input_line_pointer++;
      SKIP_WHITESPACE ();
      return 1;
    }
  else if (required)
    {
      as_bad (_("missing separator"));
      ignore_rest_of_line ();
    }
  else
    demand_empty_rest_of_line ();
  return 0;
}
