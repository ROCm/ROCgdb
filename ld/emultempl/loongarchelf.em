# This shell script emits a C file. -*- C -*-
#   Copyright (C) 2021-2026 Free Software Foundation, Inc.
#   Contributed by Loongson Ltd.
#
# This file is part of the GNU Binutils.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING3. If not,
# see <http://www.gnu.org/licenses/>.

fragment <<EOF

#include "ldmain.h"
#include "ldctor.h"
#include "elf/loongarch.h"
#include "elfxx-loongarch.h"

EOF

# Disable linker relaxation if set address of section or segment.
PARSE_AND_LIST_ARGS_CASES=${PARSE_AND_LIST_ARGS_CASES}'
    case OPTION_SECTION_START:
    case OPTION_TTEXT:
    case OPTION_TBSS:
    case OPTION_TDATA:
    case OPTION_TTEXT_SEGMENT:
    case OPTION_TRODATA_SEGMENT:
    case OPTION_TLDATA_SEGMENT:
      link_info.disable_target_specific_optimizations = 2;
      return false;
'

fragment <<EOF

/* Fake input file for align.  */
static lang_input_statement_type *align_file;

static void
larch_elf_before_allocation (void)
{
  gld${EMULATION_NAME}_before_allocation ();

  if (link_info.discard == discard_sec_merge)
    link_info.discard = discard_l;

  if (!bfd_link_relocatable (&link_info))
    {
      /* We always need at least some relaxation to handle code alignment.  */
      if (RELAXATION_DISABLED_BY_USER)
	TARGET_ENABLE_RELAXATION;
      else
	ENABLE_RELAXATION;
    }

  link_info.relax_pass = 3;
}

/* Traverse the linker tree to insert the align section
   before input section.  */

static bool
hook_in_align (lang_statement_list_type add,
	       asection *input_section,
	       lang_statement_union_type **lp)
{
  bool ret;
  lang_statement_union_type *l;

  for (; (l = *lp) != NULL; lp = &l->header.next)
    {
      switch (l->header.type)
	{
	case lang_constructors_statement_enum:
	  ret = hook_in_align (add, input_section, &constructor_list.head);
	  if (ret)
	    return ret;
	  break;

	case lang_output_section_statement_enum:
	  ret = hook_in_align (add, input_section,
			       &l->output_section_statement.children.head);
	  if (ret)
	    return ret;
	  break;

	case lang_wild_statement_enum:
	  ret = hook_in_align (add, input_section,
			       &l->wild_statement.children.head);
	  if (ret)
	    return ret;
	  break;

	case lang_group_statement_enum:
	  ret = hook_in_align (add, input_section,
			       &l->group_statement.children.head);
	  if (ret)
	    return ret;
	  break;

	case lang_input_section_enum:
	  if (l->input_section.section == input_section)
	    {
	      /* We've found our section.  Insert the align immediately
		 before its associated input section.  */
	      *lp = add.head;
	      add.head->header.next = l;
	      return true;
	    }
	  break;

	case lang_data_statement_enum:
	case lang_reloc_statement_enum:
	case lang_object_symbols_statement_enum:
	case lang_output_statement_enum:
	case lang_target_statement_enum:
	case lang_input_statement_enum:
	case lang_assignment_statement_enum:
	case lang_padding_statement_enum:
	case lang_address_statement_enum:
	case lang_fill_statement_enum:
	  break;

	default:
	  FAIL ();
	  break;
	}
    }

  return false;
}

/* Create a new align section, and arrange for it to be linked
   immediately before INPUT_SECTION.  */

static asection *
elf${ELFSIZE}_loongarch_add_align_section (const char *align_sec_name,
					   asection *input_section)
{
  flagword flags;
  asection *align_sec;
  asection *output_section;
  lang_statement_list_type add_child;
  lang_output_section_statement_type *os;

  flags = (SEC_ALLOC | SEC_LOAD | SEC_READONLY | SEC_CODE
	   | SEC_HAS_CONTENTS | SEC_RELOC | SEC_IN_MEMORY | SEC_KEEP);
  align_sec = bfd_make_section_anyway_with_flags (align_file->the_bfd,
						  align_sec_name, flags);
  if (align_sec == NULL)
    goto err_ret;

  align_sec->veneer = 1;
  bfd_set_section_alignment (align_sec, 2);

  output_section = input_section->output_section;
  os = lang_output_section_get (output_section);

  lang_list_init (&add_child);
  lang_add_section (&add_child, align_sec, NULL, NULL, os);

  if (add_child.head == NULL)
    goto err_ret;

  align_sec->size = (1 << input_section->alignment_power) - 4;
  align_sec->contents = bfd_alloc (align_file->the_bfd, align_sec->size);
  if (align_sec->contents == NULL && align_sec->size != 0)
    goto err_ret;
  align_sec->alloced = 1;

  if (hook_in_align (add_child, input_section, &os->children.head))
    return align_sec;

 err_ret:
  einfo (_("%X%P: can not make align section: %E\n"));
  return NULL;
}

static void
gldloongarch_layout_sections_again (void)
{
  /* If we have changed sizes of the align sections, then we need
     to recalculate all the section offsets.  */
  ldelf_map_segments (true);
}

static void
gld${EMULATION_NAME}_after_allocation (void)
{
  int need_layout = 0;

  /* Don't attempt to discard unused .eh_frame sections until the final link,
     as we can't reliably tell if they're used until after relaxation.  */
  if (!bfd_link_relocatable (&link_info))
    {
      need_layout = bfd_elf_discard_info (&link_info);
      if (need_layout < 0)
	{
	  einfo (_("%X%P: .eh_frame/.stab edit: %E\n"));
	  return;
	}
    }

  /* If generating a relocatable output file, we have to add align
     at the start of sections.  */
  if (align_file != NULL && bfd_link_relocatable (&link_info))
    {
      if (! elf${ELFSIZE}_loongarch_size_aligns (link_info.output_bfd,
			align_file->the_bfd,
			&link_info,
			&elf${ELFSIZE}_loongarch_add_align_section,
			&gldloongarch_layout_sections_again))
	{
	  einfo (_("%X%P: can not size align section: %E\n"));
	  return;
	}
    }

  /* The program header size of executable file may increase.  */
  if (bfd_get_flavour (link_info.output_bfd) == bfd_target_elf_flavour
      && !bfd_link_relocatable (&link_info))
    {
      if (lang_phdr_list == NULL)
	elf_seg_map (link_info.output_bfd) = NULL;
      if (!bfd_elf_map_sections_to_segments (link_info.output_bfd,
					     &link_info, NULL))
	fatal (_("%P: map sections to segments failed: %E\n"));
    }

  /* Adjust program header size and .eh_frame_hdr size before
     lang_relax_sections. Without it, the vma of data segment may increase.  */
  lang_do_assignments (lang_allocating_phase_enum);
  lang_reset_memory_regions ();
  lang_size_sections (NULL, true);

  enum phase_enum *phase = &(expld.dataseg.phase);
  bfd_elf${ELFSIZE}_loongarch_set_data_segment_info (&link_info, (int *) phase);
  /* gld${EMULATION_NAME}_map_segments (need_layout); */
  ldelf_map_segments (need_layout);
}

/* This is called before the input files are opened.  We create a new
   fake input file to hold the align sections.  */

static void
loongarch_elf_create_output_section_statements (void)
{
  if (! bfd_link_relocatable (&link_info))
    return;

  align_file = lang_add_input_file ("linker aligns",
				    lang_input_file_is_fake_enum,
				    NULL);
  align_file->the_bfd = bfd_create ("linker aligns",
				    link_info.output_bfd);
  if (align_file->the_bfd == NULL
      || ! bfd_set_arch_mach (align_file->the_bfd,
			      bfd_get_arch (link_info.output_bfd),
			      bfd_get_mach (link_info.output_bfd)))
    {
      fatal (_("%P: can not create BFD: %E\n"));
      return;
    }

  align_file->the_bfd->flags |= BFD_LINKER_CREATED;

  Elf_Internal_Ehdr *ehdr = elf_elfheader (align_file->the_bfd);
  elf_backend_data *bed = get_elf_backend_data (link_info.output_bfd);
  ehdr->e_ident[EI_CLASS] = bed->s->elfclass;

  ldlang_add_file (align_file);
}

EOF

LDEMUL_BEFORE_ALLOCATION=larch_elf_before_allocation
LDEMUL_AFTER_ALLOCATION=gld${EMULATION_NAME}_after_allocation
LDEMUL_CREATE_OUTPUT_SECTION_STATEMENTS=loongarch_elf_create_output_section_statements
