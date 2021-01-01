/* Support for AMDHSA ELF.

   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   Copyright (C) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include "sysdep.h"
#include "bfd.h"
#include "libbfd.h"
#include "elf-bfd.h"
#include "elf/amdgcn.h"

#include <string.h>


static unsigned int
bfd_amdgcn_get_mach_from_notes (bfd *abfd)
{
  asection *     note_section;
  bfd_size_type  buffer_size;
  bfd_byte *     ptr, * end, * buffer = NULL;
  unsigned int   mach = 0;

  note_section = bfd_get_section_by_name (abfd, ".note");
  if (note_section == NULL)
    return 0;

  buffer_size = note_section->size;
  if (buffer_size == 0)
    return 0;

  if (!bfd_malloc_and_get_section (abfd, note_section, &buffer))
    return 0;

  ptr = buffer;
  end = &buffer[buffer_size];
  while (mach == 0 && buffer < end)
    {
      unsigned long namesz;
      unsigned long descsz;
      unsigned long type;

      if ((ptr + 12)  >= end)
        break;

      namesz = bfd_get_32 (abfd, ptr);
      descsz = bfd_get_32 (abfd, ptr + 4);
      type   = bfd_get_32 (abfd, ptr + 8);

      if (namesz == 4 && (ptr + 16) <= end
          && (!strcmp((char*) ptr + 12, "AMDGPU")
              || !strcmp((char*) ptr + 12, "AMD"))
          && type == NT_AMDGPU_HSA_ISA)
        {
          unsigned int major, minor, patch;

          major = bfd_get_32 (abfd, ptr + 20);
          minor = bfd_get_32 (abfd, ptr + 24);
          patch = bfd_get_32 (abfd, ptr + 28);

#define GFX(major, minor, patch) (((major) << 16) + ((minor) << 8) + (patch))
          switch (GFX (major, minor, patch))
            {
            case GFX (8, 0, 1): mach = bfd_mach_amdgcn_gfx801; break;
            case GFX (8, 0, 2): mach = bfd_mach_amdgcn_gfx802; break;
            case GFX (8, 0, 3): mach = bfd_mach_amdgcn_gfx803; break;
            case GFX (8, 1, 0): mach = bfd_mach_amdgcn_gfx810; break;
            case GFX (9, 0, 0): mach = bfd_mach_amdgcn_gfx900; break;
            case GFX (9, 0, 2): mach = bfd_mach_amdgcn_gfx902; break;
            case GFX (9, 0, 4): mach = bfd_mach_amdgcn_gfx904; break;
            case GFX (9, 0, 6): mach = bfd_mach_amdgcn_gfx906; break;
            case GFX (9, 0, 8): mach = bfd_mach_amdgcn_gfx908; break;
            case GFX (10, 1, 0): mach = bfd_mach_amdgcn_gfx1010; break;
            case GFX (10, 1, 1): mach = bfd_mach_amdgcn_gfx1011; break;
            case GFX (10, 1, 2): mach = bfd_mach_amdgcn_gfx1012; break;
            case GFX (10, 3, 0): mach = bfd_mach_amdgcn_gfx1030; break;
            case GFX (10, 3, 1): mach = bfd_mach_amdgcn_gfx1031; break;
            default:  mach = bfd_mach_amdgcn_unknown; break;
            }
#undef GFX
        }

      ptr += 12 + ((namesz + 3) & ~3) + ((descsz + 3) & ~3);
    }


  if (buffer != NULL)
    free (buffer);

  return mach;
}

static bfd_boolean
elf64_amdgcn_object_p (bfd *abfd)
{
  unsigned int mach;

  if (elf_elfheader (abfd)->e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA)
    return FALSE;

  if (elf_elfheader (abfd)->e_ident[EI_ABIVERSION] < 1)
    mach = bfd_amdgcn_get_mach_from_notes (abfd);
  else
    mach = elf_elfheader (abfd)->e_flags & EF_AMDGPU_MACH;

  bfd_default_set_arch_mach (abfd, bfd_arch_amdgcn, mach);
  return TRUE;
}


#define TARGET_LITTLE_SYM	amdgcn_elf64_le_vec
#define TARGET_LITTLE_NAME	"elf64-amdgcn"
#define ELF_ARCH		bfd_arch_amdgcn
#define ELF_TARGET_ID		AMDGCN_ELF_DATA
#define ELF_MACHINE_CODE	EM_AMDGPU
#define ELF_OSABI		ELFOSABI_AMDGPU_HSA
#define ELF_MAXPAGESIZE		0x10000 /* 64KB */
#define ELF_COMMONPAGESIZE	0x1000  /* 4KB */

#define bfd_elf64_bfd_reloc_type_lookup \
  bfd_default_reloc_type_lookup
#define bfd_elf64_bfd_reloc_name_lookup \
  _bfd_norelocs_bfd_reloc_name_lookup

#define elf_backend_object_p \
  elf64_amdgcn_object_p

#include "elf64-target.h"
