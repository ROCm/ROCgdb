/* Support for AMDHSA ELF.

   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.

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

static bool
elf64_amdgcn_object_p (bfd *abfd)
{
  Elf_Internal_Ehdr *hdr = elf_elfheader (abfd);
  unsigned int mach;
  unsigned char osabi_version;

  /* We should not get here if the OS ABI is not HSA, since we define
     ELF_OSABI below.  */
  BFD_ASSERT (hdr->e_ident[EI_OSABI] == ELFOSABI_AMDGPU_HSA);

  /* We only support HSA code objects v3 and above.  */
  osabi_version = hdr->e_ident[EI_ABIVERSION];
  if (osabi_version < ELFABIVERSION_AMDGPU_HSA_V3)
    return false;

  /* Read the specific processor model from e_flags.  */
  mach = elf_elfheader (abfd)->e_flags & EF_AMDGPU_MACH;
  bfd_default_set_arch_mach (abfd, bfd_arch_amdgcn, mach);

  return true;
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
