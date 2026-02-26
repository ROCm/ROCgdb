/* Solaris2 link support for BFD.
   Copyright (C) 2026 Free Software Foundation, Inc.

   This file is part of BFD, the Binary File Descriptor library.

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

#include "sysdep.h"
#include "bfd.h"
#include "elf-bfd.h"
#include "elf-solaris2.h"

/* Global symbols required by the Solaris 2 ABI.  */
const char *const elf_solaris2_global_syms[] = {
  "_DYNAMIC",
  "_GLOBAL_OFFSET_TABLE_",
  "_PROCEDURE_LINKAGE_TABLE_",
  "_edata",
  "_end",
  "_etext",
  NULL
};

/* Strip these symbols out of any dynamic library.  Their value in a
   shared library can never be of use in the output executable or
   shared library being produced by the linker.  They will of course
   be defined relative to the current output, but in the case of a PDE
   that happens for the first three symbols after loading the first
   shared library.  The trouble with that is that if the first shared
   library happens to be as-needed, references from crt*.o to those
   symbols will always make the library seem to be needed.  */
bool
elf_solaris2_add_symbol_hook (bfd *abfd,
			      struct bfd_link_info *info ATTRIBUTE_UNUSED,
			      Elf_Internal_Sym *isym ATTRIBUTE_UNUSED,
			      const char **name,
			      flagword *flags ATTRIBUTE_UNUSED,
			      asection **sec ATTRIBUTE_UNUSED,
			      bfd_vma *value ATTRIBUTE_UNUSED)
{
  if ((bfd_get_file_flags (abfd) & DYNAMIC) == 0 || *name == NULL)
    return true;

  if (is_solaris2_abi_global_sym (*name))
    *name = NULL;

  return true;
}
