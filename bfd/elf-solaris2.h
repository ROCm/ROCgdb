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

extern const char *const elf_solaris2_global_syms[];

#ifdef OBJ_MAYBE_ELF_SOLARIS2

extern bool elf_solaris2_add_symbol_hook
  (bfd *, struct bfd_link_info *, Elf_Internal_Sym *, const char **,
   flagword *, asection **, bfd_vma *) ATTRIBUTE_HIDDEN;

static inline bool
is_solaris2_abi_global_sym (const char *name)
{
  for (const char *const *sym = elf_solaris2_global_syms; *sym; ++sym)
    if (strcmp (name, *sym) == 0)
      return true;
  return false;
}

#else /* !OBJ_MAYBE_ELF_SOLARIS2 */

static inline bool
elf_solaris2_add_symbol_hook (bfd *abfd ATTRIBUTE_UNUSED,
			      struct bfd_link_info *info ATTRIBUTE_UNUSED,
			      Elf_Internal_Sym *isym ATTRIBUTE_UNUSED,
			      const char **name ATTRIBUTE_UNUSED,
			      flagword *flags ATTRIBUTE_UNUSED,
			      asection **sec ATTRIBUTE_UNUSED,
			      bfd_vma *value ATTRIBUTE_UNUSED)
{
  return true;
}

#endif /* OBJ_MAYBE_ELF_SOLARIS2 */
