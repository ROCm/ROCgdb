/* readelf-nn.c -- ELF32 / ELF64 helper of readelf.c
   Copyright (C) 1998-2026 Free Software Foundation, Inc.

   Originally developed by Eric Youngdale <eric@andante.jic.com>
   Modifications by Nick Clifton <nickc@redhat.com>

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

/* The difference between readelf and objdump:

  Both programs are capable of displaying the contents of ELF format files,
  so why does the binutils project have two file dumpers ?

  The reason is that objdump sees an ELF file through a BFD filter of the
  world; if BFD has a bug where, say, it disagrees about a machine constant
  in e_flags, then the odds are good that it will remain internally
  consistent.  The linker sees it the BFD way, objdump sees it the BFD way,
  GAS sees it the BFD way.  There was need for a tool to go find out what
  the file actually says.

  This is why the readelf program does not link against the BFD library - it
  exists as an independent program to help verify the correct working of BFD.

  There is also the case that readelf can provide more information about an
  ELF file than is provided by objdump.  In particular it can display DWARF
  debugging information which (at the moment) objdump cannot.  */

/* Read in the program headers from FILEDATA and store them in PHEADERS.
   Returns TRUE upon success, FALSE otherwise.  */

static bool
ElfXX(_get_program_headers) (Filedata * filedata, Elf_Internal_Phdr * pheaders)
{
  ElfXX(_External_Phdr) * phdrs;
  const ElfXX(_External_Phdr) * external;
  Elf_Internal_Phdr *   internal;
  unsigned int i;
  unsigned int size = filedata->file_header.e_phentsize;
  unsigned int num  = filedata->file_header.e_phnum;

  /* PR binutils/17531: Cope with unexpected section header sizes.  */
  if (size == 0 || num == 0)
    return false;
  if (size < sizeof * phdrs)
    {
      error (_("The e_phentsize field in the ELF header is less than the size of an ELF program header\n"));
      return false;
    }
  if (size > sizeof * phdrs)
    warn (_("The e_phentsize field in the ELF header is larger than the size of an ELF program header\n"));

  phdrs = get_data (NULL, filedata, filedata->file_header.e_phoff, size, num,
		    _("program headers"));
  if (phdrs == NULL)
    return false;

  for (i = 0, internal = pheaders, external = phdrs;
       i < filedata->file_header.e_phnum;
       i++, internal++, external++)
    {
      internal->p_type   = BYTE_GET (external->p_type);
      internal->p_offset = BYTE_GET (external->p_offset);
      internal->p_vaddr  = BYTE_GET (external->p_vaddr);
      internal->p_paddr  = BYTE_GET (external->p_paddr);
      internal->p_filesz = BYTE_GET (external->p_filesz);
      internal->p_memsz  = BYTE_GET (external->p_memsz);
      internal->p_flags  = BYTE_GET (external->p_flags);
      internal->p_align  = BYTE_GET (external->p_align);
    }

  free (phdrs);
  return true;
}

/* Allocate memory and load the sections headers into FILEDATA->filedata->section_headers.
   If PROBE is true, this is just a probe and we do not generate any error
   messages if the load fails.  */

static bool
ElfXX(_get_section_headers) (Filedata * filedata, bool probe)
{
  ElfXX(_External_Shdr) * shdrs;
  Elf_Internal_Shdr *   internal;
  Elf_Internal_Shdr **  orig_internal;
  unsigned int          i;
  unsigned int          size = filedata->file_header.e_shentsize;
  unsigned int          num = probe ? 1 : filedata->file_header.e_shnum;

  /* PR binutils/17531: Cope with unexpected section header sizes.  */
  if (size == 0 || num == 0)
    return false;

  /* The section header cannot be at the start of the file - that is
     where the ELF file header is located.  A file with absolutely no
     sections in it will use a shoff of 0.  */
  if (filedata->file_header.e_shoff == 0)
    return false;

  if (size < sizeof * shdrs)
    {
      if (! probe)
	error (_("The e_shentsize field in the ELF header is less than the size of an ELF section header\n"));
      return false;
    }
  if (!probe && size > sizeof * shdrs)
    warn (_("The e_shentsize field in the ELF header is larger than the size of an ELF section header\n"));

  shdrs = get_data (NULL, filedata, filedata->file_header.e_shoff, size, num,
		    probe ? NULL : _("section headers"));
  if (shdrs == NULL)
    return false;

  filedata->section_headers = (Elf_Internal_Shdr *)
    cmalloc (num, sizeof (Elf_Internal_Shdr));
  if (filedata->section_headers == NULL)
    {
      if (!probe)
	error (_("Out of memory reading %u section headers\n"), num);
      free (shdrs);
      return false;
    }

  if (!probe)
    filedata->orig_section_headers = xcalloc2 (num,
					       sizeof (Elf_Internal_Shdr *));

  orig_internal = filedata->orig_section_headers;
  for (i = 0, internal = filedata->section_headers;
       i < num;
       i++, internal++, orig_internal++)
    {
      internal->sh_name      = BYTE_GET (shdrs[i].sh_name);
      internal->sh_type      = BYTE_GET (shdrs[i].sh_type);
      internal->sh_flags     = BYTE_GET (shdrs[i].sh_flags);
      internal->sh_addr      = BYTE_GET (shdrs[i].sh_addr);
      internal->sh_offset    = BYTE_GET (shdrs[i].sh_offset);
      internal->sh_size      = BYTE_GET (shdrs[i].sh_size);
      internal->sh_link      = BYTE_GET (shdrs[i].sh_link);
      internal->sh_info      = BYTE_GET (shdrs[i].sh_info);
      internal->sh_addralign = BYTE_GET (shdrs[i].sh_addralign);
      internal->sh_entsize   = BYTE_GET (shdrs[i].sh_entsize);
      if (!probe)
	validate_section_info (internal, orig_internal, i, filedata);
    }

  free (shdrs);
  return true;
}

#undef ElfXX
