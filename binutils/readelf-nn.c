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

static Elf_Internal_Sym *
ElfXX(_get_symbols) (Filedata *filedata, const Elf_Internal_Shdr *section,
		     uint64_t *num_syms_return)
{
  uint64_t number = 0;
  ElfXX(_External_Sym) * esyms = NULL;
  Elf_External_Sym_Shndx * shndx = NULL;
  Elf_Internal_Sym * isyms = NULL;
  Elf_Internal_Sym * psym;
  unsigned int j;
  elf_section_list * entry;

  if (section->sh_size == 0)
    {
      if (num_syms_return != NULL)
	* num_syms_return = 0;
      return NULL;
    }

  /* Run some sanity checks first.  */
  if (section->sh_entsize == 0 || section->sh_entsize > section->sh_size)
    {
      error (_("Section %s has an invalid sh_entsize of %#" PRIx64 "\n"),
	     printable_section_name (filedata, section),
	     section->sh_entsize);
      goto exit_point;
    }

  if (section->sh_size > filedata->file_size)
    {
      error (_("Section %s has an invalid sh_size of %#" PRIx64 "\n"),
	     printable_section_name (filedata, section),
	     section->sh_size);
      goto exit_point;
    }

  number = section->sh_size / section->sh_entsize;

  if (number * sizeof (*esyms) > section->sh_size + 1)
    {
      error (_("Size (%#" PRIx64 ") of section %s "
	       "is not a multiple of its sh_entsize (%#" PRIx64 ")\n"),
	     section->sh_size,
	     printable_section_name (filedata, section),
	     section->sh_entsize);
      goto exit_point;
    }

  esyms = get_data (NULL, filedata, section->sh_offset, 1, section->sh_size,
		    _("symbols"));
  if (esyms == NULL)
    goto exit_point;

  shndx = NULL;
  for (entry = filedata->symtab_shndx_list; entry != NULL; entry = entry->next)
    {
      if (entry->hdr->sh_link != (size_t) (section - filedata->section_headers))
	continue;

      if (shndx != NULL)
	{
	  error (_("Multiple symbol table index sections associated with the same symbol section\n"));
	  free (shndx);
	}

      shndx = (Elf_External_Sym_Shndx *) get_data (NULL, filedata,
						   entry->hdr->sh_offset,
						   1, entry->hdr->sh_size,
						   _("symbol table section indices"));
      if (shndx == NULL)
	goto exit_point;

      /* PR17531: file: heap-buffer-overflow */
      if (entry->hdr->sh_size / sizeof (Elf_External_Sym_Shndx) < number)
	{
	  error (_("Index section %s has an sh_size of %#" PRIx64 " - expected %#" PRIx64 "\n"),
		 printable_section_name (filedata, entry->hdr),
		 entry->hdr->sh_size,
		 section->sh_size);
	  goto exit_point;
	}
    }

  isyms = (Elf_Internal_Sym *) cmalloc (number, sizeof (Elf_Internal_Sym));

  if (isyms == NULL)
    {
      error (_("Out of memory reading %" PRIu64 " symbols\n"), number);
      goto exit_point;
    }

  for (j = 0, psym = isyms; j < number; j++, psym++)
    {
      psym->st_name  = BYTE_GET (esyms[j].st_name);
      psym->st_value = BYTE_GET (esyms[j].st_value);
      psym->st_size  = BYTE_GET (esyms[j].st_size);
      psym->st_shndx = BYTE_GET (esyms[j].st_shndx);

      if (psym->st_shndx == (SHN_XINDEX & 0xffff) && shndx != NULL)
	psym->st_shndx
	  = byte_get ((unsigned char *) &shndx[j], sizeof (shndx[j]));
      else if (psym->st_shndx >= (SHN_LORESERVE & 0xffff))
	psym->st_shndx += SHN_LORESERVE - (SHN_LORESERVE & 0xffff);

      psym->st_info  = BYTE_GET (esyms[j].st_info);
      psym->st_other = BYTE_GET (esyms[j].st_other);
    }

 exit_point:
  free (shndx);
  free (esyms);

  if (num_syms_return != NULL)
    * num_syms_return = isyms == NULL ? 0 : number;

  return isyms;
}

static bool
ElfXX(_get_dynamic_section) (Filedata * filedata)
{
  ElfXX(_External_Dyn) * edyn, * ext;
  Elf_Internal_Dyn * entry;

  edyn = get_data (NULL, filedata, filedata->dynamic_addr, 1,
		   filedata->dynamic_size, _("dynamic section"));
  if (!edyn)
    return false;

  /* SGI's ELF has more than one section in the DYNAMIC segment, and we
     might not have the luxury of section headers.  Look for the DT_NULL
     terminator to determine the number of entries.  */
  for (ext = edyn, filedata->dynamic_nent = 0;
       /* PR 17533 file: 033-67080-0.004 - do not read past end of buffer.  */
       (char *) (ext + 1) <= (char *) edyn + filedata->dynamic_size;
       ext++)
    {
      filedata->dynamic_nent++;
      if (BYTE_GET (ext->d_tag) == DT_NULL)
	break;
    }

  filedata->dynamic_section
    = (Elf_Internal_Dyn *) cmalloc (filedata->dynamic_nent, sizeof (* entry));
  if (filedata->dynamic_section == NULL)
    {
      error (_("Out of memory allocating space for %" PRIu64 " dynamic entries\n"),
	     filedata->dynamic_nent);
      free (edyn);
      return false;
    }

  /* Convert from external to internal formats.  */
  for (ext = edyn, entry = filedata->dynamic_section;
       entry < filedata->dynamic_section + filedata->dynamic_nent;
       ext++, entry++)
    {
      entry->d_tag      = BYTE_GET (ext->d_tag);
      entry->d_un.d_val = BYTE_GET (ext->d_un.d_val);
    }

  free (edyn);

  return true;
}

#undef ElfXX
