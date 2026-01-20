/* Read AIX xcoff symbol tables and convert to internal format, for GDB.
   Copyright (C) 1986-2026 Free Software Foundation, Inc.
   Derived from coffread.c, dbxread.c, and a lot of hacking.
   Contributed by IBM Corporation.

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

#include "bfd.h"
#include "event-top.h"

#include "coff/internal.h"
#include "libcoff.h"
#include "coff/xcoff.h"
#include "coff/rs6000.h"
#include "xcoffread.h"

#include "symtab.h"
#include "gdbtypes.h"
#include "symfile.h"
#include "objfiles.h"
#include "complaints.h"
#include "dwarf2/sect-names.h"
#include "dwarf2/public.h"

struct xcoff_symfile_info
{
  /* Offset in data section to TOC anchor.  */
  CORE_ADDR toc_offset = 0;
};

/* Key for XCOFF-associated data.  */

static const registry<objfile>::key<xcoff_symfile_info> xcoff_objfile_data_key;

/* XCOFF names for dwarf sections.  There is no compressed sections.  */

static const dwarf2_debug_sections dwarf2_xcoff_names = {
  { ".dwinfo", NULL },
  { ".dwabrev", NULL },
  { ".dwline", NULL },
  { ".dwloc", NULL },
  { NULL, NULL }, /* debug_loclists */
  /* AIX XCOFF defines one, named DWARF section for macro debug information.
     XLC does not generate debug_macinfo for DWARF4 and below.
     The section is assigned to debug_macro for DWARF5 and above. */
  { NULL, NULL },
  { ".dwmac", NULL },
  { ".dwstr", NULL },
  { NULL, NULL }, /* debug_str_offsets */
  { NULL, NULL }, /* debug_line_str */
  { ".dwrnges", NULL },
  { NULL, NULL }, /* debug_rnglists */
  { ".dwpbtyp", NULL },
  { NULL, NULL }, /* debug_addr */
  { ".dwframe", NULL },
  { NULL, NULL }, /* eh_frame */
  { NULL, NULL }, /* gdb_index */
  { NULL, NULL }, /* debug_names */
  { NULL, NULL }, /* debug_aranges */
  23
};

/* Search all BFD sections for the section whose target_index is
   equal to N_SCNUM.

   If no match is found, return nullptr.  */

static asection *
xcoff_secnum_to_section (int n_scnum, objfile *objfile)
{
  for (asection *sec : gdb_bfd_sections (objfile->obfd.get ()))
    if (sec->target_index == n_scnum)
      return sec;

  return nullptr;
}

/* Do initialization in preparation for reading symbols from OBJFILE.

   We will only be called if this is an XCOFF or XCOFF-like file.
   BFD handles figuring out the format of the file, and code in symfile.c
   uses BFD's determination to vector to us.  */

static void
xcoff_symfile_init (objfile *objfile)
{
  /* Allocate struct to keep track of the symfile.  */
  xcoff_objfile_data_key.emplace (objfile);
}

/* Swap raw symbol at *RAW.  Put the symbol in *SYMBOL and the first auxent in
   *AUX.  Advance *RAW and *SYMNUMP over the symbol and its auxents.  */

static void
swap_sym (struct internal_syment *symbol, union internal_auxent *aux,
	  char **raw, unsigned int *symnump, struct objfile *objfile)
{
  bfd_coff_swap_sym_in (objfile->obfd.get (), *raw, symbol);

  ++*symnump;
  *raw += coff_data (objfile->obfd)->local_symesz;
  if (symbol->n_numaux > 0)
    {
      bfd_coff_swap_aux_in (objfile->obfd.get (), *raw, symbol->n_type,
			    symbol->n_sclass, 0, symbol->n_numaux, aux);

      *symnump += symbol->n_numaux;
      *raw += coff_data (objfile->obfd)->local_symesz * symbol->n_numaux;
    }
}

/* Locate the TOC offset in the XCOFF symbol table and assign
   xcoff_symfile_info::toc_offset.  */

static void
xcoff_find_toc_offset (objfile *objfile)
{
  bfd *abfd = objfile->obfd.get ();
  file_ptr symtab_offset = obj_sym_filepos (abfd);

  /* Seek to symbol table location.  */
  if (bfd_seek (abfd, symtab_offset, SEEK_SET) < 0)
    error (_("Error reading symbols from %s: %s"),
	   objfile_name (objfile), bfd_errmsg (bfd_get_error ()));

  unsigned int num_symbols = bfd_get_symcount (abfd);
  size_t size = coff_data (abfd)->local_symesz * num_symbols;
  gdb::char_vector symtbl (size);

  /* Read in symbol table.  */
  if (int ret = bfd_read (symtbl.data (), size, abfd);
      ret != size)
    error (_("reading symbol table: %s"), bfd_errmsg (bfd_get_error ()));

  char *sraw_symbol = symtbl.data ();
  CORE_ADDR toc_offset = 0;
  for (unsigned int ssymnum = 0; ssymnum < num_symbols; )
    {
      QUIT;

      internal_syment symbol;
      bfd_coff_swap_sym_in (abfd, sraw_symbol, &symbol);

      switch (symbol.n_sclass)
	{
	case C_HIDEXT:
	  {
	    /* The CSECT auxent--always the last auxent.  */
	    internal_auxent csect_aux;
	    internal_auxent main_aux[5];

	    swap_sym (&symbol, &main_aux[0], &sraw_symbol, &ssymnum, objfile);
	    if (symbol.n_numaux > 1)
	      {
		bfd_coff_swap_aux_in
		  (objfile->obfd.get (),
		   sraw_symbol - coff_data (abfd)->local_symesz,
		   symbol.n_type,
		   symbol.n_sclass,
		   symbol.n_numaux - 1,
		   symbol.n_numaux,
		   &csect_aux);
	      }
	    else
	      csect_aux = main_aux[0];

	    if ((csect_aux.x_csect.x_smtyp & 0x7) == XTY_SD
		&& csect_aux.x_csect.x_smclas == XMC_TC0)
	      {
		if (toc_offset != 0)
		  warning (_("More than one XMC_TC0 symbol found."));

		toc_offset = symbol.n_value;

		/* Make TOC offset relative to start address of section.  */
		asection *bfd_sect
		  = xcoff_secnum_to_section (symbol.n_scnum, objfile);
		if (bfd_sect != nullptr)
		  toc_offset -= bfd_section_vma (bfd_sect);
		break;
	      }
	  }
	  break;

	default:
	  complaint (_("Storage class %d not recognized during scan"),
		     symbol.n_sclass);
	  [[fallthrough]];

	case C_RSYM:
	  {
	    /* We probably could save a few instructions by assuming that
	       C_LSYM, C_PSYM, etc., never have auxents.  */
	    int naux1 = symbol.n_numaux + 1;

	    ssymnum += naux1;
	    sraw_symbol += bfd_coff_symesz (abfd) * naux1;
	  }
	  break;
	}
    }

  /* Record the toc offset value of this symbol table into objfile
     structure.  If no XMC_TC0 is found, toc_offset should be zero.
     Another place to obtain this information would be file auxiliary
     header.  */

  xcoff_objfile_data_key.get (objfile)->toc_offset = toc_offset;
}

/* Return the toc offset value for a given objfile.  */

CORE_ADDR
xcoff_get_toc_offset (objfile *objfile)
{
  if (objfile != nullptr)
    return xcoff_objfile_data_key.get (objfile)->toc_offset;

  return 0;
}

/* Read the XCOFF symbol table.  The only thing we are interested in is the TOC
   offset value.  */

static void
xcoff_symfile_read (objfile *objfile, symfile_add_flags symfile_flags)
{
  xcoff_find_toc_offset (objfile);

  /* DWARF2 sections.  */
  dwarf2_initialize_objfile (objfile, &dwarf2_xcoff_names);
}

static void
xcoff_symfile_offsets (objfile *objfile, const section_addr_info &addrs)
{
  default_symfile_offsets (objfile, addrs);

  /* Oneof the weird side-effects of default_symfile_offsets is that
     it sometimes sets some section indices to zero for sections that,
     in fact do not exist. See the body of default_symfile_offsets
     for more info on when that happens. Undo that, as this then allows
     us to test whether the associated section exists or not, and then
     access it quickly (without searching it again).  */

  if (objfile->section_offsets.empty ())
    return; /* Is that even possible?  Better safe than sorry.  */

  const char *first_section_name
    = bfd_section_name (objfile->sections_start[0].the_bfd_section);

  if (objfile->sect_index_text == 0
      && strcmp (first_section_name, ".text") != 0)
    objfile->sect_index_text = -1;

  if (objfile->sect_index_data == 0
      && strcmp (first_section_name, ".data") != 0)
    objfile->sect_index_data = -1;

  if (objfile->sect_index_bss == 0
      && strcmp (first_section_name, ".bss") != 0)
    objfile->sect_index_bss = -1;

  if (objfile->sect_index_rodata == 0
      && strcmp (first_section_name, ".rodata") != 0)
    objfile->sect_index_rodata = -1;
}

/* Register our ability to parse symbols for xcoff BFD files.  */

static const sym_fns xcoff_sym_fns =
{
  xcoff_symfile_init,		/* read initial info, setup for sym_read() */
  xcoff_symfile_read,		/* read a symbol file into symtab */
  xcoff_symfile_offsets,	/* xlate offsets ext->int form */
  default_symfile_segments,	/* Get segment information from a file.  */
  default_symfile_relocate,	/* Relocate a debug section.  */
  NULL,				/* sym_probe_fns */
};

/* Same as xcoff_get_n_import_files, but for core files.  */

static int
xcoff_get_core_n_import_files (bfd *abfd)
{
  asection *sect = bfd_get_section_by_name (abfd, ".ldinfo");
  if (sect == NULL)
    return -1;  /* Not a core file.  */

  int n_entries = 0;
  for (file_ptr offset = 0; offset < bfd_section_size (sect);)
    {
      n_entries++;

      gdb_byte buf[4];
      if (!bfd_get_section_contents (abfd, sect, buf, offset, 4))
	return -1;

      int next = bfd_get_32 (abfd, buf);
      if (next == 0)
	break;  /* This is the last entry.  */

      offset += next;
    }

  /* Return the number of entries, excluding the first one, which is
     the path to the executable that produced this core file.  */
  return n_entries - 1;
}

/* Return the number of import files (shared libraries) that the given
   BFD depends on.  Return -1 if this number could not be computed.  */

int
xcoff_get_n_import_files (bfd *abfd)
{
  asection *sect = bfd_get_section_by_name (abfd, ".loader");

  /* If the ".loader" section does not exist, the objfile is probably
     not an executable.  Might be a core file...  */
  if (sect == NULL)
    return xcoff_get_core_n_import_files (abfd);

  /* The number of entries in the Import Files Table is stored in
     field l_nimpid.  This field is always at offset 16, and is
     always 4 bytes long.  Read those 4 bytes.  */
  gdb_byte buf[4];
  if (!bfd_get_section_contents (abfd, sect, buf, 16, 4))
    return -1;

  int l_nimpid = bfd_get_32 (abfd, buf);

  /* By convention, the first entry is the default LIBPATH value
     to be used by the system loader, so it does not count towards
     the number of import files.  */
  return l_nimpid - 1;
}

INIT_GDB_FILE (xcoffread)
{
  add_symtab_fns (bfd_target_xcoff_flavour, &xcoff_sym_fns);
}
