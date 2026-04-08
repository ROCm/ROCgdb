# This shell script emits a C file. -*- C -*-
#   Copyright (C) 1991-2026 Free Software Foundation, Inc.
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
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston,
# MA 02110-1301, USA.
#

# This file is sourced from elf.em, and defines extra hppa-elf
# specific routines.
#
fragment <<EOF

#include "ldctor.h"
#include "elf64-hppa.h"

/* Stop the linker merging .text sections on relocatable links,
   add millicode library to the list of input files, and ignore
   unresolved symbols in shared libraries.  */

static void
hppa64elf_after_parse (void)
{
  /* Stop the linker merging .text sections on relocatable links.  */
  if (bfd_link_relocatable (&link_info))
    lang_add_unique (".text");

  /* We always need to link against milli.a on HP-UX.  */
  lang_add_input_file ("milli.a",
		       lang_input_file_is_search_file_enum,
		       NULL);

  /* Default 64-bit search paths on HP-UX.  */
  ldfile_add_library_path ("/lib/pa20_64", false);
  ldfile_add_library_path ("/usr/lib/pa20_64", false);

  /* HP-UX shared libraries have some unresolved symbols.  We need to
     ignore unresolved symbols in shared libraries.  */
  link_info.unresolved_syms_in_shared_libs = RM_IGNORE;

  ldelf_after_parse ();
}
EOF

# Put these extra hppaelf routines in ld_${EMULATION_NAME}_emulation
#
LDEMUL_AFTER_PARSE=hppa64elf_after_parse
