# This shell script emits a C file. -*- C -*-
#   Copyright (C) 2026 Free Software Foundation, Inc.
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

# frv-elf is an embedded target and does not support shared libraries
# or position independent executables.  Like other embedded targets it
# uses genelf.em, but unlike most it defines a relocate_section function
# and thus uses elf_link_hash_table rather than generic_link_hash_table.
# This allows the target to use some dynamic linking features, eg. it
# generates a .got section to contain a _gp symbol.  Set up dynobj by
# calling ldelf_after_open_output.
#
fragment <<EOF
#include "ldelf.h"
EOF
source_em ${srcdir}/emultempl/genelf.em

LDEMUL_AFTER_OPEN_OUTPUT=ldelf_after_open_output
