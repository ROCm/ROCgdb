#!/bin/sh

# Copyright (C) 2026 Free Software Foundation, Inc.
#
# This file is part of GDB.
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Post-process a file produced by flex or bison/byacc, reading it from
# standard input and writing the result to standard output.
#
# Usage:
#
#    ./post-process-parser-output.sh GENERATOR NAME < INPUT > OUTPUT
#
# Where GENERATOR is "bison" (regardless of whether we're using the actual
# bison or byacc) or "flex", and NAME is the name of the parser (e.g.
# "cp-name-parser"), used to make the global symbols names bison produces
# unique.

set -e

generator="$1"

# Convert e.g. "cp-name-parser" to "cp_name_parser".
name=$(echo "$2" | sed -e 's/-/_/g')

common='
/extern.*malloc/d
/extern.*realloc/d
/extern.*free/d
/include.*malloc.h/d
s/\([^x]\)malloc/\1xmalloc/g
s/\([^x]\)realloc/\1xrealloc/g
s/\([ \t;,(]\)free\([ \t]*[&(),]\)/\1xfree\2/g
s/\([ \t;,(]\)free$/\1xfree/g
'

case "$generator" in
    bison)
	sed -e "$common" \
	    -e '/^#line.*y.tab.c/d' \
	    -e "s/YYSTYPE/${name}_YYSTYPE/g" \
	    -e "s/yyalloc/${name}_yyalloc/g" \
	    -e "s/yysymbol_kind_t/${name}_yysymbol_kind_t/g" \
	    -e "s/YYSTACKDATA/${name}_YYSTACKDATA/g"
	;;
    flex)
	sed -e "$common" \
	    -e 's/yy_flex_xrealloc/yyxrealloc/g'
	;;
    *)
	echo "$0: unknown parser generator \"$generator\"" >&2
	exit 1
	;;
esac
