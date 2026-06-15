#!/bin/bash

# Copyright (C) 2026 Free Software Foundation, Inc.
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

set -e
set -o pipefail

no_exec_files=()
for f in "$@"; do
    case $f in
	*/*.py \
	    | */*.sh \
	    | */configure \
	    | gdb/gstack-1.in \
	    | gdb/gcore-1.in \
	    | gdb/po/gdbtext \
	    | gdb/make-init-c \
	    | gdb/testsuite/lib/notty-wrap \
	    | gdb/testsuite/lib/pdtrace.in )
	    continue
	    ;;
	*)
	    no_exec_files=("${no_exec_files[@]}" "$f")
	    ;;
    esac
done

if [ ${#no_exec_files[@]} -eq 0 ]; then
    exit 0
fi

# Flag files that are executable, but not meant to be executable.

git ls-files --stage -- "${no_exec_files[@]}" \
    | (! grep '^100755 ')
