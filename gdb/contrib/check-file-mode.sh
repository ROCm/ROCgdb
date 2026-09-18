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

# Flag files that are executable, but not meant to be executable.
check_exec ()
{
    no_exec_files=()
    for f in "$@"; do
	case $f in
	    */*.py \
		| */*.sh )
		# Shell script or python.
		continue
		;;
	    gdb/po/gdbtext \
		| gdb/make-init-c \
		| gdb/testsuite/lib/notty-wrap )
		# Shell script without .sh extension.
		continue
		;;
	    */configure \
		| gdb/gstack-1.in \
		| gdb/gcore-1.in )
		# Used to generate shell script.
		continue
		;;
	    *)
		no_exec_files=("${no_exec_files[@]}" "$f")
		;;
	esac
    done

    if [ ${#no_exec_files[@]} -eq 0 ]; then
	return
    fi

    if ! git ls-files --stage -- "${no_exec_files[@]}" \
	    | (! grep '^100755 '); then
	echo "Found executable mode (100755) on file without .sh or .py"
	echo "Please fix or add to exception list in $0"
	exit 1
    fi
}

# Flag symlinks.  Symlinks are support by git, but can be problematic on
# platforms without proper support for it [1].
# [1] https://gitforwindows.org/symbolic-links.html
check_symlinks ()
{
    if ! git ls-files --stage -- "$@" \
	    | (! grep '^120000 '); then
	echo "Found symlink mode (120000)"
	echo "Please replace by copy"
	exit 1
    fi
}

check_exec "$@"
check_symlinks "$@"
