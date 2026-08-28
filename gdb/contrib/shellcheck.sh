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

version=$(shellcheck --version | grep "^version" | awk '{print $2}')

case "$version" in
    0.11.0)
	# Preferred version.
	true
	;;
    0.10.0)
	# Allowed version, but mention preferred version.  This will be
	# visible with "pre-commit run shellcheck --all-files -v".
	echo "Using shellcheck version 0.10.0, but version 0.11.0 is preferred."
	;;
    *)
	echo "Please install shellcheck version 0.11.0 (preferred) or 0.10.0"
	exit 1
	;;
esac

files=()
for f in "$@"; do
    case "$f" in
	*/configure)
	    # Skip generated files.
	    continue
	    ;;
	gdb/config/djgpp/djcheck.sh \
	    | gdb/config/djgpp/djconfig.sh \
	    | gdb/contrib/cc-with-tweaks.sh \
	    | gdb/contrib/gdb-add-index.sh \
	    | gdb/gdb_buildall.sh \
	    | gdb/gdb_mbuild.sh \
	    | gdb/regformats/regdat.sh \
	    | gdb/testsuite/lib/pdtrace.in)
	    # Skip unclean files.
	    continue
	    ;;
	*)
	    files=("${files[@]}" "$f")
	    ;;
    esac
done

if [ ${#files[@]} -eq 0 ]; then
    # Nothing to do.
    exit 0
fi

shellcheck "${files[@]}"
