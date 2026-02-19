#!/bin/sh

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

# Wrapper around makeinfo to check makeinfo version.

required_major="$1"
required_minor="$2"
prog="$3"

shift 3

major=$("$prog" --version \
	    | grep "GNU texinfo" \
	    | sed 's/^.* \([0-9][0-9]*\)\.[0-9][0-9]*\(.*\)\?$/\1/')
minor=$("$prog" --version \
	    | grep "GNU texinfo" \
	    | sed 's/^.* [0-9][0-9]*\.\([0-9][0-9]*\)\(.*\)\?$/\1/')

if [ "$major" = "" ] || [ "$major" = "" ]; then
    echo "Cannot determine makeinfo version for $prog.  Info documentation will not be build."
    exit
fi

if [ "$major" -lt "$required_major" ] \
       || { [ "$major" -eq "$required_major" ] \
		&& [ "$minor" -lt "$required_minor" ]; }; then
    echo "$prog is too old, have $major.$minor, require $required_major.$required_minor.  Info documentation will not be build."
    exit
fi

exec "$prog" "$@"
