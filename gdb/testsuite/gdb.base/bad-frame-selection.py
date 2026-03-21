# Copyright 2026 Free Software Foundation, Inc.
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

import gdb.printing

# When set to True the pretty printer will invalidate the frame cache.
invalidate = False


class FooPrinter(gdb.ValuePrinter):
    """Pretty printer for struct foo that optionally invalidates
    the frame cache, to test that GDB handles frame cache
    invalidation from within a pretty printer."""

    def __init__(self, val):
        self.__val = val

    def to_string(self):
        global invalidate

        if invalidate:
            gdb.invalidate_cached_frames()
            invalidate = False
            prefix = "<invalidate-frame>"
        else:
            prefix = ""

        return (
            prefix + "a=<" + str(self.__val["a"]) + "> b=<" + str(self.__val["b"]) + ">"
        )


def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("bad-frame-selection")
    pp.add_printer("foo", "^foo$", FooPrinter)
    return pp


gdb.printing.register_pretty_printer(gdb.current_objfile(), build_pretty_printer())
