# Copyright (C) 2026 Free Software Foundation, Inc.
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

import gdb


class PythonListCommand(gdb.Command):
    def __init__(self):
        gdb.Command.__init__(self, "py-list", gdb.COMMAND_USER)

    def invoke(self, args, from_tty):
        argv = gdb.string_to_argv(args)
        assert len(argv) == 0 or len(argv) == 1

        pc = gdb.parse_and_eval("$pc")
        sal = gdb.current_progspace().find_pc_line(pc)
        assert sal.symtab is not None

        if len(argv) == 0:
            source_lines = sal.symtab.source_lines()
            start = 1
        else:
            parts = argv[0].split(",")
            source_lines = sal.symtab.source_lines(
                first=int(parts[0]), last=int(parts[1])
            )
            start = int(parts[0])
        if source_lines is None:
            print("Source file {} not found.".format(sal.symtab.fullname()))
        else:
            lns = gdb.Style("line-number")
            for i, l in enumerate(source_lines, start=start):
                print("%s\t%s" % (lns.apply(str(i)), l), end="")


class PythonListUnstyledCommand(gdb.Command):
    def __init__(self):
        gdb.Command.__init__(self, "py-list-unstyled", gdb.COMMAND_USER)

    def invoke(self, args, from_tty):
        argv = gdb.string_to_argv(args)
        assert len(argv) == 0 or len(argv) == 1

        pc = gdb.parse_and_eval("$pc")
        sal = gdb.current_progspace().find_pc_line(pc)
        assert sal.symtab is not None

        if len(argv) == 0:
            source_lines = sal.symtab.source_lines(unstyled=True)
            start = 1
        else:
            parts = argv[0].split(",")
            source_lines = sal.symtab.source_lines(
                first=int(parts[0]), last=int(parts[1]), unstyled=True
            )
            start = int(parts[0])
        if source_lines is None:
            print("Source file {} not found.".format(sal.symtab.fullname()))
        else:
            for i, l in enumerate(source_lines, start=start):
                print("%d\t%s" % (i, l), end="")


PythonListCommand()
PythonListUnstyledCommand()
