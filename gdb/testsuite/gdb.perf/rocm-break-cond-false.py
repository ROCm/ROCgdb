# Copyright (C) 2021 Free Software Foundation, Inc.
# Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

from perftest import perftest

# Set a breakpoint with a condition that evals false for all waves
# except one.  GDB will need to step all (except one) waves over the
# breakpoint.
class ROCmBreakCondFalse(perftest.TestCaseWithBasicMeasurements):
    def __init__(self):
        super(ROCmBreakCondFalse, self).__init__("rocm-break-cond-false")

    def warm_up(self):
        pass

    def _run(self):
        # By default _run stops at main(), the first "c" will make the
        # program hit the conditional breakpoint, the second "c" will
        # make it run to the end.
        gdb.execute("c", False, True)
        gdb.execute("c", False, True)

    def execute_test(self):
        gdb.execute("set pagination off", False, True)
        gdb.execute("set breakpoint pending on", False, True)
        gdb.execute('break kernel if $_streq($_wave_id, "(0,0,0)/0")', False, True)
        func = lambda: self._run()
        self.measure.measure(func, 1)
