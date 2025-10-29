# Pretty-printers for ROCm OCP microscaling floating point types
# Copyright (C) 2025 Free Software Foundation, Inc.

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
import gdb.printing
from gdb.function.rocm_get_scaled_float import (
    compute_uint8e8m0,
    extract_mx_values,
    hipexp_ocp_fp_re,
)


class OcpMxFpValuePrinter(gdb.ValuePrinter):
    def __init__(self, val):
        self._val = val

    def to_string(self):
        fp = extract_mx_values(self._val)
        return "{{{}}}".format(", ".join([str(f) for f in fp]))


class OcpMxScaleValuePrinter(gdb.ValuePrinter):
    def __init__(self, val):
        self._val = val

    def to_string(self):
        return compute_uint8e8m0(self._val)


def build_ocp_mx_subprinters():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("rocm_ocp_mx")
    pp.add_printer("hipext_ocp_fp", hipexp_ocp_fp_re(), OcpMxFpValuePrinter)
    pp.add_printer("amd_scale_t", "^__amd_scale_t$", OcpMxScaleValuePrinter)
    return pp


# Register pretty printer collection automatically when the script is sourced by GDB
gdb.printing.register_pretty_printer(gdb.current_objfile(), build_ocp_mx_subprinters())
