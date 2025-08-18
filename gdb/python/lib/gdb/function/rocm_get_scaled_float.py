# ROCm OCP microscaling floating point convenience functions.
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

"""$_mx_get_scaled_float"""

import re
import math
import gdb


def extract_mx_bits(val, sgn_msk, exp_msk, mnts_msk):
    """
    Extract sign, exponent, and mantissa bits from a given value.

    Arguments:
    val      -- Input 8-bit value from which bits are extracted.
    sgn_msk  -- Bitmask to isolate the sign bit(s).
    exp_msk  -- Bitmask to isolate the exponent bits.
    mnts_msk -- Bitmask to isolate the mantissa bits.

    Returns:
    A tuple (sgn, exp, mnts) containing the extracted sign, exponent, and mantissa.
    """
    val &= 0xFF
    ctz = (sgn_msk & -sgn_msk).bit_length() - 1
    sgn = (val & sgn_msk) >> ctz
    ctz = (exp_msk & -exp_msk).bit_length() - 1
    exp = (val & exp_msk) >> ctz
    ctz = (mnts_msk & -mnts_msk).bit_length() - 1
    mnts = (val & mnts_msk) >> ctz
    return sgn, exp, mnts


def compute_mx_value(sign, exponent, bias, mantissa, mbits):
    """Calculate microscaling subnormal or normal value"""
    if exponent == 0:
        result = ((-1) ** sign) * (2 ** (1 - bias)) * ((2**-mbits) * mantissa)
    else:
        result = (
            ((-1) ** sign)
            * (2 ** (exponent - bias))
            * (1 + 2 ** (-1 * mbits) * mantissa)
        )
    return result


def compute_fp8e4m3(val):
    bias = 7
    mbits = 3
    sign, exponent, mantissa = extract_mx_bits(val, 0x80, 0x78, 0x07)
    if exponent == 0xF and mantissa == 0x7:
        return math.nan
    if exponent == 0 and mantissa == 0:
        return float(0)
    return compute_mx_value(sign, exponent, bias, mantissa, mbits)


def mx_fp8e4m3(val):
    fp = [compute_fp8e4m3(int(val))]
    return fp


def mx_fp8x2e4m3(val):
    fp = [compute_fp8e4m3(int(val) >> (8 if k else 0)) for k in range(0, 2)]
    return fp


def compute_fp8e5m2(val):
    bias = 15
    mbits = 2
    sign, exponent, mantissa = extract_mx_bits(val, 0x80, 0x7C, 0x03)
    if exponent == 0x1F and mantissa == 0:
        return -math.inf if sign else math.inf
    if exponent == 0x1F and (mantissa == 1 or mantissa == 2 or mantissa == 3):
        return math.nan
    if exponent == 0 and mantissa == 0:
        return float(0)
    return compute_mx_value(sign, exponent, bias, mantissa, mbits)


def mx_fp8e5m2(val):
    fp = [compute_fp8e5m2(int(val))]
    return fp


def mx_fp8x2e5m2(val):
    fp = [compute_fp8e5m2(int(val) >> (8 if k else 0)) for k in range(0, 2)]
    return fp


def compute_fp6e2m3(val):
    bias = 1
    mbits = 3
    sign, exponent, mantissa = extract_mx_bits(val, 0x20, 0x18, 0x07)
    if exponent == 0 and mantissa == 0:
        return float(0)
    return compute_mx_value(sign, exponent, bias, mantissa, mbits)


def mx_fp6x32e2m3(val):
    v192bits = (
        (int(val[5]) << 160)
        | (int(val[4]) << 128)
        | (int(val[3]) << 96)
        | (int(val[2]) << 64)
        | (int(val[1]) << 32)
        | (int(val[0]) << 0)
    )
    fp = [compute_fp6e2m3((v192bits >> (6 * k)) & 0xFF) for k in range(0, 32)]
    return fp


def compute_fp6e3m2(val):
    bias = 3
    mbits = 2
    sign, exponent, mantissa = extract_mx_bits(val, 0x20, 0x1C, 0x03)
    if exponent == 0 and mantissa == 0:
        return float(0)
    return compute_mx_value(sign, exponent, bias, mantissa, mbits)


def mx_fp6x32e3m2(val):
    v192bits = (
        (int(val[5]) << 160)
        | (int(val[4]) << 128)
        | (int(val[3]) << 96)
        | (int(val[2]) << 64)
        | (int(val[1]) << 32)
        | (int(val[0]) << 0)
    )
    fp = [compute_fp6e3m2((v192bits >> (6 * k)) & 0xFF) for k in range(0, 32)]
    return fp


def compute_fp4e2m1(val):
    bias = 1
    mbits = 1
    sign, exponent, mantissa = extract_mx_bits(val, 0x8, 0x6, 0x1)
    if exponent == 0 and mantissa == 0:
        return float(0)
    else:
        return compute_mx_value(sign, exponent, bias, mantissa, mbits)


def mx_fp4x2e2m1(val):
    fp = [compute_fp4e2m1(int(val) >> (4 if k else 0)) for k in range(0, 2)]
    return fp


def compute_uint8e8m0(val):
    v = int(val)
    if v < -127 or v >= 128:
        return math.nan
    else:
        return float(2 ** (float(v) + 127))


def extract_mx_values(mxvar):
    if re.match(r"^__hipext_ocp_fp8_e4m3$", str(mxvar.type)):
        return mx_fp8e4m3(mxvar["__x"])

    if re.match(r"^__hipext_ocp_fp8x2_e4m3$", str(mxvar.type)):
        return mx_fp8x2e4m3(mxvar["__x"])

    if re.match(r"^__hipext_ocp_fp8_e5m2$", str(mxvar.type)):
        return mx_fp8e5m2(mxvar["__x"])

    if re.match(r"^__hipext_ocp_fp8x2_e5m2$", str(mxvar.type)):
        return mx_fp8x2e5m2(mxvar["__x"])

    if re.match(r"^__hipext_ocp_fp6x32_e2m3$", str(mxvar.type)):
        return mx_fp6x32e2m3(mxvar["__x"])

    if re.match(r"^__hipext_ocp_fp6x32_e3m2$", str(mxvar.type)):
        return mx_fp6x32e3m2(mxvar["__x"])

    if re.match(r"^__hipext_ocp_fp4x2_e2m1$", str(mxvar.type)):
        return mx_fp4x2e2m1(mxvar["__x"])

    raise TypeError("unknown ROCm microscaling scalar type")


def hipexp_ocp_fp_re():
    return r"^__hipext_ocp_fp\d+(x\d+)?_e\dm\d$"


class _MxGetScaledFloat(gdb.Function):
    """$_mx_get_scaled_float - get scaled values(s) of ROCm microscaling objects.

    Usage: $_mx_get_scaled_float(mx_scalar, mx_scale=1)

    Returns:
      Scaled value(s) of the ROCm microscaling scalar object"""

    def __init__(self):
        super().__init__("_mx_get_scaled_float")

    def invoke(self, mxvar, mxscale=None):
        if not re.match(hipexp_ocp_fp_re(), str(mxvar.type)):
            raise TypeError("mx_scalar must be of ROCm microscaling scalar type")

        if mxscale is None:
            mxscale = gdb.Value(1.0).cast(gdb.lookup_type("float"))
        mxscalev = None

        # If mxscale is __amd_scale_t var, convert its bit encoding into float value.
        # Otherwise, mxscale defaults to 1 or is a user specified int/float value.
        if re.match(r"^__amd_scale_t$", str(mxscale.type)):
            mxscalev = compute_uint8e8m0(mxscale)
        else:
            mxscalev = float(mxscale)
        if not mxscalev >= 1:
            raise ValueError("usage: _mx_get_scaled_float(mx_scalar, mx_scale >= 1)")

        # Extract the mx var values into list, create list-sized bytearray.
        mxvarvl = extract_mx_values(mxvar)
        ftype = gdb.lookup_type("float")
        vtype = ftype.vector(len(mxvarvl) - 1)
        barray = bytearray(vtype.sizeof)

        # Convert values list to gdb.Value list, then copy it into a bytearray.
        # Below float(v) is a host (python) float type, created from the lower
        # precision inferior microscaling type.
        gvals = [gdb.Value(float(v) * mxscalev).cast(ftype) for v in mxvarvl]
        fsize = ftype.sizeof
        for i, val in enumerate(gvals):
            barray[i * fsize : (i + 1) * fsize] = gvals[i].bytes

        # Convert the bytearray into the returned gdb.Value.
        return gdb.Value(barray, vtype)


# GDB will import us automagically via gdb/__init__.py.
_MxGetScaledFloat()
