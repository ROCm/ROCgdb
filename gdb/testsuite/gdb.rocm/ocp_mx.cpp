/* This testcase is part of GDB, the GNU debugger.

   Copyright 2025 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <hip/hip_runtime.h>
#include <hip/hip_ext_ocp.h>
#include <stdio.h>
#include <stdlib.h>


#define CHECK(cmd)                                                           \
  {                                                                          \
    hipError_t error = cmd;                                                  \
    if (error != hipSuccess) {                                               \
	fprintf(stderr, "error: '%s'(%d) at %s:%d\n",                        \
		hipGetErrorString(error), error, __FILE__, __LINE__);        \
	exit(EXIT_FAILURE);                                                  \
    }                                                                        \
  }


__host__ __device__ void
fp8e4m3_values ()
{
  /* FP8 E4M3 format
  +---------------+--------------------+
  |   Attribute   |        E4M3        |
  +---------------+--------------------+
  | Inf           | N/A                |
  | NAN           | S-1111-111         |
  | Zero          | S-0000-000         |
  | Max Normal    | S-1111-110: +-448  |
  | Min Normal    | S-0001-000: +-2^-6 |
  | Max Subnorm   | S-0000-111         |
  | Min Subnorm   | S-0000-001         |
  +---------------+--------------------+ */
  union u_e4m3
  {
    /* type punning, interpret same memory as different types */
    __amd_fp8x2_storage_t s;
    __hipext_ocp_fp8x2_e4m3 e;
  };

  /* fp8_e4m3 +/- zero */
  union u_e4m3 fp8e4m3zero  {0x00 | 0x80 << 8};
  /* nan */
  union u_e4m3 fp8e4m3nan   {0x7F | 0xFF << 8};
  /* +/- normal maximum */
  union u_e4m3 fp8e4m3nmax  {0x7E | 0xFE << 8};
  /* +/- normal minimum */
  union u_e4m3 fp8e4m3nmin  {0x08 | 0x88 << 8};
  /* +/- normal random */
  union u_e4m3 fp8e4m3nrndm {0x53 | 0xC9 << 8};
  /* +/- subnormal maximum */
  union u_e4m3 fp8e4m3smax  {0x07 | 0x87 << 8};
  /* +/- subnormal minimum */
  union u_e4m3 fp8e4m3smin  {0x01 | 0x81 << 8};
  /* +/- subnormal random */
  union u_e4m3 fp8e4m3srndm {0x09 | 0x85 << 8};

  /* builtin fp8e4m3 +/- (sub)normal randoms */
  __hipext_ocp_fp8_e4m3 e4m3 {112.00};
  __hipext_ocp_fp8x2_e4m3 e4m3n {224.00, -128.00};
  __hipext_ocp_fp8x2_e4m3 e4m3s {0.02734375, -0.00390625};

  /* builtin uint8e8m0 */
  __amd_scale_t e8m0 {-127};

  /* break here 1 */
  return;
}

__host__ __device__ void
fp8e5m2_values ()
{
  /* FP8 E5M2 format
  +---------------+----------------------+
  |   Attribute   |         E5M2         |
  +---------------+----------------------+
  | Inf           | S-11111-00           |
  | NAN           | S-11111-**           |
  | Zero          | S-00000-00           |
  | Max Normal    | S-11110-11: +-57,344 |
  | Min Normal    | S-00001-00: +-2^-14  |
  | Max Subnorm   | S-00000-11           |
  | Min Subnorm   | S-00000-01           |
  +---------------+----------------------+ */
  union u_e5m2
  {
    __amd_fp8x2_storage_t s;
    __hipext_ocp_fp8x2_e5m2 e;
  } u_e4m3;

  /* fp8_e5m2 +/- zero */
  union u_e5m2 fp8e5m2zero  {0x00 | 0x80 << 8};
  /* fp8_e5m2 +/- infinity */
  union u_e5m2 fp8e5m2inf   {0x7C | 0xFC << 8};
  /* fp8_e5m2 nan(s) */
  union u_e5m2 fp8e5m2nan0  {0x7D | 0xFD << 8};
  union u_e5m2 fp8e5m2nan1  {0x7E | 0xFE << 8};
  union u_e5m2 fp8e5m2nan2  {0x7F | 0xFF << 8};
  /* fp8_e5m2 +/- normal maximum */
  union u_e5m2 fp8e5m2nmax  {0x7B | 0xFB << 8};
  /* fp8_e5m2 +/- normal minimum */
  union u_e5m2 fp8e5m2nmin  {0x04 | 0x84 << 8};
  /* fp8_e5m2 +/- normal random */
  union u_e5m2 fp8e5m2nrndm {0x41 | 0xA8 << 8};
  /* fp8_e5m2 +/- subnormal maximum */
  union u_e5m2 fp8e5m2smax  {0x03 | 0x83 << 8};
  /* fp8_e5m2 +/- subnormal minimum */
  union u_e5m2 fp8e5m2smin  {0x01 | 0x81 << 8};
  /* fp8_e5m2 +/- subnormal random */
  union u_e5m2 fp8e5m2srndm {0x02 | 0x82 << 8};

  /* builtin fp8e5m2 +/- (sub)normal randoms */
  __hipext_ocp_fp8_e5m2 e5m2 {-224.00};
  __hipext_ocp_fp8x2_e5m2 e5m2n {28672.0, -3072.0};
  __hipext_ocp_fp8x2_e5m2 e5m2s {0.000091552734375, -0.000030517578125};

  /* builtin uint8e8m0 */
  __amd_scale_t e8m0 {-126};

  /* break here 2 */
  return;
}

__host__ __device__ void
fp6e2m3_values ()
{
  /* FP6 format
  +---------------+------------------------------------+
  |   Attribute   |               E2M3                 |
  +---------------+------------------------------------+
  | Infinities    | N/A                                |
  | NaN           | N/A                                |
  | Zeros         | S-00-000                           |
  | Max normal    | S-11-111 = +-2^2 ×1.875 = +-7.5    |
  | Min normal    | S-01-000 = +-2^0 ×1.0 = +-1.0      |
  | Max subnorm   | S-00-111 = +-2^0 ×0.875 = +-0.875  |
  | Min subnorm   | S-00-001 = +-2^0 ×0.125 = +-0.125  |
  +---------------+------------------------------------+ */
  union u_e2m3
  {
    __amd_fp6x32_storage_t s;
    __hipext_ocp_fp6x32_e2m3 e;
  };

  /* fp6_e2m3 zeroes */
  union u_e2m3 fp6e2m3zero  {0x00 | 0x8 << 8};
  /* fp6_e2m3 +/- normal maximum */
  union u_e2m3 fp6e2m3nmax  {0xDF | 0xF << 8};
  /* fp6_e2m3 +/- normal minimum */
  union u_e2m3 fp6e2m3nmin  {0x08 | 0xA << 8};
  /* fp6_e2m3 +/- normal random */
  union u_e2m3 fp6e2m3nrndm {0x1A | 0xB << 8};
  /* fp6_e2m3 +/- subnormal maximum */
  union u_e2m3 fp6e2m3smax  {0xC7 | 0x9 << 8};
  /* fp6_e2m3 +/- subnormal minimum */
  union u_e2m3 fp6e2m3smin  {0x41 | 0x8 << 8};
  /* fp6_e2m3 +/- subnormal random */
  union u_e2m3 fp6e2m3srndm {0xC5 | 0x8 << 8};

  /* builtin fp6e2m3 +/- (sub)normal randoms */
  __amd_fp16x32_storage_t n {7.5, -1.25};
  __hipext_ocp_fp6x32_e2m3 e2m3n {n, 0};
  __amd_fp16x32_storage_t s {0.375, -0.125};
  __hipext_ocp_fp6x32_e2m3 e2m3s {s, 0};

  /* builtin uint8e8m0 */
  __amd_scale_t e8m0 {-125};

  /* break here 3 */
  return;
}

__host__ __device__ void
fp6e3m2_values ()
{
  /* FP6 format
  +---------------+-------------------------------------+
  |   Attribute   |                E3M2                 |
  +---------------+-------------------------------------+
  | Infinities    | N/A                                 |
  | NaN           | N/A                                 |
  | Zeros         | S-000-00                            |
  | Max normal    | S-111-11 = +-2^4 ×1.75 = +-28.0     |
  | Min normal    | S-001-00 = +-2^-2 ×1.0 = +-0.25     |
  | Max subnorm   | S-000-11 = +-2^-2 ×0.75 = +-0.1875  |
  | Min subnorm   | S-000-01 = +-2^-2 ×0.25 = +-0.0625  |
  +---------------+-------------------------------------+ */
  union u_e3m2
  {
    __amd_fp6x32_storage_t s;
    __hipext_ocp_fp6x32_e3m2 e;
  };

  /* fp6_e3m2 zeroes */
  union u_e3m2 fp6e3m2zero  {0x00 | 0x8 << 8};
  /* fp6_e3m2 +/- normal maximum */
  union u_e3m2 fp6e3m2nmax  {0xDF | 0xF << 8};
  /* fp6_e3m2 +/- normal minimum */
  union u_e3m2 fp6e3m2nmin  {0x04 | 0x9 << 8};
  /* fp6_e3m2 +/- normal random */
  union u_e3m2 fp6e3m2nrndm {0x4B | 0xC << 8};
  /* fp6_e3m2 +/- subnormal maximum */
  union u_e3m2 fp6e3m2smax  {0xC3 | 0x8 << 8};
  /* fp6_e3m2 +/- subnormal minimum */
  union u_e3m2 fp6e3m2smin  {0x41 | 0x8 << 8};
  /* fp6_e3m2 +/- subnormal random */
  union u_e3m2 fp6e3m2srndm {0x82 | 0x8 << 8};

  /* builtin fp6e3m2 +/- (sub)normal randoms */
  __amd_fp16x32_storage_t n {14.0, -1.25};
  __hipext_ocp_fp6x32_e3m2 e3m2n {n, 0};
  __amd_fp16x32_storage_t s {0.125, -0.75};
  __hipext_ocp_fp6x32_e3m2 e3m2s {s, 0};

  /* builtin uint8e8m0 */
  __amd_scale_t e8m0 {-124};

  /* break here 4 */
  return;
}

__host__ __device__ void
fp4e2m1_values ()
{
  /* FP4 format
  +---------------+------------------------------+
  |   Attribute   |            E2M1              |
  +---------------+------------------------------+
  | Infinities    | N/A                          |
  | NaN           | N/A                          |
  | Zeros         | S-00-0                       |
  | Max normal    | S-11-1 = +-2^2 ×1.5 = +-6.0  |
  | Min normal    | S-01-0 = +-2^0 ×1.0 = +-1.0  |
  | Max subnorm   | S-00-1 = +-2^0 ×0.5 = +-0.5  |
  | Min subnorm   | S-00-1 = +-2^0 ×0.5 = +-0.5  |
  +---------------+------------------------------+ */
  union u_e2m1
  {
    __amd_fp4x2_storage_t s;
    __hipext_ocp_fp4x2_e2m1 e;
  };

  /* fp4_e2mi zeroes */
  union u_e2m1 fp4e2m1zero  {0x80};
  /* fp4_e2mi +/- normal maximum */
  union u_e2m1 fp4e2m1nmax  {0xF7};
  /* fp4_e2mi +/- normal minimum */
  union u_e2m1 fp4e2m1nmin  {0xA2};
  /* fp4_e2mi +/- normal random */
  union u_e2m1 fp4e2m1nrndm {0xC5};
  /* fp4_e2mi +/- subnormal maximum */
  union u_e2m1 fp4e2m1smax  {0x91};
  /* fp4_e2mi +/- subnormal minimum */
  union u_e2m1 fp4e2m1smin  {0x91};
  /* fp4_e2mi +/- subnormal random */
  union u_e2m1 fp4e2m1srndm {0x91};

  /* builtin fp4 +/- (sub)normal randoms */
  __amd_floatx2_storage_t n {3.0, -1.5};
   __hipext_ocp_fp4x2_e2m1 e2m1n {n, 0};
  __amd_floatx2_storage_t s {0.5, -0.5};
  __hipext_ocp_fp4x2_e2m1 e2m1s {s, 0};

  /* builtin uint8e8m0 */
  __amd_scale_t e8m0 {-123};

  /* break here 5 */
  return;
}

__host__ __device__ void
do_fpmx_pretty_printer ()
{
  fp8e4m3_values ();
  fp8e5m2_values ();
  fp6e2m3_values ();
  fp6e3m2_values ();
  fp4e2m1_values ();
}

__global__ void
do_fpmx_pretty_printer_device ()
{
  do_fpmx_pretty_printer ();
}

void
do_fpmx_pretty_printer_host ()
{
  do_fpmx_pretty_printer ();
}

int
main ()
{
  do_fpmx_pretty_printer_device<<<dim3(1), dim3(1), 0, 0>>> ();
  CHECK (hipDeviceSynchronize ());
  do_fpmx_pretty_printer_host ();
  return 0;
}
