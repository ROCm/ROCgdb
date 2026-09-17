/* Copyright 2021-2024 Free Software Foundation, Inc.

   This file is part of GDB.

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
#include <stdio.h>
#include <stdlib.h>

#include "rocm-test-utils.h"

__global__ void
kernel ()
{

  /* Simple kernel which loads from address 0 to trigger a pagefault.
     With precise-memory disabled the fault is reported after the s_nop.
     With precise-memory enabled the reported location depends on the
     XNACK mode.  It is the s_load_dword itself when XNACK is enabled, or
     the following s_nop when XNACK is disabled.  */
  asm volatile ("s_mov_b64 [s10, s11], 0\n"
		"s_load_dword s12, [s10, s11]\n"
		"s_nop 0"
		:
		:
		: "s10", "s11", "s12");
}

int
main (int argc, char* argv[])
{
  kernel<<<1, 1>>> ();
  CHECK (hipDeviceSynchronize ());
  return 0;
}
