/*
Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime.h>

/* Helper function where the debugger places a breakpoint to be sure no wave
   can run to completion without the debugger noticing.  __forceinline__ and
   volatile asm guarantee the s_nop is emitted in-line in the kernel, giving
   the debugger a reliable address to break on.  */
__device__ __forceinline__ void
end_of_kernel ()
{
  __asm__ volatile ("s_nop 0" ::: );  /* Debugger breaks on the inlined s_nop.  */
}

__global__ void __attribute__ ((optnone))
kern ()
{
  /* The test will need to step a few times.  */
  __builtin_amdgcn_s_sleep (20);
  __builtin_amdgcn_s_sleep (20);
  __builtin_amdgcn_s_sleep (20);
  __builtin_amdgcn_s_sleep (20);
  __builtin_amdgcn_s_sleep (20);
  __builtin_amdgcn_s_sleep (20);

  end_of_kernel ();
}


int
main ()
{
  /* Dispatch many workitems to ensure that when hog terminates there are
     enough pending waves that are created by hw to occupy vacated slots.

     TODO: Replace hard coded values based on output of "info agents". */
  kern<<<4096, 256>>> ();
  return hipDeviceSynchronize () != hipSuccess;
}
