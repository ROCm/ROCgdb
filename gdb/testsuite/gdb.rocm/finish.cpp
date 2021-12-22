/* Copyright (C) 2022 Free Software Foundation, Inc.
   Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cstdio>
#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                           \
  {                                                                          \
    hipError_t error = cmd;                                                  \
    if (error != hipSuccess) {                                               \
	fprintf(stderr, "error: '%s'(%d) at %s:%d\n",                        \
		hipGetErrorString(error), error, __FILE__, __LINE__);        \
	  exit(EXIT_FAILURE);                                                \
    }                                                                        \
  }

/* Define a set of types that will be returned from functions.  */

struct Empty {};

template<typename T, size_t N>
struct Custom
{
  T data[N];
};

union Union
{
  int as_int;
  char as_chars[4];
};

/* Structs (or unions) with flexible arrays are never returned by value.  */
struct FlexibleArrayMember
{
  int non_flexible;
  char flex[];
};

struct PackedWithBitField
{
  int x : 7;
  int y : 7;
  int z : 7;
};

/* A structure that is more than 8 bytes (otherwise it is returned as packed),
   which contains bitfields.  */
struct NotPackedWithBitField
{
  int foo[4];
  unsigned int x : 4;
  unsigned int y : 4;
  unsigned int z : 7;
  char c1;
  char c2;
  char c3 : 8;
  char c4 : 8;
  char : 0;
  char c5 : 8;
  char c6 : 8;
};

struct OnlyStatic
{
  __device__ static int something;
};
__device__ int OnlyStatic::something = 42;

struct WithStaticFields
{
  int a[2];
  OnlyStatic sub;
  float b;
  __device__ static int c;
  double d;
};
__device__ int WithStaticFields::c = 12;

__device__ static Empty
returnEmpty ()
{
  return {};
}

template<size_t N>
__device__ static Custom<char, N>
returnSized ()
{
  Custom<char, N> ret;
  static_assert (sizeof (ret) == N, "Invalid size");
  for (int i = 0; i < N; ++i)
    ret.data[i] = 'a' + static_cast<char> (threadIdx.x);
  return ret;
}

template<size_t N>
__device__ static Custom<Union, N>
returnCustomWithUnion ()
{
  Custom<Union, N> ret;
  for (size_t i = 0; i < N; i++)
    ret.data[i].as_int = 0x61616161;
  return ret;
}

__device__ static FlexibleArrayMember
returnFlexibleArrayMember ()
{
  return {
      .non_flexible = static_cast<int> (threadIdx.x)
  };
}

__device__ static NotPackedWithBitField
returnNotPackedWithBitField ()
{
  const int val = static_cast<int> (threadIdx.x);
  return {
      .foo = {val, val + 1, val + 2, val + 3},
      .x = static_cast<unsigned int> (val + 4),
      .y = static_cast<unsigned int> (val + 5),
      .z = static_cast<unsigned int> (val + 6),
      .c1 = 'a',
      .c2 = 'b',
      .c3 = 'c',
      .c4 = 'd',
      .c5 = 'e',
      .c6 = 'f',
  };
}

__device__ static PackedWithBitField
returnPackedWithBitField ()
{
  const int val = static_cast<int> (threadIdx.x);
  return {
      .x = static_cast<int> (val),
      .y = static_cast<int> (val + 1),
      .z = static_cast<int> (val + 2)
  };
}

__device__ static const Custom<char, 8> *
returnPtr ()
{
  static Custom<char, 8> vv {
      .data = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}
  };
  return &vv;
}

__device__ static int *
returnPtr2 (int *a)
{
  return a;
}

__device__ static const Custom<char, 8> &
returnRef ()
{
  static Custom<char, 8> vv {
      .data = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}
  };
  return vv;
};

__device__ static int &
returnRef2 (int &a)
{
  return a;
}

__device__ static WithStaticFields
returnWithStatic ()
{
  OnlyStatic::something = 12;
  WithStaticFields::c = 42;
  return {
	.a = { 8, 16 },
	.b = 3.14,
	.d = 1.60218e-19
  };
}

__device__ static OnlyStatic
returnEmptyWithStatic ()
{
  OnlyStatic::something = 56;
  return {};
}

__device__ int someGlobal = 42;

__global__ void
kernel ()
{
  returnEmpty ();
  returnEmptyWithStatic ();

  returnSized<1> ();
  returnSized<2> ();
  returnSized<4> ();
  returnSized<8> ();
  returnSized<9> ();
  returnSized<16> ();
  returnSized<32> ();
  returnSized<33> ();
  returnSized<64> ();
  returnSized<128> ();

  returnCustomWithUnion<2> ();
  returnCustomWithUnion<4> ();
  returnCustomWithUnion<6> ();
  returnCustomWithUnion<8> ();
  returnCustomWithUnion<16> ();
  returnCustomWithUnion<20> ();

  returnFlexibleArrayMember ();

  returnPackedWithBitField ();
  returnNotPackedWithBitField ();

  int myId = threadIdx.x
    + threadIdx.y * blockDim.x
    + threadIdx.z * blockDim.x * blockDim.y;

  returnPtr ();
  returnPtr2 (&myId);
  returnPtr2 (&someGlobal);

  returnRef ();
  returnRef2 (myId);
  returnRef2 (someGlobal);

  returnWithStatic ();
  returnWithStatic ();
}

int
main ()
{
  hipDeviceProp_t prop;
  CHECK (hipGetDeviceProperties (&prop, 0));
  hipLaunchKernelGGL (kernel, dim3 (1), dim3 (prop.warpSize), 0, 0);
  CHECK (hipDeviceSynchronize ());
}
