/* Copyright 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

/* C++-specific OpenMP-offload features.  Exercises a templated device
   function and a callable functor used inside a '#pragma omp target'
   region.  */

#include <cstdio>

#define N 16

#pragma omp declare target
template <typename T>
static T
add_t (T a, T b)
{
  return a + b;		/* tmpl-line */
}

/* A named caller of add_t, so the backtrace at the add_t breakpoint
   shows a stable intermediate frame (sum_pair -> add_t) rather than
   only the compiler-generated kernel frame.  */
template <typename T>
static T __attribute__ ((noinline))
sum_pair (T a, T b)
{
  return add_t<T> (a, b);
}

struct multiplier
{
  int factor;

  int
  operator() (int x) const
  {
    return x * factor;	/* functor-line */
  }
};

/* Likewise a named caller of the functor.  */
static int __attribute__ ((noinline))
scale (const multiplier &m, int x)
{
  return m (x);
}
#pragma omp end declare target

int
main ()
{
  int a[N];
  int b[N];
  int sum[N];
  int prod[N];

  for (int i = 0; i < N; i++)
    {
      a[i] = i;
      b[i] = 2 * i;
      sum[i] = 0;
      prod[i] = 0;
    }

  multiplier m { 3 };

  #pragma omp target						\
	      map(to:a[0:N], b[0:N], m)				\
	      map(tofrom:sum[0:N], prod[0:N])
  {
    for (int i = 0; i < N; i++)
      {
	sum[i] = sum_pair<int> (a[i], b[i]);
	prod[i] = scale (m, a[i]);
      }
  }

  std::printf ("sum[1]=%d prod[1]=%d\n", sum[1], prod[1]);
  return 0;
}
