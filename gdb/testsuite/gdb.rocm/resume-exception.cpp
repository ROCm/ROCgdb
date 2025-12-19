/* Copyright 2024 Free Software Foundation, Inc.
<<<<<<< HEAD
   Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
=======
   Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
>>>>>>> 04e0a5a0bb887a3ed8ba4e116f0383893a39442c

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

__global__ void
raise_fpe ()
{
  volatile float a = 1, b = 0;
  float c = a / b;		/* Raise an exception.  */
  __builtin_debugtrap ();	/* A trap that mustn't be reached.  */
}

int
main ()
{
  raise_fpe<<<1, 1>>> ();
  return (hipDeviceSynchronize () != hipSuccess);
}
