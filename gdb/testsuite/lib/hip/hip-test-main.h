/* GDB HIP testing header defining main.

   Copyright (C) 2021 Free Software Foundation, Inc.
   Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef GDB_HIP_TEST_MAIN_H
#define GDB_HIP_TEST_MAIN_H

/* The driver doesn't know which overload is used by each testcase.
   To address that, we make each possible overload weak, and then the
   driver checks which one was actually defined.  */
int __device__ __attribute__((weak)) gdb_hip_test_main (int argc, char **argv, char **envp);
int __device__ __attribute__((weak)) gdb_hip_test_main (int argc, char **argv);
int __device__ __attribute__((weak)) gdb_hip_test_main (int argc, const char **argv);
int __device__ __attribute__((weak)) gdb_hip_test_main (int argc, char *const *argv);
int __device__ __attribute__((weak)) gdb_hip_test_main ();

#endif
