/* Copyright (C) 2021 Free Software Foundation, Inc.
   Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

   This file is used only as a source text for source level debugging.  */

struct test_struct
{
  int int_elem;
  char char_elem;
  int array_elem[32];
};

__constant struct test_struct const_struct = {32, '2',
    {32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}};

__constant int const_array[32] = {
  1, 1, 1, 1, 1, 5, 5, 7,
  2, 2, 2, 2, 2, 5, 5, 10,
  3, 3, 3, 3, 3, 5, 5, 2,
  4, 4, 4, 4, 4, 5, 5, 3};

 __attribute__((noinline)) int GenValue ()
{
  int array[32] = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                   5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8};
  unsigned gid = get_global_id(0);
  __constant int *pconst_array_elem1, *pconst_array_elem2;

  if (gid % 2 == 1)
    pconst_array_elem1 = &const_array[10];
  else
    pconst_array_elem1 = &const_array[gid];

  if (gid % 7 == 1)
    pconst_array_elem2 = &const_array[gid];
  else
    pconst_array_elem2 = &const_array[12];

  return array[gid] + *pconst_array_elem1 * *pconst_array_elem2;
}

void ChangeLocalContent (__local int* plocal_content)
{
  int array[32] = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                   5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8};
  unsigned gid = get_global_id(0);

  if (gid % 4)
  {
    constant int *pconst_array_elem;
    int temp;

    if (gid % 2 == 1)
      pconst_array_elem = &const_array[10];
    else
      pconst_array_elem =  &const_array[gid];

    if (gid % 3 == 1)
      temp = array[gid];
    else
      temp = array[10];

    (*plocal_content) = temp + *pconst_array_elem;
  }
  else
    (*plocal_content) = GenValue();
}

 __attribute__((noinline)) void SendResults (__local struct test_struct* plocal_struct, __global struct test_struct* out)
{
  unsigned gid = get_global_id(0);

  if (!gid)
  {
    out->int_elem = plocal_struct->int_elem + const_struct.int_elem;
    out->char_elem = plocal_struct->char_elem + const_struct.char_elem;
  }

  out->array_elem[gid] = plocal_struct->array_elem[gid] + const_struct.array_elem[gid] + const_array[gid];
}

void kernel AddrClassTest(const __global int* in, __global struct test_struct* out)
{
  unsigned gid = get_global_id(0);
  __local struct test_struct local_struct;
  local_struct.char_elem = '2';

  if (gid % 3)
  {
    atomic_add(&local_struct.int_elem, GenValue());
    local_struct.char_elem = (char)in[gid];
  }
  else
    local_struct.int_elem = in[gid];

  if (gid % 8)
    ChangeLocalContent (&local_struct.array_elem[gid]);
  else
    local_struct.array_elem[gid] = const_array[gid];

  SendResults (&local_struct, out);
}
