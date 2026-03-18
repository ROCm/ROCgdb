/* Common Linux target-dependent definitions for AArch64 MTE

   Copyright (C) 2021-2026 Free Software Foundation, Inc.

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

#ifndef GDB_ARCH_AARCH64_MTE_LINUX_H
#define GDB_ARCH_AARCH64_MTE_LINUX_H

#define AARCH64_HWCAP2_MTE  (1 << 18)

/* The MTE regset consists of a single 64-bit register.  */
#define AARCH64_LINUX_SIZEOF_MTE_REGSET 8

/* Memory tagging definitions.  */
#define AARCH64_SEGV_MTEAERR 8
#define AARCH64_SEGV_MTESERR 9

/* Given a TAGS vector containing 1 MTE tag per byte, pack the data as
   2 tags per byte and resize the vector.  */
extern void aarch64_mte_pack_tags (gdb::byte_vector &tags);

/* Given a TAGS vector containing 2 MTE tags per byte, unpack the data as
   1 tag per byte and resize the vector.  If SKIP_FIRST is TRUE, skip the
   first unpacked element.  Otherwise leave it in the unpacked vector.  */
extern void aarch64_mte_unpack_tags (gdb::byte_vector &tags, bool skip_first);

#endif /* GDB_ARCH_AARCH64_MTE_LINUX_H */
