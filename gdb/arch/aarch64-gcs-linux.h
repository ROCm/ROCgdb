/* Common Linux target-dependent definitions for AArch64 GCS

   Copyright (C) 2025-2026 Free Software Foundation, Inc.

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

#ifndef GDB_ARCH_AARCH64_GCS_LINUX_H
#define GDB_ARCH_AARCH64_GCS_LINUX_H

/* Feature check for Guarded Control Stack.  */
#define AARCH64_HWCAP_GCS (1ULL << 32)

#define AARCH64_SEGV_CPERR 10 /* Control protection error.  */

/* Flag which enables shadow stack in PR_SET_SHADOW_STACK_STATUS prctl.  */
#define AARCH64_PR_SHADOW_STACK_ENABLE (1UL << 0)
#define AARCH64_PR_SHADOW_STACK_WRITE (1UL << 1)
#define AARCH64_PR_SHADOW_STACK_PUSH (1UL << 2)

/* The GCS regset consists of 3 64-bit registers.  */
#define AARCH64_LINUX_SIZEOF_GCS_REGSET (3 * 8)

#endif /* GDB_ARCH_AARCH64_GCS_LINUX_H */
