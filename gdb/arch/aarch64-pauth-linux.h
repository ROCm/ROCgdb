/* Common Linux target-dependent definitions for AArch64 PAuth.

   Copyright (C) 2019-2026 Free Software Foundation, Inc.

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

#ifndef GDB_ARCH_AARCH64_PAUTH_LINUX_H
#define GDB_ARCH_AARCH64_PAUTH_LINUX_H

/* Feature check for Pointer Authentication Code Extension.  */
#define AARCH64_HWCAP_PACA (1 << 30)

/* The pauth regset consists of 2 64-bit registers.  */
#define AARCH64_LINUX_SIZEOF_PAUTH 16

#endif /* GDB_ARCH_AARCH64_PAUTH_LINUX_H */
