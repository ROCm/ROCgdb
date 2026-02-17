/* Common Linux target-dependent definitions for AArch64 FPMR.

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

#ifndef GDB_ARCH_AARCH64_FPMR_LINUX_H
#define GDB_ARCH_AARCH64_FPMR_LINUX_H

/* Feature check for Floating Point Mode Register.  */
#define AARCH64_HWCAP2_FPMR (1ULL << 48)

#endif /* GDB_ARCH_AARCH64_FPMR_LINUX_H */
