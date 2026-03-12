/* CodeView register definitions for AMD64.

   Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 1994-2026 Advanced Micro Devices, Inc. All rights reserved.

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

/* CodeView AMD64 register IDs from Microsoft's cvconst.h.
   Each architecture has its own CV register numbering; see
   pdb-cv-regs-<arch>.h for other targets.  */

#ifndef GDB_PDB_PDB_CV_REGS_AMD64_H
#define GDB_PDB_PDB_CV_REGS_AMD64_H

/* General-purpose registers.  */
#define CV_AMD64_RAX  328
#define CV_AMD64_RBX  329
#define CV_AMD64_RCX  330
#define CV_AMD64_RDX  331
#define CV_AMD64_RSI  332
#define CV_AMD64_RDI  333
#define CV_AMD64_RBP  334
#define CV_AMD64_RSP  335
#define CV_AMD64_R8   336
#define CV_AMD64_R9   337
#define CV_AMD64_R10  338
#define CV_AMD64_R11  339
#define CV_AMD64_R12  340
#define CV_AMD64_R13  341
#define CV_AMD64_R14  342
#define CV_AMD64_R15  343

/* Map a CodeView AMD64 register ID to a DWARF register number.
   DWARF AMD64 numbering: RAX=0 RDX=1 RCX=2 RBX=3 RSI=4 RDI=5
   RBP=6 RSP=7 R8..R15=8..15.
   Returns -1 for unrecognized registers.  */
static inline int
cv_amd64_reg_to_dwarf (uint16_t cv_reg)
{
  switch (cv_reg)
    {
    case CV_AMD64_RAX: return 0;
    case CV_AMD64_RDX: return 1;
    case CV_AMD64_RCX: return 2;
    case CV_AMD64_RBX: return 3;
    case CV_AMD64_RSI: return 4;
    case CV_AMD64_RDI: return 5;
    case CV_AMD64_RBP: return 6;
    case CV_AMD64_RSP: return 7;
    case CV_AMD64_R8:  return 8;
    case CV_AMD64_R9:  return 9;
    case CV_AMD64_R10: return 10;
    case CV_AMD64_R11: return 11;
    case CV_AMD64_R12: return 12;
    case CV_AMD64_R13: return 13;
    case CV_AMD64_R14: return 14;
    case CV_AMD64_R15: return 15;
    default:           return -1;
    }
}

#endif /* GDB_PDB_PDB_CV_REGS_AMD64_H */
