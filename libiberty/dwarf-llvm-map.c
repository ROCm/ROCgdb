/* Mapping between DWARF LLVM tags and LLVM USER tags.
   Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

This file is part of GNU CC.

GNU CC is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

In addition to the permissions in the GNU General Public License, the
Free Software Foundation gives you unlimited permission to link the
compiled version of this file into combinations with other programs,
and to distribute those combinations without any restriction coming
from the use of this file.  (The General Public License restrictions
do apply in other respects; for example, they cover modification of
the file, and distribution when not linked into a combined
executable.)

GNU CC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU CC; see the file COPYING.  If not, write to
the Free Software Foundation, 51 Franklin Street - Fifth Floor,
Boston, MA 02110-1301, USA.  */

#include "dwarf2.h"

#define DW_FIRST_TAG(name, value)
#define DW_END_TAG
#define DW_FIRST_FORM(name, value)
#define DW_END_FORM
#define DW_FIRST_AT(name, value)
#define DW_END_AT
#define DW_FIRST_OP(name, value)
#define DW_END_OP
#define DW_BEGIN_OP_LLVM_USER(name, value)
#define DW_END_OP_LLVM_USER
#define DW_BEGIN_OP_TO_LLVM_USER(name, value) \
  unsigned int get_DW_OP_LLVM_USER (unsigned int op) {	\
  switch (op) { \
  DW_OP_TO_LLVM_USER (name, value)
#define DW_END_OP_TO_LLVM_USER } return 0; }
#define DW_FIRST_ATE(name, value)
#define DW_END_ATE
#define DW_FIRST_CFA(name, value)
#define DW_END_CFA
#define DW_FIRST_IDX(name, value)
#define DW_END_IDX
#define DW_FIRST_UT(name, value)
#define DW_END_UT

#define DW_TAG(name, value)
#define DW_TAG_DUP(name, value)
#define DW_FORM(name, value)
#define DW_AT(name, value)
#define DW_AT_DUP(name, value)
#define DW_OP(name, value)
#define DW_OP_DUP(name, value)
#define DW_OP_LLVM_USER(name, value)
#define DW_OP_TO_LLVM_USER(name, value) case DW_OP_##value: return DW_OP_LLVM_USER_##name ;
#define DW_ATE(name, value)
#define DW_ATE_DUP(name, value)
#define DW_CFA(name, value)
#define DW_CFA_DUP(name, value)
#define DW_IDX(name, value)
#define DW_IDX_DUP(name, value)
#define DW_UT(name, value)

#include "dwarf2.def"

#undef DW_FIRST_TAG
#undef DW_END_TAG
#undef DW_FIRST_FORM
#undef DW_END_FORM
#undef DW_FIRST_AT
#undef DW_END_AT
#undef DW_FIRST_OP
#undef DW_END_OP
#undef DW_BEGIN_OP_LLVM_USER
#undef DW_END_OP_LLVM_USER
#undef DW_BEGIN_OP_TO_LLVM_USER
#undef DW_END_OP_TO_LLVM_USER
#undef DW_FIRST_ATE
#undef DW_END_ATE
#undef DW_FIRST_CFA
#undef DW_END_CFA
#undef DW_FIRST_IDX
#undef DW_END_IDX
#undef DW_END_UT

#undef DW_TAG
#undef DW_TAG_DUP
#undef DW_FORM
#undef DW_AT
#undef DW_AT_DUP
#undef DW_OP
#undef DW_OP_DUP
#undef DW_OP_LLVM_USER
#undef DW_OP_TO_LLVM_USER
#undef DW_ATE
#undef DW_ATE_DUP
#undef DW_CFA
#undef DW_CFA_DUP
#undef DW_IDX
#undef DW_IDX_DUP
