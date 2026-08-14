/* SEH .pdata/.xdata COFF object file format on AArch64
   Copyright (C) 2026 Free Software Foundation, Inc.

   This file is part of GAS.

   GAS is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GAS is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GAS; see the file COPYING.  If not, write to the Free
   Software Foundation, 51 Franklin Street - Fifth Floor, Boston, MA
   02110-1301, USA.  */

/* SEH COFF AArch64 implementation partially intersects with the x64
   version, however it has a different extension to the unwind codes.
   It emits SEH data to pdata and xdata sections.  In some cases SEH
   data could be emitted to a packed record in the pdata section
   without the need for data in the xdata section.  However, the packed
   pdata record is not implemented yet.

   The current implementation does not include:
   - Packed .pdata record.
   - Support for AdvSIMD and SVE.
   - Epilogue start index different than 0.
   - Number of unwind codes and epilogue scopes limits are applied to the
     entire function, without splitting it into fragments.  */

#ifndef OBJ_COFF_SEH_AARCH64_H
#define OBJ_COFF_SEH_AARCH64_H

/* Unwind codes for AArch64 are based on
   "Microsoft ARM64 exception handling, unwind codes documentation".  */

typedef enum seh_aarch64_unwind_types
{
  unwind_alloc_s,
  unwind_alloc_m,
  unwind_alloc_l,
  unwind_save_reg,
  unwind_save_reg_x,
  unwind_save_regp,
  unwind_save_regp_x,
  unwind_save_fregp,
  unwind_save_fregp_x,
  unwind_save_freg,
  unwind_save_freg_x,
  unwind_save_lrpair,
  unwind_save_fplr,
  unwind_save_fplr_x,
  unwind_save_r19r20_x,
  unwind_add_fp,
  unwind_set_fp,
  unwind_save_next,
  unwind_nop,
  unwind_pac_sign_lr,
  unwind_end,
  unwind_end_c,
  unwind_last_type = unwind_end_c
} seh_aarch64_unwind_types;

#define SEH_CMDS							\
  /* Start a function that contains SEH.  */				\
  {"seh_proc", obj_coff_seh_proc, 0},					\
									\
  /* End a function that contains SEH.  */				\
  {"seh_endproc", obj_coff_seh_endproc, 0},				\
									\
  /* End a SEH prologue with unwinding codes.  */			\
  {"seh_endprologue", obj_coff_seh_endprologue, 0},			\
									\
  /* Allocate stack.  */						\
  {"seh_stackalloc", obj_coff_seh_stackalloc, 0},			\
									\
  /* Set a SEH handler.  */						\
  {"seh_handler", obj_coff_seh_handler, 0},				\
									\
  /* Set a SEH handler data.  */					\
  {"seh_handlerdata", obj_coff_seh_handlerdata, 0},			\
									\
  /* Switch back to the code section.  */				\
  {"seh_code", obj_coff_seh_code, 0},					\
									\
  /* Start a SEH epilogue.  */						\
  {"seh_startepilogue", obj_coff_seh_startepilogue, 0},			\
									\
  /* End a SEH epilogue.  */						\
  {"seh_endepilogue", obj_coff_seh_endepilogue, 0},			\
									\
  /* Save an 'x' register.  */						\
  {"seh_save_reg", obj_coff_seh_save_reg, unwind_save_reg},		\
									\
  /* Save an 'x' register with a pre-indexed offset.  */		\
  {"seh_save_reg_x", obj_coff_seh_save_reg, unwind_save_reg_x},		\
									\
  /* Save an 'x' register pair.  */					\
  {"seh_save_regp", obj_coff_seh_save_reg, unwind_save_regp},		\
									\
  /* Save an 'x' register pair with a pre-indexed offset.  */		\
  {"seh_save_regp_x", obj_coff_seh_save_reg, unwind_save_regp_x},	\
									\
  /* Save an 'x' register and lr.  */					\
  {"seh_save_lrpair", obj_coff_seh_save_reg, unwind_save_lrpair},	\
									\
  /* Save a 'd' register pair.  */					\
  {"seh_save_fregp", obj_coff_seh_save_reg, unwind_save_fregp},		\
									\
  /* Save a 'd' register pair with a pre-indexed offset.  */		\
  {"seh_save_fregp_x", obj_coff_seh_save_reg, unwind_save_fregp_x},	\
									\
  /* Save a 'd' register.  */						\
  {"seh_save_freg", obj_coff_seh_save_reg, unwind_save_freg},		\
									\
  /* Save a 'd' register with a pre-indexed offset.  */			\
  {"seh_save_freg_x", obj_coff_seh_save_reg, unwind_save_freg_x},	\
									\
  /* Save fp and lr registers.  */					\
  {"seh_save_fplr", obj_coff_seh_save_reg, unwind_save_fplr},		\
									\
  /* Save fp and lr registers with a pre-indexed offset.  */		\
  {"seh_save_fplr_x", obj_coff_seh_save_reg, unwind_save_fplr_x},	\
									\
  /* Save x19 and x20 registers with a pre-indexed offset.  */		\
  {"seh_save_r19r20_x", obj_coff_seh_save_reg, unwind_save_r19r20_x},	\
									\
  /* Set fp by sp + offset.  */						\
  {"seh_add_fp", obj_coff_seh_save_reg, unwind_add_fp},			\
									\
  /* Unwind operation is not required.  */				\
  {"seh_nop", obj_coff_seh_save_reg, unwind_nop},			\
									\
  /* Sign the return address in lr with pacibsp.  */			\
  {"seh_pac_sign_lr", obj_coff_seh_save_reg, unwind_pac_sign_lr},	\
									\
  /* Set fp by sp.  */							\
  {"seh_set_fp", obj_coff_seh_save_reg, unwind_set_fp},			\
									\
  /* Save next register pair.  */					\
  {"seh_save_next", obj_coff_seh_save_reg, unwind_save_next},

/* AArch64 exceptions handling and unwinding structures are based on
   "Microsoft ARM64 exception handling, pdata records documentation".  */

typedef struct seh_aarch64_unwind_code
{
  unsigned value;
  seh_aarch64_unwind_types type;
} seh_aarch64_unwind_code;

typedef struct seh_aarch64_epilogue_scope
{
  uint32_t epilogue_start_offset_reduced : 18;
  uint32_t reserved : 4;
  uint32_t epilogue_start_index : 10;
  bfd_vma epilogue_start_offset;
  bfd_vma epilogue_end_offset;
} seh_aarch64_epilogue_scope;

typedef struct seh_aarch64_func_fragment
{
  bfd_vma offset;
  symbolS *xdata_addr;
  struct seh_aarch64_func_fragment *next;
} seh_aarch64_func_fragment;

/* AARCH64_MAX_UNWIND_CODES is limited by
   seh_aarch64_xdata_header::ext_code_words.  */
#define AARCH64_MAX_UNWIND_CODES (255 * 4)
#define AARCH64_MAX_UNWIND_CODES_SIZE (255 * 4)
/* AARCH64_MAX_EPILOGUE_SCOPES is limited by
   seh_aarch64_xdata_header::ext_epilogue_count.  */
#define AARCH64_MAX_EPILOGUE_SCOPES 65535

typedef struct seh_aarch64_context
{
  struct seh_aarch64_context *next;

  /* Initial code-segment.  */
  segT code_seg;
  /* Function name.  */
  char *func_name;
  /* BeginAddress.  */
  symbolS *start_addr;
  /* EndAddress.  */
  symbolS *end_addr;
  /* PrologueEnd.  */
  symbolS *endprologue_addr;

  symbolS *handler_data_xdata_addr;
  /* ExceptionHandler.  */
  expressionS handler;
  /* ExceptionHandlerData.  */
  expressionS handler_data;

  subsegT subsection;

  unsigned unwind_codes_count;
  unsigned unwind_codes_byte_count;
  seh_aarch64_unwind_code unwind_codes[AARCH64_MAX_UNWIND_CODES];
  unsigned epilogue_scopes_count;
  unsigned epilogue_scopes_capacity;
  seh_aarch64_epilogue_scope *epilogue_scopes;
  bool has_exception_data;
  /* The function fragments.  */
  seh_aarch64_func_fragment func_fragment;
} seh_context;

#endif /* OBJ_COFF_SEH_AARCH64_H.  */
