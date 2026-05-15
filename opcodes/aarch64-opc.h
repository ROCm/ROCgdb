/* aarch64-opc.h -- Header file for aarch64-opc.c and aarch64-opc-2.c.
   Copyright (C) 2012-2026 Free Software Foundation, Inc.
   Contributed by ARM Ltd.

   This file is part of the GNU opcodes library.

   This library is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   It is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
   License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; see the file COPYING3. If not,
   see <http://www.gnu.org/licenses/>.  */

#ifndef OPCODES_AARCH64_OPC_H
#define OPCODES_AARCH64_OPC_H

#include <string.h>
#include "opcode/aarch64.h"

/* Field description.

   If is_const is false, this identifies a bitfield in an instruction encoding
   that has size WIDTH and has its least significant bit at position NUM.

   If is_const is true, this represents the constant bit string of size WIDTH
   bits stored in the least significant bits of NUM.  In this case, the
   leading 8-WIDTH bits of VALUE must be zero.

   A sequence of fields can be used to describe how instruction operands are
   represented in the 32-bit instruction encoding.

   For example, consider an instruction operand Zd that is an even numbered
   register in z16-z30, with the middle three bits of the register number
   stored in bits [19:17] of the encoding.  The register number can then be
   constructed by concatenating:
   - a constant bit '1' (represented here as {1, 1, true}),
   - bits [19:17] of the encoding (represented here as {3, 17, false}), and
   - a constant bit '0' (represented here as {1, 0, true}).
   This sequence of fields fully describes both the constraints on which
   register numbers are valid, and how valid register numbers are represented
   in the instruction encoding.  */
struct aarch64_field
{
  unsigned int width:8;
  unsigned int num:7;
  bool is_const:1;
};

typedef struct aarch64_field aarch64_field;

#define AARCH64_FIELD(lsb, width) ((aarch64_field) {width, lsb, false})
#define AARCH64_FIELD_CONST(val, width) ((aarch64_field) {width, val, true})
#define AARCH64_FIELD_NIL ((aarch64_field) {0, 0, false})

#define FLD_CONST_0 AARCH64_FIELD_CONST (0, 1)
#define FLD_CONST_00 AARCH64_FIELD_CONST (0, 2)
#define FLD_CONST_01 AARCH64_FIELD_CONST (1, 2)
#define FLD_CONST_1 AARCH64_FIELD_CONST (1, 1)

/* Instruction fields.  These defines are included to reduce the initial diff
   size, but the indirection should eventually be eliminated.  */
#define FLD_CRm                AARCH64_FIELD( 8,  4)
#define FLD_CRm_dsb_nxs        AARCH64_FIELD(10,  2)
#define FLD_CRn                AARCH64_FIELD(12,  4)
#define FLD_H                  AARCH64_FIELD(11,  1)
#define FLD_L                  AARCH64_FIELD(21,  1)
#define FLD_M                  AARCH64_FIELD(20,  1)
#define FLD_N                  AARCH64_FIELD(22,  1)
#define FLD_Q                  AARCH64_FIELD(30,  1)
#define FLD_Rm                 AARCH64_FIELD(16,  5)
#define FLD_Rn                 AARCH64_FIELD( 5,  5)
#define FLD_Rt                 AARCH64_FIELD( 0,  5)
#define FLD_S                  AARCH64_FIELD(12,  1)
#define FLD_SM3_imm2           AARCH64_FIELD(12,  2)
#define FLD_SME_Q              AARCH64_FIELD(16,  1)
#define FLD_SME_size_12        AARCH64_FIELD(12,  2)
#define FLD_SME_size_22        AARCH64_FIELD(22,  2)
#define FLD_SME_sz_23          AARCH64_FIELD(23,  1)
#define FLD_SME_tszh           AARCH64_FIELD(22,  1)
#define FLD_SME_tszl           AARCH64_FIELD(18,  3)
#define FLD_SVE_M_4            AARCH64_FIELD( 4,  1)
#define FLD_SVE_M_14           AARCH64_FIELD(14,  1)
#define FLD_SVE_M_16           AARCH64_FIELD(16,  1)
#define FLD_SVE_Pd             AARCH64_FIELD( 0,  4)
#define FLD_SVE_Pg4_10         AARCH64_FIELD(10,  4)
#define FLD_SVE_Pm             AARCH64_FIELD(16,  4)
#define FLD_SVE_Pn             AARCH64_FIELD( 5,  4)
#define FLD_SVE_Zd             AARCH64_FIELD( 0,  5)
#define FLD_SVE_Zm_16          AARCH64_FIELD(16,  5)
#define FLD_SVE_Zn             AARCH64_FIELD( 5,  5)
#define FLD_SVE_imm4           AARCH64_FIELD(16,  4)
#define FLD_SVE_imm6           AARCH64_FIELD(16,  6)
#define FLD_SVE_msz            AARCH64_FIELD(10,  2)
#define FLD_SVE_size           AARCH64_FIELD(17,  2)
#define FLD_SVE_sz             AARCH64_FIELD(22,  1)
#define FLD_SVE_sz2            AARCH64_FIELD(30,  1)
#define FLD_SVE_sz3            AARCH64_FIELD(17,  1)
#define FLD_SVE_sz4            AARCH64_FIELD(14,  1)
#define FLD_SVE_tszh           AARCH64_FIELD(22,  2)
#define FLD_SVE_tszl_8         AARCH64_FIELD( 8,  2)
#define FLD_SVE_tszl_19        AARCH64_FIELD(19,  2)
#define FLD_abc                AARCH64_FIELD(16,  3)
#define FLD_asisdlso_opcode    AARCH64_FIELD(13,  3)
#define FLD_cmode              AARCH64_FIELD(12,  4)
#define FLD_cond               AARCH64_FIELD(12,  4)
#define FLD_cond2              AARCH64_FIELD( 0,  4)
#define FLD_defgh              AARCH64_FIELD( 5,  5)
#define FLD_hw                 AARCH64_FIELD(21,  2)
#define FLD_imm1_22            AARCH64_FIELD(22,  1)
#define FLD_imm3_5             AARCH64_FIELD( 5,  3)
#define FLD_imm3_10            AARCH64_FIELD(10,  3)
#define FLD_imm3_19            AARCH64_FIELD(19,  3)
#define FLD_imm4_5             AARCH64_FIELD( 5,  4)
#define FLD_imm4_11            AARCH64_FIELD(11,  4)
#define FLD_imm5               AARCH64_FIELD(16,  5)
#define FLD_imm6_10            AARCH64_FIELD(10,  6)
#define FLD_imm12              AARCH64_FIELD(10, 12)
#define FLD_immb               AARCH64_FIELD(16,  3)
#define FLD_immh               AARCH64_FIELD(19,  4)
#define FLD_ldst_size          AARCH64_FIELD(30,  2)
#define FLD_len                AARCH64_FIELD(13,  2)
#define FLD_lse_sz             AARCH64_FIELD(30,  1)
#define FLD_op0                AARCH64_FIELD(19,  2)
#define FLD_op1                AARCH64_FIELD(16,  3)
#define FLD_op2                AARCH64_FIELD( 5,  3)
#define FLD_opc                AARCH64_FIELD(22,  2)
#define FLD_opc1               AARCH64_FIELD(23,  1)
#define FLD_opcode             AARCH64_FIELD(12,  4)
#define FLD_option             AARCH64_FIELD(13,  3)
#define FLD_scale              AARCH64_FIELD(10,  6)
#define FLD_sf                 AARCH64_FIELD(31,  1)
#define FLD_shift              AARCH64_FIELD(22,  2)
#define FLD_size               AARCH64_FIELD(22,  2)
#define FLD_sz                 AARCH64_FIELD(22,  1)
#define FLD_type               AARCH64_FIELD(22,  2)
#define FLD_vldst_size         AARCH64_FIELD(10,  2)
#define FLD_off2               AARCH64_FIELD( 5,  2)
#define FLD_ol                 AARCH64_FIELD( 5,  1)
#define FLD_opc2               AARCH64_FIELD(12,  4)
#define FLD_rcpc3_size         AARCH64_FIELD(30,  2)
#define FLD_ZA8_1              AARCH64_FIELD( 8,  1)
#define FLD_ZA7_2              AARCH64_FIELD( 7,  2)
#define FLD_ZA6_3              AARCH64_FIELD( 6,  3)
#define FLD_ZA5_4              AARCH64_FIELD( 5,  4)


/* Operand description.  */

struct aarch64_operand
{
  enum aarch64_operand_class op_class;

  /* Name of the operand code; used mainly for the purpose of internal
     debugging.  */
  const char *name;

  unsigned int flags;

  /* The associated instruction bit-fields; no operand has more than 5
     bit-fields */
  aarch64_field fields[6];

  /* Brief description */
  const char *desc;
};

typedef struct aarch64_operand aarch64_operand;

extern const aarch64_operand aarch64_operands[];

enum err_type
verify_constraints (const struct aarch64_inst *, const aarch64_insn, bfd_vma,
		    bool, aarch64_operand_error *, aarch64_instr_sequence*);

/* Operand flags.  */

#define OPD_F_HAS_INSERTER	0x00000001
#define OPD_F_HAS_EXTRACTOR	0x00000002
#define OPD_F_SEXT		0x00000004	/* Require sign-extension.  */
#define OPD_F_SHIFT_BY_2	0x00000008	/* Need to left shift the field
						   value by 2 to get the value
						   of an immediate operand.  */
#define OPD_F_MAYBE_SP		0x00000010	/* May potentially be SP.  */
#define OPD_F_OD_MASK		0x000001e0	/* Operand-dependent data.  */
#define OPD_F_OD_LSB		5
#define OPD_F_NO_ZR		0x00000200	/* ZR index not allowed.  */
#define OPD_F_SHIFT_BY_3	0x00000400	/* Need to left shift the field
						   value by 3 to get the value
						   of an immediate operand.  */
#define OPD_F_SHIFT_BY_4	0x00000800	/* Need to left shift the field
						   value by 4 to get the value
						   of an immediate operand.  */
#define OPD_F_UNSIGNED		0x00001000	/* Expect an unsigned value.  */


/* Register flags.  */

#undef F_DEPRECATED
#define F_DEPRECATED	(1 << 0)  /* Deprecated system register.  */

/*			(1 << 1)     Unused.  */

#undef F_HASXT
#define F_HASXT		(1 << 2)  /* System instruction register <Xt>
				     operand.  */

#undef F_REG_READ
#define F_REG_READ	(1 << 3)  /* Register can only be used to read values
				     out of.  */

#undef F_REG_WRITE
#define F_REG_WRITE	(1 << 4)  /* Register can only be written to but not
				     read from.  */

#undef F_REG_IN_CRM
#define F_REG_IN_CRM	(1 << 5)  /* Register extra encoding in CRm.  */

#undef F_REG_ALIAS
#define F_REG_ALIAS	(1 << 6)  /* Register name aliases another.  */

#undef F_REG_128
#define F_REG_128	(1 << 7) /* System register implementable as 128-bit wide.  */

#undef F_TLBID_XT
#define F_TLBID_XT	(1 << 8)  /* System instruction register <Xt> as optional operand.  */


/* PSTATE field name for the MSR instruction this is encoded in "op1:op2:CRm".
   Part of CRm can be used to encode <pstatefield>. E.g. CRm[3:1] for SME.
   In order to set/get full PSTATE field name use flag F_REG_IN_CRM and below
   macros to encode and decode CRm encoding.
*/
#define PSTATE_ENCODE_CRM(val) (val << 6)
#define PSTATE_DECODE_CRM(flags) ((flags >> 6) & 0x0f)

#undef F_IMM_IN_CRM
#define F_IMM_IN_CRM	(1 << 10)  /* Immediate extra encoding in CRm.  */

/* Also CRm may contain, in addition to <pstatefield> immediate.
   E.g. CRm[0] <imm1> at bit 0 for SME. Use below macros to encode and decode
   immediate mask.
*/
#define PSTATE_ENCODE_CRM_IMM(mask) (mask << 11)
#define PSTATE_DECODE_CRM_IMM(mask) ((mask >> 11) & 0x0f)

/* Helper macro to ENCODE CRm and its immediate.  */
#define PSTATE_ENCODE_CRM_AND_IMM(CVAL,IMASK) \
        (F_REG_IN_CRM | PSTATE_ENCODE_CRM(CVAL) \
         | F_IMM_IN_CRM | PSTATE_ENCODE_CRM_IMM(IMASK))

/* Bits [15, 18] contain the maximum value for an immediate MSR.  */
#define F_REG_MAX_VALUE(X) ((X) << 15)
#define F_GET_REG_MAX_VALUE(X) (((X) >> 15) & 0x0f)

static inline bool
operand_has_inserter (const aarch64_operand *operand)
{
  return (operand->flags & OPD_F_HAS_INSERTER) != 0;
}

static inline bool
operand_has_extractor (const aarch64_operand *operand)
{
  return (operand->flags & OPD_F_HAS_EXTRACTOR) != 0;
}

static inline bool
operand_need_sign_extension (const aarch64_operand *operand)
{
  return (operand->flags & OPD_F_SEXT) != 0;
}

static inline bool
operand_need_shift_by_two (const aarch64_operand *operand)
{
  return (operand->flags & OPD_F_SHIFT_BY_2) != 0;
}

static inline bool
operand_need_shift_by_three (const aarch64_operand *operand)
{
  return (operand->flags & OPD_F_SHIFT_BY_3) != 0;
}

static inline bool
operand_need_shift_by_four (const aarch64_operand *operand)
{
  return (operand->flags & OPD_F_SHIFT_BY_4) != 0;
}

static inline bool
operand_need_unsigned_offset (const aarch64_operand *operand)
{
  return (operand->flags & OPD_F_UNSIGNED) != 0;
}

static inline bool
operand_maybe_stack_pointer (const aarch64_operand *operand)
{
  return (operand->flags & OPD_F_MAYBE_SP) != 0;
}

/* Return the value of the operand-specific data field (OPD_F_OD_MASK).  */
static inline unsigned int
get_operand_specific_data (const aarch64_operand *operand)
{
  return (operand->flags & OPD_F_OD_MASK) >> OPD_F_OD_LSB;
}

/* Return the width of field number N of operand *OPERAND.  */
static inline unsigned
get_operand_field_width (const aarch64_operand *operand, unsigned n)
{
  assert (operand->fields[n].width != 0);
  return operand->fields[n].width;
}

/* Return the total width of the operand *OPERAND.  */
static inline unsigned
get_operand_fields_width (const aarch64_operand *operand)
{
  int i = 0;
  unsigned width = 0;
  while (operand->fields[i].width != 0)
    width += operand->fields[i++].width;
  assert (width > 0 && width < 32);
  return width;
}

static inline const aarch64_operand *
get_operand_from_code (enum aarch64_opnd code)
{
  return aarch64_operands + code;
}

/* Operand qualifier and operand constraint checking.  */

bool aarch64_match_operands_constraint (aarch64_inst *,
				       aarch64_operand_error *);

/* Operand qualifier related functions.  */
const char* aarch64_get_qualifier_name (aarch64_opnd_qualifier_t);
unsigned char aarch64_get_qualifier_nelem (aarch64_opnd_qualifier_t);
aarch64_insn aarch64_get_qualifier_standard_value (aarch64_opnd_qualifier_t);
int aarch64_find_best_match (const aarch64_inst *,
			     const aarch64_opnd_qualifier_seq_t *,
			     int, aarch64_opnd_qualifier_t *, int *);

static inline void
reset_operand_qualifier (aarch64_inst *inst, int idx)
{
  assert (idx >=0 && idx < aarch64_num_of_operands (inst->opcode));
  inst->operands[idx].qualifier = AARCH64_OPND_QLF_UNKNOWN;
}

/* Inline functions operating on instruction bit-field(s).  */

/* Generate a mask that has WIDTH number of consecutive 1s.  */

static inline aarch64_insn
gen_mask (int width)
{
  return ((aarch64_insn) 1 << width) - 1;
}

/* LSB_REL is the relative location of the lsb in the sub field, starting from 0.  */
static inline int
gen_sub_field (aarch64_field field, int lsb_rel, int width, aarch64_field *ret)
{
  if (lsb_rel < 0 || width <= 0 || lsb_rel + width > field.width)
    return 0;
  ret->num = field.num + lsb_rel;
  ret->width = width;
  return 1;
}

/* Insert VALUE into FIELD of CODE.  MASK can be zero or the base mask
   of the opcode.  */

static inline void
insert_field (aarch64_field field, aarch64_insn *code,
	      aarch64_insn value, aarch64_insn mask)
{
  assert (field.width < 32 && field.width >= 1
	  && (field.is_const ? (field.num < 1 << field.width)
			      : (field.num + field.width <= 32)));
  value &= gen_mask (field.width);
  if (field.is_const)
    {
      assert (value == field.num);
      return;
    }
  value <<= field.num;
  /* In some opcodes, field can be part of the base opcode, e.g. the size
     field in FADD.  The following helps avoid corrupt the base opcode.  */
  value &= ~mask;
  *code |= value;
}

/* Extract FIELD of CODE and return the value.  MASK can be zero or the base
   mask of the opcode.  */

static inline aarch64_insn
extract_field (aarch64_field field, aarch64_insn code,
	       aarch64_insn mask)
{
  aarch64_insn value;
  /* Check for constant field.  */
  if (field.is_const)
    return field.num;

  /* Clear any bit that is a part of the base opcode.  */
  code &= ~mask;
  value = (code >> field.num) & gen_mask (field.width);
  return value;
}

extern aarch64_insn
extract_fields (aarch64_insn code, aarch64_insn mask, ...);

/* Inline functions selecting operand to do the encoding/decoding for a
   certain instruction bit-field.  */

/* Select the operand to do the encoding/decoding of the 'sf' field.
   The heuristic-based rule is that the result operand is respected more.  */

static inline int
select_operand_for_sf_field_coding (const aarch64_opcode *opcode)
{
  int idx = -1;
  if (opcode->iclass == fprcvtfloat2int)
    return 0;
  else if (opcode->iclass == fprcvtint2float)
    return 1;

  if (aarch64_get_operand_class (opcode->operands[0])
      == AARCH64_OPND_CLASS_INT_REG)
    /* normal case.  */
    idx = 0;
  else if (aarch64_get_operand_class (opcode->operands[1])
	   == AARCH64_OPND_CLASS_INT_REG)
    /* e.g. float2fix.  */
    idx = 1;
  else
    { assert (0); abort (); }
  return idx;
}

/* Select the operand to do the encoding/decoding of the 'type' field in
   the floating-point instructions.
   The heuristic-based rule is that the source operand is respected more.  */

static inline int
select_operand_for_fptype_field_coding (const aarch64_opcode *opcode)
{
  int idx;
  if (opcode->iclass == fprcvtfloat2int)
    return 1;
  else if (opcode->iclass == fprcvtint2float)
    return 0;

  if (aarch64_get_operand_class (opcode->operands[1])
      == AARCH64_OPND_CLASS_FP_REG)
    /* normal case.  */
    idx = 1;
  else if (aarch64_get_operand_class (opcode->operands[0])
	   == AARCH64_OPND_CLASS_FP_REG)
    /* e.g. float2fix.  */
    idx = 0;
  else
    { assert (0); abort (); }
  return idx;
}

/* Select the operand to do the encoding/decoding of the 'size' field in
   the AdvSIMD scalar instructions.
   The heuristic-based rule is that the destination operand is respected
   more.  */

static inline int
select_operand_for_scalar_size_field_coding (const aarch64_opcode *opcode)
{
  int src_size = 0, dst_size = 0;
  if (aarch64_get_operand_class (opcode->operands[0])
      == AARCH64_OPND_CLASS_SISD_REG)
    dst_size = aarch64_get_qualifier_esize (opcode->qualifiers_list[0][0]);
  if (aarch64_get_operand_class (opcode->operands[1])
      == AARCH64_OPND_CLASS_SISD_REG)
    src_size = aarch64_get_qualifier_esize (opcode->qualifiers_list[0][1]);
  if (src_size == dst_size && src_size == 0)
    { assert (0); abort (); }
  /* When the result is not a sisd register or it is a long operation.  */
  if (dst_size == 0 || dst_size == src_size << 1)
    return 1;
  else
    return 0;
}

/* Select the operand to do the encoding/decoding of the 'size:Q' fields in
   the AdvSIMD instructions.  */

int aarch64_select_operand_for_sizeq_field_coding (const aarch64_opcode *);

/* Miscellaneous.  */

aarch64_insn aarch64_get_operand_modifier_value (enum aarch64_modifier_kind);
enum aarch64_modifier_kind
aarch64_get_operand_modifier_from_value (aarch64_insn, bool);


bool aarch64_wide_constant_p (uint64_t, int, unsigned int *);
bool aarch64_logical_immediate_p (uint64_t, int, aarch64_insn *);
int aarch64_shrink_expanded_imm8 (uint64_t);

/* Copy the content of INST->OPERANDS[SRC] to INST->OPERANDS[DST].  */
static inline void
copy_operand_info (aarch64_inst *inst, int dst, int src)
{
  assert (dst >= 0 && src >= 0 && dst < AARCH64_MAX_OPND_NUM
	  && src < AARCH64_MAX_OPND_NUM);
  memcpy (&inst->operands[dst], &inst->operands[src],
	  sizeof (aarch64_opnd_info));
  inst->operands[dst].idx = dst;
}

/* A primitive log caculator.  */

static inline unsigned int
get_logsz (unsigned int size)
{
  const unsigned char ls[16] =
    {0, 1, -1, 2, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, 4};
  if (size > 16)
    {
      assert (0);
      return -1;
    }
  assert (ls[size - 1] != (unsigned char)-1);
  return ls[size - 1];
}

#endif /* OPCODES_AARCH64_OPC_H */
