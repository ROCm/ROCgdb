/* THIS FILE IS GENERATED.  -*- buffer-read-only: t -*- vi:set ro:
  Original: 64bit-apx.xml */

#include "gdbsupport/tdesc.h"

static int
create_feature_i386_64bit_apx (struct target_desc *result, long regnum)
{
  struct tdesc_feature *feature;

  feature = tdesc_create_feature (result, "org.gnu.gdb.i386.apx");
  tdesc_create_reg (feature, "r16", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r17", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r18", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r19", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r20", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r21", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r22", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r23", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r24", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r25", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r26", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r27", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r28", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r29", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r30", regnum++, 1, NULL, 64, "int64");
  tdesc_create_reg (feature, "r31", regnum++, 1, NULL, 64, "int64");
  return regnum;
}
