/* THIS FILE IS GENERATED.  -*- buffer-read-only: t -*- vi:set ro:
  Original: aarch64-poe.xml */

#include "gdbsupport/tdesc.h"

static int
create_feature_aarch64_poe (target_desc *result, long regnum)
{
  tdesc_feature *feature;

  feature = tdesc_create_feature (result, "org.gnu.gdb.aarch64.poe");
  tdesc_create_reg (feature, "por_el0", regnum++, 1, "por", 64, "uint64");
  return regnum;
}
