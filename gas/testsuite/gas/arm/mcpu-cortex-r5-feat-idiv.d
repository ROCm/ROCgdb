#name: Cortex-R5 feat-idiv
#as: -mcpu=cortex-r5
#source: feat-idiv.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-R5"
  Tag_CPU_arch: v7
  Tag_CPU_arch_profile: Realtime
  Tag_ARM_ISA_use: Yes
  Tag_THUMB_ISA_use: Thumb-2
  Tag_FP_arch: VFPv3
  Tag_DIV_use: Allowed in v7-A with integer division extension
