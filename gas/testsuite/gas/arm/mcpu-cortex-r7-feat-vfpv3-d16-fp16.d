#name: Cortex-R7 feat-vfpv3-d16-fp16
#as: -mcpu=cortex-r7
#source: feat-vfpv3-d16-fp16.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-R7"
  Tag_CPU_arch: v7
  Tag_CPU_arch_profile: Realtime
  Tag_ARM_ISA_use: Yes
  Tag_THUMB_ISA_use: Thumb-2
  Tag_FP_arch: VFPv3-D16
  Tag_FP_HP_extension: Allowed
  Tag_DIV_use: Allowed in v7-A with integer division extension
