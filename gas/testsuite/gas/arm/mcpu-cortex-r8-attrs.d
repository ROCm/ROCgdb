#name: Cortex-R8 Attributes
#as: -mcpu=cortex-r8+nofp+nofp.dp
#source: nop-asm.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-R8"
  Tag_CPU_arch: v7
  Tag_CPU_arch_profile: Realtime
  Tag_ARM_ISA_use: Yes
  Tag_THUMB_ISA_use: Thumb-2
  Tag_DIV_use: Allowed in v7-A with integer division extension
