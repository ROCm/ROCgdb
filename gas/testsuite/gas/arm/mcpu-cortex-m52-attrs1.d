#name: Cortex-M52 Attributes 1
#as: -mcpu=cortex-m52+cdecp0+cdecp1+cdecp2+cdecp3+cdecp4+cdecp5+cdecp6+cdecp7
#source: nop-asm.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-M52"
  Tag_CPU_arch: v8.1-M.mainline
  Tag_CPU_arch_profile: Microcontroller
  Tag_THUMB_ISA_use: Yes
  Tag_FP_arch: FPv5/FP-D16 for ARMv8
  Tag_DSP_extension: Allowed
  Tag_MVE_arch: MVE Integer and FP
