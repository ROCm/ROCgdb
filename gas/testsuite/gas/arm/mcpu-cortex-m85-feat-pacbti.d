#name: Cortex-M85 feat-pacbti
#as: -mcpu=cortex-m85
#source: feat-pacbti.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-M85"
  Tag_CPU_arch: v8.1-M.mainline
  Tag_CPU_arch_profile: Microcontroller
  Tag_THUMB_ISA_use: Yes
  Tag_FP_arch: FPv5/FP-D16 for ARMv8
  Tag_DSP_extension: Allowed
  Tag_MVE_arch: MVE Integer and FP
