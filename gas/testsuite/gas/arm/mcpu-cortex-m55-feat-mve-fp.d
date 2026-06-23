#name: Cortex-M55 feat-mve.fp
#as: -mcpu=cortex-m55
#source: feat-mve.fp.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-M55"
  Tag_CPU_arch: v8.1-M.mainline
  Tag_CPU_arch_profile: Microcontroller
  Tag_THUMB_ISA_use: Yes
  Tag_FP_arch: FPv5/FP-D16 for ARMv8
  Tag_DSP_extension: Allowed
  Tag_MVE_arch: MVE Integer and FP
