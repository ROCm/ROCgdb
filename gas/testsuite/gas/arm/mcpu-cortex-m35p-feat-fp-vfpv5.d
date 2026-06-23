#name: Cortex-M35P feat-fp-vfpv5
#as: -mcpu=cortex-m35p
#source: feat-fp-vfpv5.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-M35P"
  Tag_CPU_arch: v8-M.mainline
  Tag_CPU_arch_profile: Microcontroller
  Tag_THUMB_ISA_use: Yes
  Tag_FP_arch: VFPv4-D16
  Tag_DSP_extension: Allowed
