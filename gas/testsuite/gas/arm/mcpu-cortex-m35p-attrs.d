#name: Cortex-M35P Attributes
#as: -mcpu=cortex-m35p+nofp+nodsp
#source: nop-asm.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-M35P"
  Tag_CPU_arch: v8-M.mainline
  Tag_CPU_arch_profile: Microcontroller
  Tag_THUMB_ISA_use: Yes
