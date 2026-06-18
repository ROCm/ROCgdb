#name: Cortex-M85 Attributes
#as: -mcpu=cortex-m85+nopacbti+nomve.fp+nomve+nofp+nodsp
#source: nop-asm.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-M85"
  Tag_CPU_arch: v8.1-M.mainline
  Tag_CPU_arch_profile: Microcontroller
  Tag_THUMB_ISA_use: Yes
  Tag_DSP_extension: Allowed
