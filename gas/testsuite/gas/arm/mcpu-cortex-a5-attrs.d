#name: Cortex-A5 Attributes
#as: -mcpu=cortex-a5+nofp+nosimd
#source: nop-asm.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-A5"
  Tag_CPU_arch: v7
  Tag_CPU_arch_profile: Application
  Tag_ARM_ISA_use: Yes
  Tag_THUMB_ISA_use: Thumb-2
  Tag_MPextension_use: Allowed
  Tag_Virtualization_use: TrustZone
