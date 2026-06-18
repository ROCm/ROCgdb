#name: Cortex-A32 Attributes 1
#as: -mcpu=cortex-a32+crypto
#source: nop-asm.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "Cortex-A32"
  Tag_CPU_arch: v8
  Tag_CPU_arch_profile: Application
  Tag_ARM_ISA_use: Yes
  Tag_THUMB_ISA_use: Thumb-2
  Tag_FP_arch: FP for ARMv8
  Tag_Advanced_SIMD_arch: NEON for ARMv8
  Tag_MPextension_use: Allowed
  Tag_Virtualization_use: TrustZone and Virtualization Extensions
