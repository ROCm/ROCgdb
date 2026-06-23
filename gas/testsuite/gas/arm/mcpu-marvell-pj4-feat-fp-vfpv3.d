#name: Marvell-PJ4 feat-fp-vfpv3
#as: -mcpu=marvell-pj4
#source: feat-fp-vfpv3.s
#readelf: -A
# This test is only valid on EABI based ports.
# target: *-*-*eabi*

Attribute Section: aeabi
File Attributes
  Tag_CPU_name: "MARVELL-PJ4"
  Tag_CPU_arch: v7
  Tag_CPU_arch_profile: Application
  Tag_ARM_ISA_use: Yes
  Tag_THUMB_ISA_use: Thumb-2
  Tag_FP_arch: VFPv3
  Tag_MPextension_use: Allowed
  Tag_Virtualization_use: TrustZone
