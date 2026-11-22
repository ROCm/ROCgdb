#as: -march=rva20s64
#readelf: -A
#source: empty.s
# I 2.0 includes Zicsr and Zifencei.  With I 2.1, both have ELF entries.
Attribute Section: riscv
File Attributes
  Tag_RISCV_arch: "rv64(i2p0_m2p0_a2p0_f2p0_d2p0_c2p0_ziccamoa1p0_ziccif1p0_zicclsm1p0_ziccrse1p0_zicntr2p0|i2p1_m2p0_a2p[01]_f2p2_d2p2_c2p0_ziccamoa1p0_ziccif1p0_zicclsm1p0_ziccrse1p0_zicntr2p0_zicsr2p0_zifencei2p0)_zmmul1p0_za128rs1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0_ss1p11_ssccptr1p0_sstvala1p0_sstvecd1p0_svade1p0_svbare1p0"
