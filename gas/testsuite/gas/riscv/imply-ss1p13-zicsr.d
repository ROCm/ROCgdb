#as: -march=rv64i_ss1p13
#readelf: -A
#source: imply-ss-zicsr.s
# I 2.0 includes Zicsr.  With I 2.1, Ss implies a separate Zicsr entry.
Attribute Section: riscv
File Attributes
  Tag_RISCV_arch: "rv64(i2p0|i2p1_zicsr2p0)_ss1p13"
