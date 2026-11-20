#source: table-jump-empty-mixed.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian
#ld: -melf64lriscv --relax-tbljal -Ttable-jump-empty-flags.ld
#readelf: -WSl

#...
 +\[ *[0-9]+\] \.riscv\.jvt +PROGBITS +0+3000 +[0-9a-f]+ +0*2 +00 +A +0 +0 +1
#...
 +LOAD +0x[0-9a-f]+ +0x0*3000 +0x0*3000 +0x0*2 +0x0*2 +R +0x[0-9a-f]+
#pass
