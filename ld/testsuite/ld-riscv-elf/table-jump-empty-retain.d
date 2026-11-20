#source: table-jump-empty-mixed.s
#source: table-jump-empty-retain.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian
#ld: -melf64lriscv --relax-tbljal -Ttable-jump-empty-flags.ld
#readelf: -SW

#...
 +\[ *[0-9]+\] \.riscv\.jvt +PROGBITS +0+3000 +[0-9a-f]+ +0*3 +00 +A[oR] +0 +0 +1
#pass
