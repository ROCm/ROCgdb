#source: table-jump-empty-merge.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian --defsym MIXED=1
#ld: -melf64lriscv --relax-tbljal -Ttable-jump-empty-merge.ld
#readelf: -SWx .out

#...
 +\[ *[0-9]+\] \.out +PROGBITS +0+3000 +[0-9a-f]+ +0*9 +00 +A +0 +0 +1
#...
Hex dump of section '\.out':
 +0x0*3000 61626300 78797a00 21 +abc\.xyz\.!
#pass
