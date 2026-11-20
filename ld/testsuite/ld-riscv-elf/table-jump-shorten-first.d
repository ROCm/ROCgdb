#source: table-jump-shorten-first.s
#as: -march=rv64ic_zcmt
#ld: --relax-tbljal
#readelf: -SW

#failif
#...
.*\.riscv\.jvt.*
#pass
