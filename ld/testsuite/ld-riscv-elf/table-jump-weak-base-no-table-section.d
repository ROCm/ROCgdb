#source: table-jump-weak-base.s
#as: -march=rv64ic_zcmt --defsym SITES=1
#ld: -Ttable-jump-weak-base.ld --relax-tbljal
#readelf: -SW

#failif
#...
.*\.riscv\.jvt.*
#pass
