#source: table-jump-zcmt-no-zicsr.s
#as: -march=rv64i -mno-arch-attr
#ld: --relax-tbljal
#readelf: -SWs

#failif
#...
.*(__jvt_base\$|\.riscv\.jvt).*
#pass
