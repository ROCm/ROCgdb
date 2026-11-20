#source: table-jump-shared.s
#as: -march=rv64ic_zcmt -fpic
#ld: -shared --relax-tbljal
#readelf: -SWs

#failif
#...
.*(__jvt_base\$|\.riscv\.jvt).*
#pass
