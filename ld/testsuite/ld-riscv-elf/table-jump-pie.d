#source: table-jump-pie.s
#as: -march=rv64ic_zcmt -fpic
#ld: -pie --relax-tbljal
#readelf: -SWs

#failif
#...
.*(__jvt_base\$|\.riscv\.jvt).*
#pass
