#source: table-jump-gc-dead.s
#as: -march=rv64ic_zcmt
#ld: --gc-sections --relax-tbljal -e _start
#readelf: -SWs

#failif
#...
.*(__jvt_base\$|\.riscv\.jvt).*
#pass
