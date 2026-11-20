#source: table-jump-shared.s
#as: -march=rv64ic_zcmt -fpic
#ld: -shared --relax-tbljal
#readelf: -rW

#failif
#...
.*R_RISCV_(TABLE_JUMP|DELETE|DELETE_AND_RELAX).*
#pass
