#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JT_JALT=1
#ld: --relax-tbljal
#readelf: -rW

#failif
#...
.*(R_RISCV_(TABLE_JUMP|DELETE|DELETE_AND_RELAX)|unrecognized:).*
#pass
