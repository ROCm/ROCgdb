#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JT_JALT=1
#ld: -Ttable-jump.ld --no-relax --relax-tbljal
#readelf: -SW

#failif
#...
.*\.riscv\.jvt.*
#pass
