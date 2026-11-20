#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JT_JALT=1
#ld: -Ttable-jump.ld --relax-tbljal --no-relax
#readelf: -SW

#failif
#...
.*\.riscv\.jvt.*
#pass
