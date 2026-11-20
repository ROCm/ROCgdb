#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JT_JALT=1
#ld: --relax-tbljal
#readelf: -SWs

#...
.*\.riscv\.jvt[ 	]+PROGBITS[ 	]+[0-9a-f]+[ 	]+[0-9a-f]+[ 	]+[0-9a-f]+.*
#...
[ 	]+[0-9]+:[ 	]+0+[0-9a-f]+[ 	]+0[ 	]+NOTYPE[ 	]+LOCAL[ 	]+DEFAULT[ 	]+[0-9]+[ 	]+__jvt_base\$
#pass
