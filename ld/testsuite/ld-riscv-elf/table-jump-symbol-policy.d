#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JALT=1
#ld: -Ttable-jump.ld --relax-tbljal
#readelf: -sW

#...
[ 	]+[0-9]+:[ 	]+0+[0-9a-f]+[ 	]+0[ 	]+NOTYPE[ 	]+LOCAL[ 	]+DEFAULT[ 	]+[0-9]+[ 	]+__jvt_base\$
#pass
