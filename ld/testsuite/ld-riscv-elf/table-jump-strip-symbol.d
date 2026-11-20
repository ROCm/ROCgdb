#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JALT=1
#ld: -Ttable-jump.ld --strip-all --relax-tbljal
#readelf: -sW

#failif
#...
.*__jvt_base\$.*
#pass
