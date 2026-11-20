#source: table-jump-no-benefit.s
#as: -march=rv64ic_zcmt
#ld: --relax-tbljal
#readelf: -sW

#failif
#...
.*__jvt_base\$.*
#pass
