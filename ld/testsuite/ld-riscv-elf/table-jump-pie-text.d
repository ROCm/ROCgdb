#source: table-jump-pie.s
#as: -march=rv64ic_zcmt -fpic
#ld: -pie --relax-tbljal
#objdump: -d

#failif
#...
.*cm\.j(al)?t.*
#pass
