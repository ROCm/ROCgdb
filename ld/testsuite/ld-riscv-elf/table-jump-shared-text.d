#source: table-jump-shared.s
#as: -march=rv64ic_zcmt -fpic
#ld: -shared --relax-tbljal
#objdump: -d

#failif
#...
.*cm\.j(al)?t.*
#pass
