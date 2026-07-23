#source: relax-align-ld-r.s
#as: -mno-relax
#ld: -r
#objdump: -Dr

#failif
#...
.*R_LARCH_ALIGN.*
#...
