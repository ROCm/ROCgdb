#source: relax-align-noopt.s
#as: -mrelax
#readelf: -rW

#...
.*R_LARCH_ALIGN.*c
.*R_LARCH_RELAX.*
.*R_LARCH_ADD32.*
.*R_LARCH_SUB32.*
