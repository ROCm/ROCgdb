#source: relax-reloc-no-pcrel.s
#as: -mrelax -mthin-add-sub
#readelf: -rW

#...
.*R_LARCH_RELAX.*
.*R_LARCH_ADD32.*
.*R_LARCH_SUB32.*
.*R_LARCH_ADD64.*
.*R_LARCH_SUB64.*
.*R_LARCH_ADD32.*
.*R_LARCH_SUB32.*
.*R_LARCH_ADD64.*
.*R_LARCH_SUB64.*
