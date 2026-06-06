#source: relax-uleb128.s
#as: -mrelax
#readelf: -rW

#...
.*R_LARCH_ADD_ULEB128.*10.*L2.*
.*R_LARCH_SUB_ULEB128.*00.*L1.*
.*R_LARCH_ADD_ULEB128.*1c.*L3.*
.*R_LARCH_SUB_ULEB128.*10.*L2.*
