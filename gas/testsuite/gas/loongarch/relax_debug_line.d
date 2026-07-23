#as: --gdwarf-5 -mrelax
#readelf: -r --wide

Relocation section '\.rela\.debug_line' at offset .* contains 3 entries:
#...
0+22.*R_LARCH_32[ \t]+[0-9]+.*
0+2c.*R_LARCH_32[ \t]+[0-9]+.*
0+36.*R_LARCH_(32|64)[ \t]+[0-9]+.*
#pass
