#source: relax-reloc-fixed-advance-pc.s
#as: -mrelax
#readelf: -r -wl -W

#...
Relocation section '\.rela\.text'.*
#...
.*R_LARCH_RELAX.*
#...
Raw dump of debug contents of section \.debug_line:
#...
.*  Advance Line by 1 to 11
.*  Advance PC by fixed size amount [0-9]+ to 0x[0-9a-f]+
.*  Copy .*
.*  Advance Line by 1 to 12
.*  Advance PC by fixed size amount [0-9]+ to 0x[0-9a-f]+
.*  Copy .*
#pass
