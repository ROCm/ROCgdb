#source: relax-debug-line-opcode.s
#as: -mrelax
#readelf: -r -wl -W

#...
Relocation section '\.rela\.debug_line' at offset .* contains 7 entries:
#...
.*R_LARCH_ADD16[ 	]+[0-9]+.*
.*R_LARCH_SUB16[ 	]+[0-9]+.*
.*R_LARCH_ADD16[ 	]+[0-9]+.*
.*R_LARCH_SUB16[ 	]+[0-9]+.*
#...
Raw dump of debug contents of section \.debug_line:
#...
.*  Special opcode 62: advance Address by 4 to 0x4 and Line by 1 to 11
.*  Advance Line by 1 to 12
.*  Advance PC by fixed size amount [0-9]+ to 0x[0-9a-f]+
.*  Copy .*
.*  Special opcode 62: advance Address by 4 to 0x10 and Line by 1 to 13
.*  Advance Line by 1 to 14
.*  Advance PC by fixed size amount [0-9]+ to 0x[0-9a-f]+
.*  Copy .*
.*  Advance PC by [0-9]+ to 0x[0-9a-f]+
.*  Extended opcode 1: End of Sequence
#pass
