#source: relax-debug-line-opcode.s
#as: -mno-relax
#readelf: -W -wl

#...
.*  Special opcode 62: advance Address by 4 to 0x4 and Line by 1 to 11
.*  Special opcode 118: advance Address by 8 to 0xc and Line by 1 to 12
.*  Special opcode 62: advance Address by 4 to 0x10 and Line by 1 to 13
.*  Special opcode 118: advance Address by 8 to 0x18 and Line by 1 to 14
#pass
