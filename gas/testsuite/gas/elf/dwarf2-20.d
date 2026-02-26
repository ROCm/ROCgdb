#as: -gdwarf-3
#readelf: -wr
#name: DWARF2_20: debug ranges ignore non-code sections
# score-elf, tic6x-elf and xtensa-elf need special handling to support .nop 16
#xfail: score-* tic6x-* xtensa-*

Contents of the .debug_aranges section:

[ 	]+Length:[ 	]+(16|20|28|44)
[ 	]+Version:.*
[ 	]+Offset into .debug_info:[ 	]+(0x)?0
[ 	]+Address size:[ 	]+(2|3|4|8)
[ 	]+Segment size:[ 	]+0

[ 	]+Address[ 	]+Length
[ 	]+0+000 0+010 ?
[ 	]+0+000 0+000 ?
#pass
