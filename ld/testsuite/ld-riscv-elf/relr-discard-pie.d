#source: discard.s
#as:
#ld: -pie -Tdiscard.ld -z pack-relative-relocs
#readelf: -rW

Relocation section '\.relr\.dyn'.*contains 2 entries which relocate 2 locations:
Index:[ 	]+Entry[ 	]+Address[ 	]+Symbolic Address
0000:[ 	]+0+(20004|20008)[ 	]+0+(20004|20008)[ 	]+_GLOBAL_OFFSET_TABLE_ \+ (0x4|0x8)
0001:[ 	]+0+3[ 	]+0+(20008|20010)[ 	]+_GLOBAL_OFFSET_TABLE_ \+ (0x8|0x10)
