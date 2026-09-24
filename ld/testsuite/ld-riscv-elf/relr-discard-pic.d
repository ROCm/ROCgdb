#source: discard.s
#as:
#ld: -shared -Tdiscard.ld -z pack-relative-relocs
#readelf: -rW

Relocation section '\.rela\.dyn'.*
[ 	]+Offset[ 	]+Info[ 	]+Type.*
0+(20008|20010)[ 	]+[0-9a-f]+[ 	]+R_RISCV_(32|64)[ 	]+0+1000c[ 	]+sym_global \+ 0

Relocation section '\.relr\.dyn'.*contains 1 entry which relocates 1 location:
Index:[ 	]+Entry[ 	]+Address[ 	]+Symbolic Address
0000:[ 	]+0+(20004|20008)[ 	]+0+(20004|20008)[ 	]+_GLOBAL_OFFSET_TABLE_ \+ (0x4|0x8)
