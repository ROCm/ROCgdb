#source: relr-textrel.s
#ld: -pie -z pack-relative-relocs -T relr-relocs.ld
#readelf: -drW

#...
.*\(TEXTREL\)[ 	]+0x0
#...
.*\(RELR\).*
.*\(RELRSZ\)[ 	]+(4|8) \(bytes\)
.*\(RELRENT\)[ 	]+(4|8) \(bytes\)
#...
Relocation section '\.relr\.dyn' .* contains 1 entry which relocates 1 location:
Index:[ 	]+Entry[ 	]+Address[ 	]+Symbolic Address
0000:[ 	]+(00000000|)00010000[ 	]+(00000000|)00010000[ 	]+_start
