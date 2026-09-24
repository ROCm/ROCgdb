#source: relr-relax.s
#as: -march=rv64i -mabi=lp64
#ld: -m[riscv_choose_lp64_emul] -pie -z pack-relative-relocs -T relr-relocs.ld
#readelf: -rW -x .text

Relocation section '\.relr\.dyn'.*contains 2 entries which relocate 2 locations:
Index:[ 	]+Entry[ 	]+Address[ 	]+Symbolic Address
0000:[ 	]+0000000000010008[ 	]+0000000000010008[ 	]+x
0001:[ 	]+0000000000000003[ 	]+0000000000010010[ 	]+x \+ 0x8

Hex dump of section '\.text':
  0x00010000 [0-9a-f]+ [0-9a-f]+ 18000100 00000000 .*
  0x00010010 18000100 00000000 67800000 [0-9a-f]+ .*
