#source: relr-got-start.s
#as: -march=rv64i -mabi=lp64
#ld: -m[riscv_choose_lp64_emul] -pie -z pack-relative-relocs -T relr-relocs.ld
#readelf: -rW

Relocation section '\.relr\.dyn'.*contains 1 entry which relocates 1 location:
Index:[ 	]+Entry[ 	]+Address[ 	]+Symbolic Address
0000:[ 	]+0000000000020008[ 	]+0000000000020008[ 	]+_GLOBAL_OFFSET_TABLE_ \+ 0x8
