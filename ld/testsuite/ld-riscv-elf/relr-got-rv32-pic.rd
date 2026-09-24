Relocation section '\.rela\.dyn'.*contains 3 entries:
[ 	]+Offset[ 	]+Info[ 	]+Type[ 	]+.*
00020010[ 	]+[0-9a-f]+[ 	]+R_RISCV_32[ 	]+00010008[ 	]+sym_global \+ 0
00020014[ 	]+[0-9a-f]+[ 	]+R_RISCV_32[ 	]+0000002a[ 	]+sym_global_abs \+ 0
00020018[ 	]+[0-9a-f]+[ 	]+R_RISCV_32[ 	]+00000000[ 	]+sym_weak_undef \+ 0

Relocation section '\.relr\.dyn'.*contains 2 entries which relocate 3 locations:
Index:[ 	]+Entry[ 	]+Address[ 	]+Symbolic Address
0000:[ 	]+00020004[ 	]+00020004[ 	]+_GLOBAL_OFFSET_TABLE_ \+ 0x4
0001:[ 	]+00000007[ 	]+00020008[ 	]+_GLOBAL_OFFSET_TABLE_ \+ 0x8
[ 	]+0002000c[ 	]+_GLOBAL_OFFSET_TABLE_ \+ 0xc
