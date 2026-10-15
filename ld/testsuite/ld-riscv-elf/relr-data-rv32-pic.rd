Relocation section '\.rela\.dyn'.*contains 6 entries:
[ 	]+Offset[ 	]+Info[ 	]+Type[ 	]+.*
12340000[ 	]+[0-9a-f]+[ 	]+R_RISCV_RELATIVE[ 	]+10004
12340004[ 	]+[0-9a-f]+[ 	]+R_RISCV_RELATIVE[ 	]+10008
1234000c[ 	]+[0-9a-f]+[ 	]+R_RISCV_RELATIVE[ 	]+12340028
12340008[ 	]+[0-9a-f]+[ 	]+R_RISCV_32[ 	]+0001000c[ 	]+sym_global \+ 0
12340018[ 	]+[0-9a-f]+[ 	]+R_RISCV_32[ 	]+0001000c[ 	]+sym_global \+ 0
12340020[ 	]+[0-9a-f]+[ 	]+R_RISCV_32[ 	]+00000000[ 	]+sym_weak_undef \+ 0

Relocation section '\.relr\.dyn'.*contains 2 entries which relocate 3 locations:
Index:[ 	]+Entry[ 	]+Address[ 	]+Symbolic Address
0000:[ 	]+12340010[ 	]+12340010[ 	]+aligned_local
0001:[ 	]+00000023[ 	]+12340014[ 	]+aligned_hidden
[ 	]+12340024[ 	]+aligned_DYNAMIC
