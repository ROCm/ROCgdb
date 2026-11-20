#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JT=1 -mbig-endian
#ld: -melf64briscv -Ttable-jump.ld --relax-tbljal
#objdump: -s -j .riscv.jvt

#...
.*:     file format elf64-bigriscv
#...
Contents of section .riscv.jvt:
[ 	]+0+[0-9a-f]*[ 	]+00000000[ 	]+00004016[ 	]+00000000[ 	]+00004014.*
