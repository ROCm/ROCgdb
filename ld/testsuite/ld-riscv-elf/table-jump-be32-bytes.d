#source: table-jump-rv32.s
#as: -march=rv32ic_zcmt -mbig-endian
#ld: -melf32briscv -Ttable-jump.ld --relax-tbljal
#objdump: -s -j .riscv.jvt

#...
.*:     file format elf32-bigriscv
#...
Contents of section .riscv.jvt:
[ 	]+0+[0-9a-f]*[ 	]+0000400e[ 	]+0000400c.*
