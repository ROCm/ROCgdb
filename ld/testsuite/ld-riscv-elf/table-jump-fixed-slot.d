#source: table-jump-fixed-slot.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian
#ld: -melf64lriscv -Ttable-jump-fixed-slot.ld --relax-tbljal
#objdump: -s -j .riscv.jvt

#...
Contents of section \.riscv\.jvt:
[ \t]+2000[ \t]+00003000 00000000 5a[ \t]+.*
#pass
