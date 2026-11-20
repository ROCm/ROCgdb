#source: table-jump-empty-mixed.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian
#ld: -melf64lriscv --relax-tbljal -Ttable-jump-empty-script.ld
#objdump: -s -j .riscv.jvt

#...
Contents of section \.riscv\.jvt:
[ 	]+1004 aa55 .*
#pass
