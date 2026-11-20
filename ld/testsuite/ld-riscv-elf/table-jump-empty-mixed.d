#source: table-jump-empty-mixed.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian
#ld: -melf64lriscv -Ttable-jump-empty-mixed.ld --relax-tbljal
#objdump: -s -j .mixed

#...
Contents of section .mixed:
 1004 11223344 55667788 .*
#pass
