#source: table-jump-fixed-slot.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian
#ld: -melf64lriscv --relax-tbljal -Ttable-jump-misaligned.ld
#error: .*Zcmt jump table is not 64-byte aligned
