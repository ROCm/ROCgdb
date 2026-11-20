#source: table-jump-rv32.s
#as: -march=rv32ic_zcmt
#ld: -Ttable-jump.ld --relax-tbljal=zcmt -melf32lriscv
#objdump: -d
#dump: table-jump-rv32.d
