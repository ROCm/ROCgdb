#source: table-jump-rv32.s
#as: -march=rv32ic_zcmt
#ld: -Ttable-jump.ld --relax-tbljal=zcmt -m[riscv_choose_ilp32_emul]
#objdump: -d
#dump: table-jump-rv32.d
