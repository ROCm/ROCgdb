#source: table-jump-page.s
#as: -march=rv64ic_zcmt -mabi=lp64
#ld: -m[riscv_choose_lp64_emul] --no-warn-rwx-segments --defsym=page_size=0x1000 -Ttable-jump-page.ld --relax-tbljal
#objdump: -d -M no-aliases -j .text -j .fillers

#...
0*2000 <_start>:
.*[ \t]jal[ \t]+zero,3000 <page_target>
#...
0*3000 <page_target>:
#...
.*[ \t]cm\.jt[ \t]+0 # 314000 <far_target>
#pass
