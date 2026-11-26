#source: table-jump-page.s
#as: -march=rv64ic_zcmt -mabi=lp64 --defsym TEST_JAL=1
#ld: -m[riscv_choose_lp64_emul] --no-warn-rwx-segments --defsym=page_size=0x200000 -Ttable-jump-page.ld --relax-tbljal
#objdump: -d -M no-aliases -j .text -j .fillers

#...
0*400000 <_start>:
.*[ \t]auipc[ \t]+t1,0x200
.*[ \t]jalr[ \t]+t0,0\(t1\) # 600000 <page_target>
#...
0*600000 <page_target>:
#...
.*[ \t]cm\.jt[ \t]+0 # b10000 <far_target>
#pass
