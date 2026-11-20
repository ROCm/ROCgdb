#source: table-jump-address-edge.s
#as: -march=rv64ic_zcmt -mabi=lp64
#ld: -melf64lriscv --no-warn-rwx-segments -Ttable-jump-address-edge.ld --relax-tbljal
#objdump: -d -M no-aliases -j .text -j .riscv.jvt

#...
.*[ \t]0000000000300000[ \t]+jvt\.jt\[0\]:.*
#...
0*4000 <_start>:
.*[ \t](c\.)?lui[ \t]+a0,0x1
.*[ \t]addi[ \t]+a0,a0,-2048 # 800 <x0_target>
.*[ \t](c\.)?lui[ \t]+a1,0x3
.*[ \t]addi[ \t]+a1,a1,-2048 # 2800 <gp_target>
.*[ \t]cm\.jt[ \t]+0 # 300000 <far_target>
#pass
