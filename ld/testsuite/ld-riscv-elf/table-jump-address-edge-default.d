#source: table-jump-address-edge.s
#as: -march=rv64ic_zcmt -mabi=lp64
#ld: -melf64lriscv --no-warn-rwx-segments -Ttable-jump-address-edge.ld
#objdump: -d -M no-aliases -j .text

#...
0*4000 <_start>:
.*[ \t]addi[ \t]+a0,zero,2040
.*[ \t]addi[ \t]+a1,gp,2040 # 27f8 <gp_target>
#pass
