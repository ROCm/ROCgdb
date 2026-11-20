#source: table-jump-cross.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian
#ld: -melf64lriscv --relax-tbljal
#objdump: -d -M no-aliases

#...
.* <_start>:
.*[ 	]cm\.jt[ 	]+0 # [0-9a-f]+ <far0>
#...
.* <near_site>:
.*[ 	]jal[ 	]+zero,[0-9a-f]+ <fini_fn>
#...
Disassembly of section \.riscv\.jvt:
#...
.*[ 	]jvt\.jt\[31\]:[ 	]+far31
#pass
