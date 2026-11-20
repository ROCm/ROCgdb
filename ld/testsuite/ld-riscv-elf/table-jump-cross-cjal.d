#source: table-jump-cross.s
#as: -march=rv32ic_zcmt -mabi=ilp32 -mlittle-endian --defsym TEST_CJAL=1
#ld: -melf32lriscv --relax-tbljal
#objdump: -d -M no-aliases

#...
.* <_start>:
.*[ 	]cm\.jt[ 	]+0 # [0-9a-f]+ <far0>
#...
.* <near_site>:
.*[ 	]jal[ 	]+ra,[0-9a-f]+ <fini_fn>
#pass
