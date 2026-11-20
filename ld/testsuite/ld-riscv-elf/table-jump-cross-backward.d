#source: table-jump-cross.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian --defsym TEST_BACKWARD=1
#ld: -melf64lriscv --relax-tbljal
#objdump: -d -M no-aliases

#...
.* <_start>:
.*[ 	]cm\.jt[ 	]+0 # [0-9a-f]+ <far0>
#...
.* <fini_fn>:
.*[ 	]jal[ 	]+zero,[0-9a-f]+ <near_site>
#pass
