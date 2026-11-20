#source: table-jump-cross.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian --defsym TEST_JAL=1
#ld: -melf64lriscv --relax-tbljal
#objdump: -d -M no-aliases

#...
.* <_start>:
.*[ 	]cm\.jt[ 	]+0 # [0-9a-f]+ <far0>
#...
.* <near_site>:
.*[ 	]auipc[ 	]+t1,0x[0-9a-f]+
.*[ 	]jalr[ 	]+zero,.*\(t1\) # [0-9a-f]+ <fini_fn>
#pass
