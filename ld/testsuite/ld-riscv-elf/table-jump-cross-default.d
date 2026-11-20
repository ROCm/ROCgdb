#source: table-jump-cross.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian
#ld: -melf64lriscv
#objdump: -d -M no-aliases

#...
.* <_start>:
.*[ 	]auipc[ 	]+t1,0x[0-9a-f]+
.*[ 	]jalr[ 	]+zero,.* <far0>
#...
.* <near_site>:
.*[ 	]c\.j[ 	]+[0-9a-f]+ <fini_fn>
#pass
