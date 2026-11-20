#source: table-jump-deferred-shrink.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian
#ld: -melf64lriscv --relax-tbljal
#objdump: -d -M no-aliases

#...
.* <_start>:
.*[ 	](jal[ 	]+zero,|c\.j[ 	]+)[0-9a-f]+ <nearby>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <nearby>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <nearby>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <nearby>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <nearby>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <nearby>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <nearby>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <nearby>
[ 	]+\.\.\.
.* <nearby>:
.*[ 	]c\.jr[ 	]+ra
