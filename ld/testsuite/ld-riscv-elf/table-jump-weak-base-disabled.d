#source: table-jump-weak-base.s
#as: -march=rv64ic_zcmt --defsym SITES=5
#ld: -Ttable-jump-weak-base.ld --no-relax-tbljal
#objdump: -d

#...
Disassembly of section .text:
0+3000 <_start>:
[ 	]+3000:[ 	]+00000513[ 	]+li[ 	]+a0,0
#...
.*csrw[ 	]+jvt,a0
#...
0+[0-9a-f]+ <entry>:
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <foo_local>
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <foo_global>
#pass
