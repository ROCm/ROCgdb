#source: unresolved-ignore-all.s
#ld: -e 0 -Ttext=0x100 --unresolved-symbols=ignore-all
#objdump: -d -M no-aliases

#...
0000000000000100 <a>:
[ 	]+100:[ 	]+58000085[ 	]+beq[ 	]+\$a0,[ 	]+\$a1,[ 	]+0[ 	]+#[ 	]+100[ 	]+<a>
[ 	]+104:[ 	]+40000080[ 	]+beqz[ 	]+\$a0,[ 	]+0[ 	]+#[ 	]+104[ 	]+<a\+0x4>
[ 	]+108:[ 	]+4c000000[ 	]+jirl[ 	]+\$zero,[ 	]+\$zero,[ 	]+0
[ 	]+10c:[ 	]+4c000001[ 	]+jirl[ 	]+\$ra,[ 	]+\$zero,[ 	]+0
[ 	]+110:[ 	]+1e000001[ 	]+pcaddu18i[ 	]+\$ra,[ 	]+0
[ 	]+114:[ 	]+4c000001[ 	]+jirl[ 	]+\$ra,[ 	]+\$zero,[ 	]+0
#pass
