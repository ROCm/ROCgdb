#as: -march=rv64i_zalasr
#source: zalasr.s
#objdump: -d

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[  	]+3405052f[  	]+lb.aq[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3605052f[  	]+lb.aqrl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3405152f[  	]+lh.aq[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3605152f[  	]+lh.aqrl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3405252f[  	]+lw.aq[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3605252f[  	]+lw.aqrl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3405352f[  	]+ld.aq[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3605352f[  	]+ld.aqrl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3aa5002f[  	]+sb.rl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3ea5002f[  	]+sb.aqrl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3aa5102f[  	]+sh.rl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3ea5102f[  	]+sh.aqrl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3aa5202f[  	]+sw.rl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3ea5202f[  	]+sw.aqrl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3aa5302f[  	]+sd.rl[  	]+a0,\(a0\)
[ 	]+[0-9a-f]+:[  	]+3ea5302f[  	]+sd.aqrl[  	]+a0,\(a0\)
