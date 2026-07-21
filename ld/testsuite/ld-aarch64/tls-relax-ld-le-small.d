#source: tls-relax-ld-le-small.s
#ld: -T relocs.ld -e0
#objdump: -dr
#...
 +10000:	910003fd 	mov	x29, sp
 +10004:	d53bd040 	mrs	x0, tpidr_el0
 +10008:	91004000 	add	x0, x0, #0x10
 +1000c:	d503201f 	nop
 +10010:	d503201f 	nop
 +10014:	aa0003e1 	mov	x1, x0
 +10018:	d503201f 	nop
 +1001c:	90000000 	adrp	x0, 10000 <.*>
 +10020:	d65f03c0 	ret
