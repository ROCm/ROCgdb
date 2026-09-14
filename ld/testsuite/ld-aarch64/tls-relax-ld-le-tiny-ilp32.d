#source: tls-relax-ld-le-tiny.s
#as: -mabi=ilp32
#ld: -m [aarch64_choose_ilp32_emul] -T relocs-ilp32.ld -e0
#notarget: *-*-nto*
#objdump: -dr
#...
 +10000:	910003fd 	mov	x29, sp
 +10004:	d53bd040 	mrs	x0, tpidr_el0
 +10008:	11002000 	add	w0, w0, #0x8
 +1000c:	d503201f 	nop
 +10010:	aa0003e1 	mov	x1, x0
 +10014:	d503201f 	nop
 +10018:	90000000 	adrp	x0, 10000 <main>
 +1001c:	d65f03c0 	ret
