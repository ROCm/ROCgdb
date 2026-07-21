#source: tls-tiny-gd-le.s
#as: -mabi=ilp32
#ld: -m [aarch64_choose_ilp32_emul] -T relocs-ilp32.ld -e0
#notarget: *-*-nto*
#objdump: -dr
#...

Disassembly of section .text:

00010000 \<test\>:
 +10000:	d53bd041 	mrs	x1, tpidr_el0
 +10004:	2a0103e0 	mov	w0, w1
 +10008:	11002000 	add	w0, w0, #0x8
