#source: tlsle-add-hi12-opt.s
#target: [check_shared_lib_support]
#ld: -shared -T relocs.ld -e0
#objdump: -dr

.*: .*

Disassembly of section .text:

0+10000 <test>:
 +10000:	aa0103e0 	mov	x0, x1
 +10004:	914003e2 	add	x2, sp, #0x0, lsl #12
 +10008:	9140007f 	add	sp, x3, #0x0, lsl #12
 +1000c:	914004a4 	add	x4, x5, #0x1, lsl #12
