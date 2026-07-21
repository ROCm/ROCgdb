#source: add-imm-zero-opt.s
#target: [check_shared_lib_support]
#ld: -shared -T add-imm-zero-opt.ld --defsym=got_base=0x20000 -e0
#objdump: -dr

.*: +file format .*

Disassembly of section .text:

0+10000 <test>:
 +10000:	d503201f 	nop
 +10004:	aa0203e1 	mov	x1, x2
 +10008:	d503201f 	nop
 +1000c:	aa0503e4 	mov	x4, x5
 +10010:	aa0703e6 	mov	x6, x7
 +10014:	d503201f 	nop
 +10018:	aa0a03e9 	mov	x9, x10
 +1001c:	2a0b03eb 	mov	w11, w11
 +10020:	910003ec 	mov	x12, sp
 +10024:	910001bf 	mov	sp, x13
 +10028:	910021ce 	add	x14, x14, #0x8
 +1002c:	d65f03c0 	ret
