#source: add-imm-zero-opt.s
#target: [check_shared_lib_support]
#ld: -shared -T add-imm-zero-opt.ld --defsym=got_base=0x20000 -e0 --no-relax
#objdump: -dr

.*: +file format .*

Disassembly of section .text:

0+10000 <test>:
 +10000:	91000000 	add	x0, x0, #0x0
 +10004:	91400041 	add	x1, x2, #0x0, lsl #12
 +10008:	91000063 	add	x3, x3, #0x0
 +1000c:	910000a4 	add	x4, x5, #0x0
 +10010:	914000e6 	add	x6, x7, #0x0, lsl #12
 +10014:	91000108 	add	x8, x8, #0x0
 +10018:	91000149 	add	x9, x10, #0x0
 +1001c:	1100016b 	add	w11, w11, #0x0
 +10020:	910003ec 	mov	x12, sp
 +10024:	910001bf 	mov	sp, x13
 +10028:	910021ce 	add	x14, x14, #0x8
 +1002c:	d65f03c0 	ret
