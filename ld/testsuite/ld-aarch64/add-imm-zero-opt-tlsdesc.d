#source: add-imm-zero-opt-tlsdesc.s
#target: [check_shared_lib_support]
#ld: -shared -T add-imm-zero-opt.ld --defsym=got_base=0x1ffd8 -e0 --no-warn-rwx-segments
#objdump: -dr

.*: +file format .*

Disassembly of section .text:

0+10000 <test>:
 +10000:	aa0103e0 	mov	x0, x1
 +10004:	d503201f 	nop
 +10008:	910003e3 	mov	x3, sp
 +1000c:	9100009f 	mov	sp, x4
 +10010:	d65f03c0 	ret
#...
