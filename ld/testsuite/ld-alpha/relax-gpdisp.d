#ld: -relax
#objdump: -dw --no-show-raw-insn

.*: +file format .*


Disassembly of section .text:

[a-f0-9]+ <_start>:
 +[a-f0-9]+:	ldah	gp,[0-9]+\(t12\)
 +[a-f0-9]+:	lda	gp,-?[0-9]+\(gp\)
 +[a-f0-9]+:	unop[ 	]*
 +[a-f0-9]+:	bsr	ra,[a-f0-9]+ <foo\+0x8>
 +[a-f0-9]+:	ldah	gp,[0-9]+\(ra\)
 +[a-f0-9]+:	lda	gp,-?[0-9]+\(gp\)
 +[a-f0-9]+:	ret
#pass
