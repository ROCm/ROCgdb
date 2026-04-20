#source: equ.s
#objdump: -t
#name: i386 equates (symtab check)

.*: +file format .*

SYMBOL TABLE:
.* \.text	.*
!.* \*ABS\*	.*
#pass
