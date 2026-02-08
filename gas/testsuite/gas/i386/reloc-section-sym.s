	.text
	nop
named_local:
	.byte 0
.Ltemp:
	nop

.section .text1,"ax"
	call named_local
	call .Ltemp

.data
	.quad named_local + 16
	.quad .Ltemp + 16
