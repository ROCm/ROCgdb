	.section .riscv.attributes,"",@0x70000003
	.byte 0x41
0:
	.4byte 1f-0b
	.asciz "riscv"
2:
	.byte 1
	.4byte 1f-2b
	.byte 5
	.asciz "rv64i2p1_zicsr2p0_zcmt1p0"
1:

	.section .text
	.globl _start
	.type _start, %function
_start:
	.rept 70
	call target
	.endr
	ret

	.type target, %function
target:
	ret
