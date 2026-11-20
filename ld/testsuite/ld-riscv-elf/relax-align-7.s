        .text
        .option relax
        .balign 2
        .global _start
        .type _start, @function
_start:
        lui a0, %hi(foo)
        addi a0, a0, %lo(foo)
	.option norvc
	mul a0, a1, a4
        .option push

.ifdef NORELAX
        .option norelax
.endif
.ifdef NORVC
        .option norvc
.endif
        .balign 4
SHOULD_ALIGN_4_HERE:
        .word 0x12345678

        .option pop

foo:
        ret
