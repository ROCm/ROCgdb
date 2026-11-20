        .text
        .option relax
        .balign 2
        .global _start
        .type _start, @function
_start:
	call foo
	mul a0, a1, a4
        .option push

.ifdef NORELAX
        .option norelax
.endif
.ifdef NORVC
        .option norvc
.endif
        .balign 16
SHOULD_ALIGN_16_HERE:
        .word 0x12345678

        .option pop

foo:
        ret
