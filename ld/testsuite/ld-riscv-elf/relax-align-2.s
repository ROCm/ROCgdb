        .text
        .option relax
        .balign 4
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
        .balign 8
SHOULD_ALIGN_8_HERE:
        .word 0x12345678

        .option pop

foo:
        ret
