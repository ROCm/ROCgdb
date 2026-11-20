        .text
        .option relax
        .balign 4
        .global _start
        .type _start, @function
_start:
        lui a0, %hi(foo)
        addi a0, a0, %lo(foo)
        ret
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
