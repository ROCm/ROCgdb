.macro  vector name
    .weak   \name
    .set    \name, __bad_interrupt
    XJMP    \name
.endm

    .section .vectors,"ax",@progbits
    .global __vectors
    .type __vectors,@function
__vectors:
    XJMP    __init
    vector  __vector_1
    vector  __vector_2
	vector	__vector_3
	vector	__vector_4
	vector	__vector_5
	vector	__vector_6
    XCLH

    .text
    .global __bad_interrupt
    .func   __bad_interrupt
__bad_interrupt:
    .weak   __vector_default
    .set    __vector_default, __vectors
    XJMP    __vector_default
    .endfunc

    .section .init0,"ax",@progbits
    .weak   __init
__init:
    cli
