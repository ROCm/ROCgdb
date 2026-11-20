    .section .text

    .globl _start
    .type   _start, %function
_start:
    .rept 70
    call foo_global
    call foo_local
    .endr

    .globl  foo_global
    .type   foo_global, %function
foo_global:
    ret

    .type   foo_local, %function
foo_local:
    ret
