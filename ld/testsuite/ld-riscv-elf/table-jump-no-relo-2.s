    .section .text

    .globl foo
    .type  foo, %function
foo:
    call bar


    .globl bar
    .type  bar, %function
bar:
    ret
