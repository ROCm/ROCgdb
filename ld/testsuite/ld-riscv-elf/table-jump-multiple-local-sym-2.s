    .section .text
    .type   bar, %function
bar:
    .rept 70
    call foo_local
    call foo_global
    .endr
    ret

    .type   foo_local, %function
foo_local:
    ret
