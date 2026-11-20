    .section .text.2
    .option relax
    .option rvc
h:
    call b
    .text
    .option norelax
    .option norvc
b:
    j e
    .option rvc
    .option relax
    .balign 8
c:
    .dword g + 0
e:
    .option norvc
    .option norelax
    ld x1, 0(x2)
    call h
    .balign 4
g:
    .word 0x10
