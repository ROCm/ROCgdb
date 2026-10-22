# Test for address space cast instructions
    .text
    addrc %r1, %r1, 0, 1
    addrc %r0, %r1, 1, 0
    addrc %r9, %r2, 0, 1
    mov %r2, %r3
    addrc %r3, %r4, 0x0, 0x1
    addrc %r5, %r6, 0x1, 0x0
