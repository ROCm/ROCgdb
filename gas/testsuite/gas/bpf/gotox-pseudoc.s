# Test for the indirect jump instruction
    .text
    gotox r1
    gotox r9
    goto 1
    exit
