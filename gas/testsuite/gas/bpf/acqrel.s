# Test for load-acquire and store-release instructions
    .text
    ldacqb %r1, [%r2+0]
    ldacqh %r1, [%r2+2]
    ldacqw %r1, [%r2-4]
    ldacqdw %r1, [%r2+0x7fff]
    strelb [%r1+0], %r2
    strelh [%r1+2], %r2
    strelw [%r1-4], %r2
    streldw [%r1+0x7fff], %r2
