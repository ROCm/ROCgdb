# Test for r11, used to access stack arguments
    .text
    ldxdw %r1, [%r11+8]
    stxdw [%r11-8], %r2
    stdw [%r11-16], 4
