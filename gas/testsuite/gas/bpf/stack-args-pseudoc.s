# Test for r11, used to access stack arguments
    .text
    r1 = *(u64 *)(r11 + 8)
    *(u64 *)(r11 - 8) = r2
    *(u64 *)(r11 - 16) = 4
