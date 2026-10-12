        # Instructions that exist, but not in the selected ISA spec, must be
        # diagnosed as such rather than as a generic parse failure.
        .text
        jal 1
        movs %r1, %r2, 8
        aadd [%r1+8], %r2
        jeq32 %r1, %r2, 1
