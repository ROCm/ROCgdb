        # Instructions that require ISA v4 must be diagnosed as such
        # under -misa-spec=v3.  aadd is v3 and must still assemble.
        .text
        aadd [%r1+8], %r2
        ldacqb %r1, [%r2+0]
        ldacqdw %r1, [%r2+0]
        strelh [%r1+0], %r2
        streldw [%r1+0], %r2
        ja %r1
