        # Likewise for the pseudo-C dialect.  PR gas/34558: `gotol' below v4
        # used to assemble silently as `goto' plus a relocation against an
        # undefined symbol `l'.
        .text
        gotol +1
        r1 = bswap16 r1
        r1 s/= r2
        if w1 == w2 goto 1
