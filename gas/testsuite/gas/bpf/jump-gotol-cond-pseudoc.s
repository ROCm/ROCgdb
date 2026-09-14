        # PR gas/34558: `gotol' written where a compound conditional jump
        # expects `goto'.  Conditional jumps only have the 16-bit `off'
        # field, so there is no `gotol' form of them.  These used to match
        # the embedded `goto%w%d16' of the conditional templates, with the
        # trailing `l' parsed as the start of the branch offset expression,
        # silently assembling to a conditional jump plus a relocation
        # against an undefined symbol `l'.
        .text
        if r1 > r2 gotol +1
        if r1 & 5 gotol -1
        if w1 == w2 gotol +1
