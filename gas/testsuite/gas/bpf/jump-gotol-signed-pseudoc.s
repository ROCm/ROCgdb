        # Signed branch offsets for the pseudo-C `gotol'.  PR gas/34558:
        # these used to match the shorter `goto' template, with the
        # trailing `l' parsed as the start of the offset expression.
        .text
        gotol +1
        gotol -1
        gotol 1
        gotol 1f
1:
        gotol 0
