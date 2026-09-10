        /* This test checks that flexible spacing is supported in the
           pseudoc syntax.  */
        r4 = 0xdeadbeefll
        r4 = 0xdeadbeef ll
        goto +1
        goto+1
        goto1
        if w3==3 goto+1
        if w3==3 goto1
        /* A mnemonic may still run into an operand that cannot be
           mistaken for the rest of a longer mnemonic.  */
        if r1 == 5goto+1
        call5
        lock*(u64 *)(r1 + 0) += r2
