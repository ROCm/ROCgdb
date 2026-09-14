        /* Flexible spacing stops at the end of a mnemonic: the input may
           not run a mnemonic into something that reads as the rest of a
           longer one.  See the %t tag in the opcode templates.  */
        .text
        ifr1>r2 goto+1
        callxr1
