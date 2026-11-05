target:
        .ifdef  exact_mode
        .option exact
        .endif
        li  a0, 0x8001
        li  a0, 0x1f01
        li  a0, 0x12345001
        li  a0, 0xf2345001
