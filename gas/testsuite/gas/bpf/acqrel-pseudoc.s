# Test for load-acquire and store-release instructions
    .text
    w1 = load_acquire((u8 *)(r2 + 0))
    w1 = load_acquire((u16 *)(r2 + 2))
    w1 = load_acquire((u32 *)(r2 - 4))
    r1 = load_acquire((u64 *)(r2 + 0x7fff))
    store_release((u8 *)(r1 + 0), w2)
    store_release((u16 *)(r1 + 2), w2)
    store_release((u32 *)(r1 - 4), w2)
    store_release((u64 *)(r1 + 0x7fff), r2)
