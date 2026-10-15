        # Likewise for the pseudo-C dialect.
        .text
        lock *(u64 *)(r1 + 8) += r2
        w1 = load_acquire((u8 *)(r2 + 0))
        r1 = load_acquire((u64 *)(r2 + 0))
        store_release((u16 *)(r1 + 0), w2)
        store_release((u64 *)(r1 + 0), r2)
        gotox r1
