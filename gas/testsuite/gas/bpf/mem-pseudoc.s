# eBPF tests for MEM pseudo-C instructions, modulus lddw.

        .text
	r0 = *(u32 *)skb[48879]
	r0 = *(u16 *)skb[48879]
	r0 = *(u8 *)skb[48879]
	r0 = *(u64 *)skb[48879]
	r0 = *(u32 *)skb[r3 + 0xbeef]
	r0 = *(u16 *)skb[r5 + 0xbeef]
	r0 = *(u8 *)skb[r7 + 0xbeef]
	r0 = *(u64 *)skb[r9 + 0xbeef]
	r2 = *(u32 *)(r1 + 32495)
	r2 = *(u16 *)(r1 + 32495)
	r2 = *(u8 *)(r1 + 32495)
	r2 = *(u64 *)(r1 - 2)
	*(u32 *)(r1 + 32495) = r2
	*(u16 *)(r1 + 32495) = r2
	*(u8 *)(r1 + 32495) = r2
	*(u64 *)(r1 - 2) = r2
        *(u8 *)(r1 + 0x7eef) = 0x11223344
	*(u16 *)(r1 + 0x7eef) = 0x11223344
	*(u32 *)(r1 + 0x7eef) = 0x11223344
	*(u64 *)(r1 + -2) = 0x11223344
        r2 = *(s32*)(r1+0x7eef)
        r2 = *(s16*)(r1+0x7eef)
        r2 = *(s8*)(r1+0x7eef)
        r2 = *(s64*)(r1+0x7eef)
	r0 = *(u64 *)skb[r5 + 0]
	r2 = *(u32 *)(r1 + 0)
