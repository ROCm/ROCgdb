# Test for address space cast instructions
    .text
    r1 = addr_space_cast(r1, 0, 1)
    r0 = addr_space_cast(r1, 1, 0)
    r9 = addr_space_cast(r2, 0, 1)
    r2 = r3
