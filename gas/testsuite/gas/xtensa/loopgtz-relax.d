#as:
#objdump: -d

#...
.*beqz.*a9,.*
.*bltz.*a9,.*
.*loopgtz.*a9,.*
.*rsr.lend.*a9
.*wsr.lbeg.*a9
.*l32r.*a9,.*
.*nop
.*wsr.lend.*a9
.*isync
.*rsr.lcount.*a9
.*addi.*a9, a9, 1
#...
