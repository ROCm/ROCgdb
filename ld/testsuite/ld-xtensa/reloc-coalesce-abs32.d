#source: reloc-coalesce-abs32-1.s
#source: reloc-coalesce-abs32-2.s
#as: --no-abs32-rela
#ld: -T coalesce.t
#objdump: -d
#name: coalesce R_XTENSA_32 and R_XTENSA_32_ABS literals

# Both objects load g_name + 4, one through R_XTENSA_32 with the addend in
# the literal word and one through R_XTENSA_32_ABS with the addend in
# r_addend.  The two literals must be coalesced, leaving three literals
# before main, and both loads must still see g_name + 4.

#...
0000000c <main>:
#...
 +[0-9a-f]+:	[0-9a-f]+ +	l32r	a6, 4 <main-0x8> \([0-9a-f]+ <g_name\+0x4>\)
#...
[0-9a-f]+ <foo>:
#...
 +[0-9a-f]+:	[0-9a-f]+ +	l32r	a6, 4 <main-0x8> \([0-9a-f]+ <g_name\+0x4>\)
#pass
