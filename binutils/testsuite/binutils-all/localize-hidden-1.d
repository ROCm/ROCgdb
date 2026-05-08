#PROG: objcopy
#objdump: --syms
#objcopy: --localize-hidden
#name: --localize-hidden test 1
#warning_output: localize-hidden-1.l
#...
0+1200 l .*\*ABS\*	0+ (\.hidden )?Lhidden
0+1300 l .*\*ABS\*	0+ (\.internal )?Linternal
0+1400 l .*\*ABS\*	0+ (\.protected )?Lprotected
0+1100 l .*\*ABS\*	0+ Ldefault
#...
0+2200 l .*\*ABS\*	0+ Ghidden
0+2300 l .*\*ABS\*	0+ Ginternal
0+3200 l .*\*ABS\*	0+ Whidden
0+3300 l .*\*ABS\*	0+ Winternal
#...
0+2100 g .*\*ABS\*	0+ Gdefault
0+2400 g .*\*ABS\*	0+ \.protected Gprotected
0+3100  w.*\*ABS\*	0+ Wdefault
0+3400  w.*\*ABS\*	0+ \.protected Wprotected
#pass
