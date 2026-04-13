# Checks to see that linking with "-O 0" does not merge strings.
# The various other -O options are there just to make sure that the parser handles them.
#source: merge5.a.s
#source: merge5.b.s
#ld: -T merge.ld -O 2 -O - -O 99 -O default -O fast -O 0
#nm: -s

0+1100 R .merge1
0+1104 R .merge2
#pass
