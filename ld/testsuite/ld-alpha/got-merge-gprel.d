#source: got-merge-gprel-a.s
#source: got-merge-gprel-b.s
#ld: -relax -melf64alpha
#readelf: -SW

# The link must not fail with "relocation truncated to fit: GPREL16", and
# the .got holds only the 8000 entries against far.
#...
 +\[ *[0-9]+\] \.got +PROGBITS +[0-9a-f]+ +[0-9a-f]+ 00fa00 .*
#pass
