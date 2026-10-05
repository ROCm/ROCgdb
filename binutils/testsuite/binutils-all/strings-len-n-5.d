#name: strings -n 5 (5-char minimum)
#source: strings-len.s
#strings: -d -n 5
# .asciz and/or `strings' fishing out strings is broken for .bits_per_byte > 8
#xfail: *c4x-*-* *c54x-*-*

ccccc
dddddddd
eeeeeeeeeee
