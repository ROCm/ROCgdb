#name: strings default minimum length (4)
#source: strings-len.s
#strings: -d
# .asciz and/or `strings' fishing out strings is broken for .bits_per_byte > 8
#xfail: *c4x-*-* *c54x-*-*

bbbb
ccccc
dddddddd
eeeeeeeeeee
