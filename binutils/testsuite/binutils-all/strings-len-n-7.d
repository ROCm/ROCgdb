#name: strings -n 7 (excludes 3-/4-/5-/6-char runs)
#source: strings-len.s
#strings: -d -n 7
# .asciz and/or `strings' fishing out strings is broken for .bits_per_byte > 8
#xfail: *c4x-*-* *c54x-*-*

dddddddd
eeeeeeeeeee
