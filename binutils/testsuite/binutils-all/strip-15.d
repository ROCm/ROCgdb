#PROG: strip
#strip: -g
#readelf: -r
#notarget: rx-*
# The RX target text section is called "P" rather than ".text", so
# creating .rela.text by hand is wrong.

Relocation section '\.rela?\.text' at offset .* contains 2 entries:
 *Offset * Info * Type * Sym\. *Value * Sym\. *Name(?: * \+ * Addend)?
0+00 * 0+0(?:1|b|32|103) * R_[^ ]* *(?: * 55aa)?
#pass
