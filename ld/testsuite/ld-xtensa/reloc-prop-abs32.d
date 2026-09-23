#source: reloc-prop-abs32.s
#ld: -T reloc-prop-abs32.t
#objdump: -s -j .xt.prop
#name: R_XTENSA_32_ABS property merge ignores relocated word

# The addends alone place the ranges at .text+0 and .text+4, so they are
# adjacent and must merge into one entry of size 8 at 0x1000.  Both byte
# orders are accepted.

.*: +file format .*xtensa.*

Contents of section \.xt\.prop:
 1008 (00001000 00000008 00000002|00100000 08000000 02000000).*
