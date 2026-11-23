#name: local IFUNC symbol referenced only from a plain non-alloc section
#ld: -Ttext=0x10000
#objdump: -s -j .probe
#...
Contents of section .probe:
 0000 (00000100 00000000|00000000 00010000) +.*
#pass
