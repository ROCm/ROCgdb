#name: local IFUNC symbol referenced only from a non-alloc note section
#ld: -Ttext=0x10000
#objdump: -s -j .note.probe
#...
Contents of section .note.probe:
 0000 (00000100 00000000|00000000 00010000) +.*
#pass
