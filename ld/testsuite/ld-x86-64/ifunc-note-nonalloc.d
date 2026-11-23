#name: local IFUNC symbol referenced only from a non-alloc note section
#as: --64
#ld: -Ttext=0x10000 -melf_x86_64
#objdump: -s -j .note.probe
#...
Contents of section .note.probe:
 0000 00000100 00000000 +.*
#pass
