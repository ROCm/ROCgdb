#name: local IFUNC symbol referenced only from a non-alloc note section
#as: --32
#ld: -Ttext=0x10000 -melf_i386
#objdump: -s -j .note.probe
#...
Contents of section .note.probe:
 0000 00000100 +.*
#pass
