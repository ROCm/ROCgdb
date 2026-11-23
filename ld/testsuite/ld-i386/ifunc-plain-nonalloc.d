#name: local IFUNC symbol referenced only from a plain non-alloc section
#as: --32
#ld: -Ttext=0x10000 -melf_i386
#objdump: -s -j .probe
#...
Contents of section .probe:
 0000 00000100 +.*
#pass
