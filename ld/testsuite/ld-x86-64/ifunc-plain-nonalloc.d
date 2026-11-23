#name: local IFUNC symbol referenced only from a plain non-alloc section
#as: --64
#ld: -Ttext=0x10000 -melf_x86_64
#objdump: -s -j .probe
#...
Contents of section .probe:
 0000 00000100 00000000 +.*
#pass
