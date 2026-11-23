#name: local IFUNC symbol referenced only from a debugging section
#as: --32
#ld: -Ttext=0x10000 -melf_i386
#objdump: -s -j .debug_info
#...
Contents of section .debug_info:
 0000 00000100 +.*
#pass
