#name: local IFUNC symbol referenced only from a debugging section
#ld: -Ttext=0x10000
#objdump: -s -j .debug_info
#...
Contents of section .debug_info:
 0000 00000100 00000000 +.*
#pass
