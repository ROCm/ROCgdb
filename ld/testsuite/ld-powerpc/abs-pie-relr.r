#source: abs-reloc.s
#as: -a64
#ld: -melf64ppc -pie --hash-style=sysv -z pack-relative-relocs --defsym a=1 --defsym 'HIDDEN(b=2)' --defsym c=0x123456789abcdef0
#readelf: -rW

Relocation section '\.relr\.dyn' at offset .* contains 1 entry:
Index: Entry            Address           Symbolic Address
0000: +[0-9a-f]+ [0-9a-f]+ +x
