#source: ifunc-global.s
#ld: -melf64alpha
#readelf: -Wrs

# One IRELATIVE for the GOT entry and one for ptr.  The symbol table
# entries confirm that the resolver kept its IFUNC type.
Relocation section '\.rela\.dyn' .* contains 2 entries:
#...
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
#...
Symbol table '\.symtab' contains .* entries:
#...
 +[0-9]+: [0-9a-f]+ +0 IFUNC +GLOBAL +DEFAULT +[0-9]+ global_ifunc
#...
 +[0-9]+: [0-9a-f]+ +0 NOTYPE +GLOBAL +DEFAULT +[0-9]+ ptr
#pass
