#source: ifunc-gc.s
#ld: -melf64alpha --gc-sections -e _start
#readelf: -Wr

Relocation section '\.rela\.dyn' .* contains 1 entry:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_IRELATIVE +[0-9a-f]+
#pass
