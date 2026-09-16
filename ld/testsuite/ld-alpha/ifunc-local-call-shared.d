#source: ifunc-local-call.s
#ld: -shared -melf64alpha
#readelf: -Wr

Relocation section '\.rela\.dyn' .* contains 1 entry:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_IRELATIVE +[0-9a-f]+
#pass
