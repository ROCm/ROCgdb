#source: ifunc-global.s
#ld: -pie -melf64alpha
#readelf: -Wr

Relocation section '\.rela\.dyn' .* contains 2 entries:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_IRELATIVE +[0-9a-f]+
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_IRELATIVE +[0-9a-f]+
#pass
