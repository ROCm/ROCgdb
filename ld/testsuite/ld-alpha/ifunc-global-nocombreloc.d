#source: ifunc-global.s
#ld: -melf64alpha -z nocombreloc
#readelf: -Wr

Relocation section '\.rela\.iplt' .* contains 2 entries:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_IRELATIVE +[0-9a-f]+
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_IRELATIVE +[0-9a-f]+
#pass
