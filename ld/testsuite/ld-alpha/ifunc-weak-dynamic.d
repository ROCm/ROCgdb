#source: ifunc-weak.s
#ld: -melf64alpha tmpdir/libalphaifunc.so
#readelf: -Wr

Relocation section '\.rela\.dyn' .* contains 2 entries:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_IRELATIVE +[0-9a-f]+
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_IRELATIVE +[0-9a-f]+
#pass
