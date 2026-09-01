#source: ifunc-rodata.s
#ld: -melf64alpha -z notext tmpdir/libalphaifunc.so
#readelf: -Wrd

#...
 +0x0+16 +\(TEXTREL\) +0x0
#...
Relocation section '\.rela\.dyn' .* contains 2 entries:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_IRELATIVE +[0-9a-f]+
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_IRELATIVE +[0-9a-f]+
#pass
