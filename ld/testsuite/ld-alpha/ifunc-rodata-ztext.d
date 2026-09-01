#source: ifunc-rodata.s
#ld: -melf64alpha -z text tmpdir/libalphaifunc.so
#readelf: -Wr
#error: \A[^\n]*: read-only segment has dynamic relocations\n?\Z
