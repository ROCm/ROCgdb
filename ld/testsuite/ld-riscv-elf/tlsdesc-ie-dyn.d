#source: tlsdesc-ie.s
#ld: -no-pie tmpdir/tlsdesc-lib.so
#readelf: -Wr

Relocation section '.rela.dyn' at offset 0x[0-9a-f]+ contains 1 entry:
 +Offset +Info +Type +Symbol's Value +Symbol's Name \+ Addend
[0-9a-f]+ +[0-9a-f]+ R_RISCV_TLS_TPREL64 +0+ sg2 \+ 0
