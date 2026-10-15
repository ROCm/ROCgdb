#source: tlsdesc-ie-32.s
#as: -march=rv32i -mabi=ilp32
#ld: -melf32lriscv -no-pie tmpdir/tlsdesc-lib32.so
#readelf: -Wr

Relocation section '.rela.dyn' at offset 0x[0-9a-f]+ contains 1 entry:
 +Offset +Info +Type +Sym. Value +Symbol's Name \+ Addend
[0-9a-f]+ +[0-9a-f]+ R_RISCV_TLS_TPREL32 +0+ +sg2 \+ 0
