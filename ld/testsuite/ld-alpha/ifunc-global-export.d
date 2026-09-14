#source: ifunc-global.s
#ld: -melf64alpha --export-dynamic tmpdir/libalphaifunc.so
#readelf: -Wr

# One IRELATIVE for the GOT entry and one for ptr, and nothing else.  The
# IFUNC is in .dynsym but is not preemptible in an executable, so the link
# resolves it itself and no PLT entry is left over from the guess that
# check_relocs makes about one.
Relocation section '\.rela\.dyn' .* contains 2 entries:
 +Offset +Info +Type +Symbol's Value +Symbol's Name \+ Addend
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
