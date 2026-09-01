#source: ifunc-extern.s
#ld: -melf64alpha tmpdir/libalphaifunc.so
#readelf: -Wr

# The IFUNC is defined in another module, so the dynamic linker resolves
# it: an ordinary PLT entry and a symbolic data relocation, no IRELATIVE.
Relocation section '\.rela\.dyn' .* contains 1 entry:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_REFQUAD .*shlib_ifunc \+ 0
#...
Relocation section '\.rela\.plt' .* contains 1 entry:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_JMP_SLOT .*shlib_ifunc \+ 0
#pass
