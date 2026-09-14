#source: ifunc-global.s
#ld: -shared -melf64alpha
#readelf: -Wr

Relocation section '\.rela\.dyn' .* contains 1 entry:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_REFQUAD .*global_ifunc \+ 0
#...
Relocation section '\.rela\.plt' .* contains 1 entry:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_JMP_SLOT .*global_ifunc \+ 0
#pass
