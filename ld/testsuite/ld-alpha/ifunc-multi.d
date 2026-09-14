#source: ifunc-multi-a.s
#source: ifunc-multi-b.s
#ld: -melf64alpha
#readelf: -Wr

# One IRELATIVE for the merged GOT entry and one for each of the two data
# references.
Relocation section '\.rela\.dyn' .* contains 3 entries:
#...
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
#pass
