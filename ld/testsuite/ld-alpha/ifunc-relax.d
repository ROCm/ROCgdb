#source: ifunc-relax-a.s
#source: ifunc-relax-b.s
#ld: -relax -melf64alpha --unresolved-symbols=ignore-all
#readelf: -Wr

# The two objects' got subsections cannot be merged until relaxation has
# shrunk them, and the merge leaves one entry for the IFUNC where the first
# sizing counted two.  One IRELATIVE for that entry and one for ptr; a third
# would mean .rela.iplt was not sized again after the got was.
Relocation section '\.rela\.dyn' .* contains 2 entries:
#...
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
#pass
