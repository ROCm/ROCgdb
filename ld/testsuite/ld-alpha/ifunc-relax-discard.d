#source: ifunc-relax-discard.s
#source: ifunc-relax-b.s
#ld: -relax -melf64alpha --unresolved-symbols=ignore-all --no-warn-rwx-segments -T $srcdir/$subdir/ifunc-discard.ld
#readelf: -Wr

# The got entry that survives the merge during relaxation is the one whose
# only reference the linker script discarded, so it carries no mark of its
# own.  It has to take the one from the entry it absorbs, or nothing is
# reserved for the reference that does survive.
Relocation section '\.rela\.dyn' .* contains 2 entries:
#...
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
#pass
