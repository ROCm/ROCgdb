#source: ifunc-discard.s
#ld: -melf64alpha --no-warn-rwx-segments -T $srcdir/$subdir/ifunc-discard.ld
#readelf: -Wr

# Only the reference that survives the script reserves anything.  A slot
# reserved for one the script discarded would be left as an R_ALPHA_NONE,
# which libc's startup code would try to apply.
Relocation section '\.rela\.dyn' .* contains 1 entry:
#...
[0-9a-f]+ +0+2a R_ALPHA_IRELATIVE +[0-9a-f]+
#pass
