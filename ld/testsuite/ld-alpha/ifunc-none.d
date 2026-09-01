#source: ifunc-none.s
#ld: -melf64alpha
#readelf: -Wr

# A link with nothing to resolve at startup gets no relocations at all, and
# in particular no .rela.iplt.
There are no relocations in this file\.
