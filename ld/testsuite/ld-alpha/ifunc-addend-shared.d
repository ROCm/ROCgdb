#source: ifunc-addend-refquad.s
#ld: -shared -melf64alpha
#readelf: -Wr

# The IFUNC is preemptible here, so the dynamic linker resolves it and the
# relocation keeps its addend.  Nothing turns into an IRELATIVE, so the
# addend is not rejected.
Relocation section '\.rela\.dyn' .* contains 1 entry:
#...
[0-9a-f]+ +[0-9a-f]+ +R_ALPHA_REFQUAD .*global_ifunc \+ 4
#pass
