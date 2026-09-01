#source: ifunc-ehframe-a.s
#source: ifunc-ehframe-b.s
#ld: -melf64alpha tmpdir/libalphaifunc.so
#error: \A[^\n]*: cannot resolve STT_GNU_IFUNC symbol `global_ifunc': the place of its REFQUAD relocation in `\.eh_frame' was deleted\n[^\n]*: final link failed[^\n]*\n?\Z
