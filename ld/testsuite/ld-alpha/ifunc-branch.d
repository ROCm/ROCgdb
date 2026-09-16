#source: ifunc-branch.s
#ld: -melf64alpha
#error: \A[^\n]*: BRADDR relocation against STT_GNU_IFUNC symbol `global_ifunc' is not supported\n[^\n]*: BRSGP relocation against STT_GNU_IFUNC symbol `global_ifunc' is not supported\n[^\n]*: final link failed[^\n]*\n?\Z
