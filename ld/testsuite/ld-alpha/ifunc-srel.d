#source: ifunc-srel.s
#ld: -melf64alpha
#error: \A[^\n]*: SREL16 relocation against STT_GNU_IFUNC symbol `global_ifunc' is not supported\n[^\n]*: SREL32 relocation against STT_GNU_IFUNC symbol `global_ifunc' is not supported\n[^\n]*: SREL64 relocation against STT_GNU_IFUNC symbol `global_ifunc' is not supported\n[^\n]*: final link failed[^\n]*\n?\Z
