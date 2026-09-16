#source: ifunc-gprel32.s
#ld: -melf64alpha
#error: \A[^\n]*: GPREL32 relocation against STT_GNU_IFUNC symbol `global_ifunc' is not supported\n[^\n]*: final link failed[^\n]*\n?\Z
