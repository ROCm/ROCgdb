#source: ifunc-gprel.s
#ld: -melf64alpha
#error: \A[^\n]*: GPRELHIGH relocation against STT_GNU_IFUNC symbol `global_ifunc' is not supported\n[^\n]*: GPRELLOW relocation against STT_GNU_IFUNC symbol `global_ifunc' is not supported\n[^\n]*: GPREL16 relocation against STT_GNU_IFUNC symbol `global_ifunc' is not supported\n[^\n]*: final link failed[^\n]*\n?\Z
