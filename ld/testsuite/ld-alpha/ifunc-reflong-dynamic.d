#source: ifunc-reflong.s
#ld: -melf64alpha --export-dynamic
#error: \A[^\n]*: REFLONG relocation against STT_GNU_IFUNC symbol `global_ifunc' is not supported\n[^\n]*: REFLONG relocation against STT_GNU_IFUNC symbol `local_ifunc' is not supported\n[^\n]*: final link failed[^\n]*\n?\Z
