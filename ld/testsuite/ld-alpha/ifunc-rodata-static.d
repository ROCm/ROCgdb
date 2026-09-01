#source: ifunc-rodata.s
#ld: -melf64alpha
#error: \A[^\n]*: address of STT_GNU_IFUNC symbol `global_ifunc' in read-only section `\.rodata' cannot be relocated in a static link\n[^\n]*: address of STT_GNU_IFUNC symbol `local_ifunc' in read-only section `\.rodata' cannot be relocated in a static link\n[^\n]*: final link failed[^\n]*\n?\Z
