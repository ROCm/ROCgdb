#source: ifunc-addend-local.s
#ld: -shared -melf64alpha
#error: \A[^\n]*: REFQUAD relocation against STT_GNU_IFUNC symbol `local_ifunc' has a non-zero addend\n[^\n]*: final link failed[^\n]*\n?\Z
