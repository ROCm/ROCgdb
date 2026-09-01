#source: ifunc-addend-literal.s
#ld: -melf64alpha
#error: \A[^\n]*: ELF_LITERAL relocation against STT_GNU_IFUNC symbol `global_ifunc' has a non-zero addend\n[^\n]*: final link failed[^\n]*\n?\Z
