#source: ifunc-comdat-a.s
#source: ifunc-comdat-b.s
#ld: -melf64alpha
#error: \A`local_ifunc' referenced in section `\.data' of [^\n]*: defined in discarded section `\.text\.grp\[grp\]' of [^\n]*\n?\Z
