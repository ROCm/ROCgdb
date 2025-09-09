#name: non-contiguous-powerpc64
#source: non-contiguous-powerpc.s
#as: -a64
#ld: -melf64ppc --enable-non-contiguous-regions -T non-contiguous-powerpc.ld
#error: Memory region `one' not large enough for the linker-created stubs section `\.text\.one\.stub' associated to output section `one'
