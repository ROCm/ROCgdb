#name: non-contiguous-arm4
#source: non-contiguous-arm.s
#ld: --enable-non-contiguous-regions -T non-contiguous-arm4.ld
# error: Memory region `RAMU' not large enough for the linker-created stubs section `\.code\.3\.__stub' associated to output section `\.ramu'
