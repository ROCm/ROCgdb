#name: all code sections fit in the available memories BUT farcall stub to jump to code4 DOES NOT not fits in .ramu
## Use case description:
## - sections .code.1 and .code.2 fit in .raml
## - section .code.3 fits in .ramu, but not its farcall stub to jump to code4.
## - section .code.4 fits in .ramz
#source: non-contiguous-mem-1.s
#ld: --enable-non-contiguous-regions -T non-contiguous-nok-2.ld
# error: Memory region `RAMU' not large enough for the linker-created stubs section `\.code\.3\.stub' associated to output section `\.ramu'
