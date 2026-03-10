#name: input section is discarded because the output section match clauses succeed the matching DISCARD clause.
## Use case description:
## - sections .code.1, .code.2 and .code.3 (+ farcall stub) fit in .raml
## - section .code.4 fits in .ramz
## - nothing fits in .ramu
## The linker script contains a DISCARD clause for ".code.4" that precedes all
## the output clauses.
#source: non-contiguous-mem-1.s
#ld: --enable-non-contiguous-regions --enable-non-contiguous-regions-warnings -T non-contiguous-nok-5.ld
#error_output: non-contiguous-nok-5.err
