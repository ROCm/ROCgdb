#name: last section .code.4 is too big to fit in any of the available memories.
## Use case description:
## - sections .code.1 and .code.2 fit in .raml
## - section .code.3 fits in .ramu
## - section .code.4 is too large to fit in .ramz, we should expect an error
##   message about inability to assign the input section to an output one.
#source: non-contiguous-mem-1.s
#ld: --enable-non-contiguous-regions -T non-contiguous-nok-1.ld
# error: .*Could not assign .?\.code\.4.? to an output section. Retry without --enable-non-contiguous-regions\.
