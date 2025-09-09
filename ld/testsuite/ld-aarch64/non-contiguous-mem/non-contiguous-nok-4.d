#name: Discard 2nd interrupt vector table because of alignment requirements
## Use case description:
## 3 IVTs are defined in section .vectors (2KB):
## - vector_table1 and vector_table3 in the first file, and vector_table3 does
##   not redefine the section, so is directly appended after vector_table1.
## - vector_table2 is defined in a second file, and redefines the section
##   .vectors with an alignment of 2KB.
## Since all .vectors sections should be moved to the VECTORS memory (2KB),
## vector_table2 will be discarded as the space was taken by the padding of
## the section defined in the first file.
#source: non-contiguous-mem-3.s
#source: non-contiguous-mem-3-ivt.s
#ld: -T non-contiguous-nok-4.ld --enable-non-contiguous-regions
#error: --enable-non-contiguous-regions was not able to allocate the input section `\.vectors' \(.*non-contiguous-mem-3-ivt\.o\) to an output section
#error: .*: final link failed$
