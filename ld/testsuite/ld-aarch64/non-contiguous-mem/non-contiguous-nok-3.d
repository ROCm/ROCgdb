#name: .bss section bigger than all the available RAMs.
## Use case description:
## The section .bss.MY_BUF (100KB) won't fit in RAM1 (64KB) or RAM2 (96KB)
#source: non-contiguous-mem-2.s
#ld: -T non-contiguous-nok-3.ld --enable-non-contiguous-regions --enable-non-contiguous-regions-warnings
#error_output: non-contiguous-nok-3.err
