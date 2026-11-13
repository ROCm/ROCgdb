#name: PE BFD_RELOC_NONE handling
#ld: --enable-reloc-section
#objdump: -p
#source: reloc-none.s
#source: reloc-none2.s
#notarget: arm*-*-* ia64-*-* loongarch*-*-* riscv*-*-*
#xfail: aarch64-*-* i?86-*-* mcore-*-* sh-*-*

.*:     file format .*

#...
PE File Base Relocations.*
Virtual Address: .* Number of fixups 2
[ 	]*reloc    0 offset   10 .* (LOW|HIGHLOW|DIR64)
[ 	]*reloc    1 offset    0 .* ABSOLUTE
#pass
