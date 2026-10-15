#source: relr-addend.s
#as: -march=rv64i -mabi=lp64
#ld: -m[riscv_choose_lp64_emul] -shared -z pack-relative-relocs -T relr-relocs.ld
#objdump: -s -j .got -j .data

.*:[ 	]+file format .*

Contents of section \.got:
 20000 [0-9a-f]+ [0-9a-f]+ 00003412 00000000  .*
 20010 08003412 00000000 [0-9a-f]+ [0-9a-f]+  .*
#...
Contents of section \.data:
 12340000 14451100 00000000 10989101 00000000  .*
 12340010 00003412 00000000 08003412 00000000  .*
