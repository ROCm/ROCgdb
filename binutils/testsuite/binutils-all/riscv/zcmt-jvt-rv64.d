#name: RISC-V Zcmt RV64 manually constructed JVT
#source: zcmt-jvt.s
#as: -march=rv64i_zca_zicsr_zcmt -mabi=lp64 -mlittle-endian --defsym entry_size=8
#ld: -m elf64lriscv --no-relax -e foo -T zcmt-jvt.ld
#objdump: -d -j .text -j .riscv.jvt

.*:     file format elf64-littleriscv

Disassembly of section \.text:

0*1000 <foo>:
[ \t]+1000:[ \t]+a002[ \t]+cm\.jt[ \t]+0[ \t]*$

0*1002 <bar>:
[ \t]+1002:[ \t]+a082[ \t]+cm\.jalt[ \t]+32[ \t]*$

Disassembly of section \.riscv\.jvt:

0*2000 <[^>]+>:
[ \t]+2000:[ \t]+0000000000001000[ \t]+jvt\.jt\[0\]:[ \t]+foo
#...
[ \t]+2100:[ \t]+0000000000001002[ \t]+jvt\.jalt\[32\]:[ \t]+bar
