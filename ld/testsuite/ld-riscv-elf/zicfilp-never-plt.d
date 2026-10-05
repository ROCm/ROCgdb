#name: Default PLT with -z zicfilp=never
#source: zicfilp-unlabeled-plt.s
#ld: -shared -melf64lriscv -z zicfilp=never
#objdump: -dr -j .plt
#as: -march=rv64gc_zicfilp -mlittle-endian

[^:]*: *file format elf64-.*riscv

Disassembly of section \.plt:

[0-9a-f]+ <\.plt>:
.*:[ 	]+[0-9a-f]+[ 	]+auipc[ 	]+t2,0x[0-9a-f]+
#...
[0-9a-f]+ <foo@plt>:
.*:[ 	]+[0-9a-f]+[ 	]+auipc[ 	]+t3,0x[0-9a-f]+
.*:[ 	]+[0-9a-f]+[ 	]+ld[ 	]+t3,[0-9]+\(t3\) # [0-9a-f]+ <foo>
.*:[ 	]+[0-9a-f]+[ 	]+jalr[ 	]+t1,t3
.*:[ 	]+[0-9a-f]+[ 	]+nop
#pass
