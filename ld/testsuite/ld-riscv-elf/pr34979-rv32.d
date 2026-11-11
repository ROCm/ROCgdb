#source: pr34979.s
#as: -march=rv32i -mabi=ilp32 -mrelax
#ld: -m[riscv_choose_ilp32_emul] -pie --relax
#objdump: -d -Mno-aliases

.*:[ 	]+file format .*

Disassembly of section \.text:

0+[0-9a-f]+ <_start>:
[ 	]+[0-9a-f]+:[ 	]+[0-9a-f]+[ 	]+auipc[ 	]+ra,0x100
[ 	]+[0-9a-f]+:[ 	]+[0-9a-f]+[ 	]+jalr[ 	]+ra,4\(ra\) # [0-9a-f]+ <target>
	\.\.\.

Disassembly of section __lcxx_override:

0+[0-9a-f]+ <target>:
[ 	]+[0-9a-f]+:[ 	]+[0-9a-f]+[ 	]+jalr[ 	]+zero,0\(ra\)
