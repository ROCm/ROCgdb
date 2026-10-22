#as: -march=rv32i_zicsr_sspmpen -mcsr-check --fatal-warnings
#objdump: -dr
#source: spmpen-rv32.s
.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <.text>:
[ 	]+[0-9a-f]+:[ 	]+18302573[ 	]+csrr[ 	]+a0,spmpen
[ 	]+[0-9a-f]+:[ 	]+193025f3[ 	]+csrr[ 	]+a1,spmpenh
