#as: -march=rv32ic -mabi=ilp32 --defsym exact_mode=1
#source: li32.s
#objdump: -dr

.*:     file format elf32-(little|big)riscv


Disassembly of section .text:

0+000 <target>:
[^:]+:[ 	]+00008537[ 	]+lui[ 	]+a0,0x8
[^:]+:[ 	]+00150513[ 	]+addi[ 	]+a0,a0,1 # .*
[^:]+:[ 	]+00002537[ 	]+lui[ 	]+a0,0x2
[^:]+:[ 	]+f0150513[ 	]+addi[ 	]+a0,a0,-255 # .*
[^:]+:[ 	]+12345537[ 	]+lui[ 	]+a0,0x12345
[^:]+:[ 	]+00150513[ 	]+addi[ 	]+a0,a0,1 # .*
[^:]+:[ 	]+f2345537[ 	]+lui[ 	]+a0,0xf2345
[^:]+:[ 	]+00150513[ 	]+addi[ 	]+a0,a0,1 # .*
