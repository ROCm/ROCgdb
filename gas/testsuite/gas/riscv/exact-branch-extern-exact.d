#as: -march=rv32ic --defsym exact_mode=1
#source: exact-branch-extern.s
#objdump: -drw -Mno-aliases

.*:     file format .*


Disassembly of section \.text:

0+ <\.text>:
[ 	]+[0-9a-f]+:[ 	]+00b50063[ 	]+beq[ 	]+a0,a1,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+feb51ee3[ 	]+bne[ 	]+a0,a1,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+feb54ce3[ 	]+blt[ 	]+a0,a1,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+feb55ae3[ 	]+bge[ 	]+a0,a1,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+feb568e3[ 	]+bltu[ 	]+a0,a1,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+feb576e3[ 	]+bgeu[ 	]+a0,a1,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+d565[ 	]+c\.beqz[ 	]+a0,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_RVC_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+f17d[ 	]+c\.bnez[ 	]+a0,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_RVC_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+fea5c2e3[ 	]+blt[ 	]+a1,a0,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+fea5d0e3[ 	]+bge[ 	]+a1,a0,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+fca5eee3[ 	]+bltu[ 	]+a1,a0,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+fca5fce3[ 	]+bgeu[ 	]+a1,a0,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+fc050ae3[ 	]+beq[ 	]+a0,zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+fc0518e3[ 	]+bne[ 	]+a0,zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+fc0546e3[ 	]+blt[ 	]+a0,zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+fca044e3[ 	]+blt[ 	]+zero,a0,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+fca052e3[ 	]+bge[ 	]+zero,a0,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+fc0550e3[ 	]+bge[ 	]+a0,zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_BRANCH[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+0001[ 	]+.*
[ 	]+[0-9a-f]+:[ 	]+fbbff56f[ 	]+jal[ 	]+a0,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+bf5d[ 	]+c\.j[ 	]+[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_RVC_JUMP[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+3f55[ 	]+c\.jal[ 	]+[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_RVC_JUMP[ 	]+ext
