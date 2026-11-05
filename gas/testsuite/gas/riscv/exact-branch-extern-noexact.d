#as: -march=rv32ic
#source: exact-branch-extern.s
#objdump: -drw -Mno-aliases

.*:     file format .*


Disassembly of section \.text:

0+ <\.text>:
[ 	]+[0-9a-f]+:[ 	]+00b51463[ 	]+bne[ 	]+a0,a1,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+ffdff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00b50463[ 	]+beq[ 	]+a0,a1,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+ff5ff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00b55463[ 	]+bge[ 	]+a0,a1,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+fedff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00b54463[ 	]+blt[ 	]+a0,a1,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+fe5ff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00b57463[ 	]+bgeu[ 	]+a0,a1,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+fddff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00b56463[ 	]+bltu[ 	]+a0,a1,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+fd5ff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00051463[ 	]+bne[ 	]+a0,zero,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+fcdff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00050463[ 	]+beq[ 	]+a0,zero,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+fc5ff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00a5d463[ 	]+bge[ 	]+a1,a0,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+fbdff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00a5c463[ 	]+blt[ 	]+a1,a0,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+fb5ff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00a5f463[ 	]+bgeu[ 	]+a1,a0,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+fadff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00a5e463[ 	]+bltu[ 	]+a1,a0,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+fa5ff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00051463[ 	]+bne[ 	]+a0,zero,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+f9dff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00050463[ 	]+beq[ 	]+a0,zero,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+f95ff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00055463[ 	]+bge[ 	]+a0,zero,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+f8dff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00a05463[ 	]+bge[ 	]+zero,a0,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+f85ff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00a04463[ 	]+blt[ 	]+zero,a0,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+f7dff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+00054463[ 	]+blt[ 	]+a0,zero,[0-9a-f]+.*
[ 	]+[0-9a-f]+:[ 	]+f75ff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+0001[ 	]+.*
[ 	]+[0-9a-f]+:[ 	]+f6fff56f[ 	]+jal[ 	]+a0,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+f6bff06f[ 	]+jal[ 	]+zero,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
[ 	]+[0-9a-f]+:[ 	]+f67ff0ef[ 	]+jal[ 	]+ra,[0-9a-f]+.*[0-9a-f]+:[ 	]+R_RISCV_JAL[ 	]+ext
