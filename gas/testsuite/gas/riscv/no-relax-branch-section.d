#as: -march=rv64ic -mno-relax
#objdump: -dr -Mno-aliases

.*:[ 	]+file format .*


Disassembly of section .text:

0+0 <.text>:
[ 	]+0:[ 	]+[0-9a-f]+[ 	]+beq[ 	]+.*
[ 	]+0:[ 	]+R_RISCV_BRANCH[ 	]+.Lother
[ 	]+4:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+.*
[ 	]+4:[ 	]+R_RISCV_JAL[ 	]+.Lother
[ 	]+8:[ 	]+[0-9a-f]+[ 	]+c.beqz[ 	]+.*
[ 	]+8:[ 	]+R_RISCV_RVC_BRANCH[ 	]+.Lother
[ 	]+a:[ 	]+[0-9a-f]+[ 	]+c.j[ 	]+.*
[ 	]+a:[ 	]+R_RISCV_RVC_JUMP[ 	]+.Lother
[ 	]+c:[ 	]+00b50663[ 	]+beq[ 	]+a0,a1,18 <.text\+0x18>
[ 	]+10:[ 	]+008000ef[ 	]+jal[ 	]+ra,18 <.text\+0x18>
[ 	]+14:[ 	]+c111[ 	]+c.beqz[ 	]+a0,18 <.text\+0x18>
[ 	]+16:[ 	]+a009[ 	]+c.j[ 	]+18 <.text\+0x18>
[ 	]+18:[ 	]+00008067[ 	]+jalr[ 	]+zero,0\(ra\)
#pass
