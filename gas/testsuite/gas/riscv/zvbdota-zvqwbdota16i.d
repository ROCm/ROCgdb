#as: -march=rv64gc_zvqwbdota16i
#objdump: -dr

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+ba8f8877[ 	]+vqwbdotau.vv[ 	]+v16,v8,v31,0
[ 	]+[0-9a-f]+:[ 	]+be9f8877[ 	]+vqwbdotas.vv[ 	]+v16,v8,v31,8
