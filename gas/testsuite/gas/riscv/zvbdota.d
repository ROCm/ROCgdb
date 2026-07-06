#as: -march=rv64gc_zvqwbdota8i_zvqwbdota16i_zvfwbdota16bf_zvfqwbdota8f_zvfbdota32f
#objdump: -dr

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+ba8f8877[ 	]+vqwbdotau.vv[ 	]+v16,v8,v31,0
[ 	]+[0-9a-f]+:[ 	]+ba9f8877[ 	]+vqwbdotau.vv[ 	]+v16,v8,v31,8
[ 	]+[0-9a-f]+:[ 	]+be8f8877[ 	]+vqwbdotas.vv[ 	]+v16,v8,v31,0
[ 	]+[0-9a-f]+:[ 	]+be9f8877[ 	]+vqwbdotas.vv[ 	]+v16,v8,v31,8
[ 	]+[0-9a-f]+:[ 	]+b28f9877[ 	]+vfwbdota.vv[ 	]+v16,v8,v31,0
[ 	]+[0-9a-f]+:[ 	]+b29f9877[ 	]+vfwbdota.vv[ 	]+v16,v8,v31,8
[ 	]+[0-9a-f]+:[ 	]+ba8f9877[ 	]+vfqwbdota.vv[ 	]+v16,v8,v31,0
[ 	]+[0-9a-f]+:[ 	]+ba9f9877[ 	]+vfqwbdota.vv[ 	]+v16,v8,v31,8
[ 	]+[0-9a-f]+:[ 	]+be8f9877[ 	]+vfqwbdota.alt.vv[ 	]+v16,v8,v31,0
[ 	]+[0-9a-f]+:[ 	]+be9f9877[ 	]+vfqwbdota.alt.vv[ 	]+v16,v8,v31,8
[ 	]+[0-9a-f]+:[ 	]+ae8f9877[ 	]+vfbdota.vv[ 	]+v16,v8,v31,0
[ 	]+[0-9a-f]+:[ 	]+ae9f9877[ 	]+vfbdota.vv[ 	]+v16,v8,v31,8
[ 	]+[0-9a-f]+:[ 	]+bc8f8877[ 	]+vqwbdotas.vv[ 	]+v16,v8,v31,0,v0.t
[ 	]+[0-9a-f]+:[ 	]+b89f9877[ 	]+vfqwbdota.vv[ 	]+v16,v8,v31,8,v0.t
