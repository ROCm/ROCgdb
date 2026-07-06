#as: -march=rv64gc_zvqwdota8i_zvqwdota16i_zvfwdota16bf_zvfqwdota8f
#objdump: -dr

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+9a8f8877[ 	]+vqwdotau.vv[ 	]+v16,v8,v31
[ 	]+[0-9a-f]+:[ 	]+9e8f8877[ 	]+vqwdotas.vv[ 	]+v16,v8,v31
[ 	]+[0-9a-f]+:[ 	]+928f9877[ 	]+vfwdota.vv[ 	]+v16,v8,v31
[ 	]+[0-9a-f]+:[ 	]+9a8f9877[ 	]+vfqwdota.vv[ 	]+v16,v8,v31
[ 	]+[0-9a-f]+:[ 	]+9e8f9877[ 	]+vfqwdota.alt.vv[ 	]+v16,v8,v31
[ 	]+[0-9a-f]+:[ 	]+988f8877[ 	]+vqwdotau.vv[ 	]+v16,v8,v31,v0.t
[ 	]+[0-9a-f]+:[ 	]+9c8f8877[ 	]+vqwdotas.vv[ 	]+v16,v8,v31,v0.t
[ 	]+[0-9a-f]+:[ 	]+908f9877[ 	]+vfwdota.vv[ 	]+v16,v8,v31,v0.t
[ 	]+[0-9a-f]+:[ 	]+980f9877[ 	]+vfqwdota.vv[ 	]+v16,v0,v31,v0.t
[ 	]+[0-9a-f]+:[ 	]+9c0f9877[ 	]+vfqwdota.alt.vv[ 	]+v16,v0,v31,v0.t
