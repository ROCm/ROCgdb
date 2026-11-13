#as: -mno-relax
#objdump: -dr
#skip: loongarch32-*-*

.*:[    ]+file format .*


Disassembly of section .text:

00000000.* <.text>:
[ 	]+0:[ 	]+14000004[ 	]+lu12i.w[ 	]+\$a0,[ 	]+0
[ 	]+0:[ 	]+R_LARCH_TLS_LE_HI20_R[ 	]+TLSL1
[ 	]+4:[ 	]+001090a5[ 	]+add.d[ 	]+\$a1,[ 	]+\$a1,[ 	]+\$a0
[ 	]+4:[ 	]+R_LARCH_TLS_LE_ADD_R[ 	]+TLSL1
[ 	]+8:[ 	]+29800085[ 	]+st.w[ 	]+\$a1,[ 	]+\$a0,[ 	]+0
[ 	]+8:[ 	]+R_LARCH_TLS_LE_LO12_R[ 	]+TLSL1
[ 	]+c:[ 	]+14000004[ 	]+lu12i.w[ 	]+\$a0,[ 	]+0
[ 	]+c:[ 	]+R_LARCH_TLS_LE_HI20_R[ 	]+TLSL1\+0x8
[ 	]+10:[ 	]+001090a5[ 	]+add.d[ 	]+\$a1,[ 	]+\$a1,[ 	]+\$a0
[ 	]+10:[ 	]+R_LARCH_TLS_LE_ADD_R[ 	]+TLSL1\+0x8
[ 	]+14:[ 	]+29800085[ 	]+st.w[ 	]+\$a1,[ 	]+\$a0,[ 	]+0
[ 	]+14:[ 	]+R_LARCH_TLS_LE_LO12_R[ 	]+TLSL1\+0x8
