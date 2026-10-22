#source: csr-insns-pseudo.s
#as: -march=rv32if
#objdump: -dr -Mno-aliases

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <pseudo_csr_insn>:
[ 	]+[0-9a-f]+:[ 	]+340022f3[ 	]+csrrs[ 	]+t0,mscratch,zero
[ 	]+[0-9a-f]+:[ 	]+34029073[ 	]+csrrw[ 	]+zero,mscratch,t0
[ 	]+[0-9a-f]+:[ 	]+3402a073[ 	]+csrrs[ 	]+zero,mscratch,t0
[ 	]+[0-9a-f]+:[ 	]+3402b073[ 	]+csrrc[ 	]+zero,mscratch,t0
[ 	]+[0-9a-f]+:[ 	]+340fd073[ 	]+csrrwi[ 	]+zero,mscratch,31
[ 	]+[0-9a-f]+:[ 	]+340fe073[ 	]+csrrsi[ 	]+zero,mscratch,31
[ 	]+[0-9a-f]+:[ 	]+340ff073[ 	]+csrrci[ 	]+zero,mscratch,31
[ 	]+[0-9a-f]+:[ 	]+c00022f3[ 	]+csrrs[ 	]+t0,cycle,zero
[ 	]+[0-9a-f]+:[ 	]+c01022f3[ 	]+csrrs[ 	]+t0,time,zero
[ 	]+[0-9a-f]+:[ 	]+c02022f3[ 	]+csrrs[ 	]+t0,instret,zero
[ 	]+[0-9a-f]+:[ 	]+c80022f3[ 	]+csrrs[ 	]+t0,cycleh,zero
[ 	]+[0-9a-f]+:[ 	]+c81022f3[ 	]+csrrs[ 	]+t0,timeh,zero
[ 	]+[0-9a-f]+:[ 	]+c82022f3[ 	]+csrrs[ 	]+t0,instreth,zero
[ 	]+[0-9a-f]+:[ 	]+003022f3[ 	]+csrrs[ 	]+t0,fcsr,zero
[ 	]+[0-9a-f]+:[ 	]+003392f3[ 	]+csrrw[ 	]+t0,fcsr,t2
[ 	]+[0-9a-f]+:[ 	]+00339073[ 	]+csrrw[ 	]+zero,fcsr,t2
[ 	]+[0-9a-f]+:[ 	]+002022f3[ 	]+csrrs[ 	]+t0,frm,zero
[ 	]+[0-9a-f]+:[ 	]+002312f3[ 	]+csrrw[ 	]+t0,frm,t1
[ 	]+[0-9a-f]+:[ 	]+00231073[ 	]+csrrw[ 	]+zero,frm,t1
[ 	]+[0-9a-f]+:[ 	]+002fd2f3[ 	]+csrrwi[ 	]+t0,frm,31
[ 	]+[0-9a-f]+:[ 	]+002fd073[ 	]+csrrwi[ 	]+zero,frm,31
[ 	]+[0-9a-f]+:[ 	]+001022f3[ 	]+csrrs[ 	]+t0,fflags,zero
[ 	]+[0-9a-f]+:[ 	]+001312f3[ 	]+csrrw[ 	]+t0,fflags,t1
[ 	]+[0-9a-f]+:[ 	]+00131073[ 	]+csrrw[ 	]+zero,fflags,t1
[ 	]+[0-9a-f]+:[ 	]+001fd2f3[ 	]+csrrwi[ 	]+t0,fflags,31
[ 	]+[0-9a-f]+:[ 	]+001fd073[ 	]+csrrwi[ 	]+zero,fflags,31
