#as: -march=generic64+ace_v1
#objdump: -dw
#name: 64-bit ACE v1 insns
#source: x86-64-ace_v1.s

.*: +file format .*

Disassembly of section \.text:

[0-9a-f]+ <start>:
[ 	]*[a-f0-9]+:[ 	]*c4 e2 7b 49 c0[ 	]+tilezero %tmm0
[ 	]*[a-f0-9]+:[ 	]*c4 e2 78 49 01[ 	]+ldtilecfg \(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e2 79 49 01[ 	]+sttilecfg \(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e2 78 49 c0[ 	]+tilerelease
[ 	]*[a-f0-9]+:[ 	]*62 f2 65 48 4a ca[ 	]+tilemovrow %ebx,%tmm2,%zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f3 7d 48 07 ca 08[ 	]+tilemovrow \$0x8,%tmm2,%zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 e5 48 4a ca[ 	]+tilemovrow %ebx,%zmm2,%tmm1
[ 	]*[a-f0-9]+:[ 	]*62 f3 fd 48 07 ca 08[ 	]+tilemovrow \$0x8,%zmm2,%tmm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 e5 48 4b ca[ 	]+tilemovcol %ebx,%zmm2,%tmm1
[ 	]*[a-f0-9]+:[ 	]*62 f3 fd 48 2f ca 08[ 	]+tilemovcol \$0x8,%zmm2,%tmm1
[ 	]*[a-f0-9]+:[ 	]*62 62 6e 48 4a f5[ 	]+tcvtrowd2ps %edx,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 7e 48 07 f5 08[ 	]+tcvtrowd2ps \$0x8,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 62 6f 48 6d f5[ 	]+tcvtrowps2bf16h %edx,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 7f 48 07 f5 08[ 	]+tcvtrowps2bf16h \$0x8,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 62 6c 48 6d f5[ 	]+tcvtrowps2phh %edx,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 7c 48 07 f5 08[ 	]+tcvtrowps2phh \$0x8,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 d6 ff 48 95 c2[ 	]+bsrmovh %zmm10,%bsr0
[ 	]*[a-f0-9]+:[ 	]*62 f6 ff 48 95 41 7f[ 	]+bsrmovh 0x1fc0\(%rcx\),%bsr0
[ 	]*[a-f0-9]+:[ 	]*62 f6 7f 48 95 c4[ 	]+bsrmovh %bsr0,%zmm4
[ 	]*[a-f0-9]+:[ 	]*62 f6 7f 48 95 41 80[ 	]+bsrmovh %bsr0,-0x2000\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*62 62 6e 48 6d f5[ 	]+tcvtrowps2bf16l %edx,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 7e 48 77 f5 08[ 	]+tcvtrowps2bf16l \$0x8,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 62 6d 48 6d f5[ 	]+tcvtrowps2phl %edx,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 7f 48 77 f5 08[ 	]+tcvtrowps2phl \$0x8,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 d6 fe 48 95 c2[ 	]+bsrmovl %zmm10,%bsr0
[ 	]*[a-f0-9]+:[ 	]*62 f6 fe 48 95 41 7f[ 	]+bsrmovl 0x1fc0\(%rcx\),%bsr0
[ 	]*[a-f0-9]+:[ 	]*62 f6 7e 48 95 c4[ 	]+bsrmovl %bsr0,%zmm4
[ 	]*[a-f0-9]+:[ 	]*62 f6 7e 48 95 41 80[ 	]+bsrmovl %bsr0,-0x2000\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e2 fb 49 c0[ 	]+bsrinit %bsr0
[ 	]*[a-f0-9]+:[ 	]*62 f6 f4 48 95 c3[ 	]+bsrmovf %zmm3,%zmm1,%bsr0
[ 	]*[a-f0-9]+:[ 	]*62 f6 f4 48 95 41 7f[ 	]+bsrmovf 0x1fc0\(%rcx\),%zmm1,%bsr0
[ 	]*[a-f0-9]+:[ 	]*62 f3 6c 48 8d c9 00[ 	]+top4mxbf8ps \$0x0,%zmm2,%zmm1,%tmm1
[ 	]*[a-f0-9]+:[ 	]*62 f3 6f 48 8d c9 00[ 	]+top4mxbhf8ps \$0x0,%zmm2,%zmm1,%tmm1
[ 	]*[a-f0-9]+:[ 	]*62 f3 6e 48 8d c9 00[ 	]+top4mxhbf8ps \$0x0,%zmm2,%zmm1,%tmm1
[ 	]*[a-f0-9]+:[ 	]*62 f3 6d 48 8d c9 00[ 	]+top4mxhf8ps \$0x0,%zmm2,%zmm1,%tmm1
[ 	]*[a-f0-9]+:[ 	]*62 f3 6f 48 8f c9 00[ 	]+top4mxbssps \$0x0,%zmm2,%zmm1,%tmm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 6e 48 5c c1[ 	]+top2bf16ps %zmm2,%zmm1,%tmm0
[ 	]*[a-f0-9]+:[ 	]*62 f2 6f 48 5e c1[ 	]+top4bssd %zmm2,%zmm1,%tmm0
[ 	]*[a-f0-9]+:[ 	]*62 f2 6e 48 5e c1[ 	]+top4bsud %zmm2,%zmm1,%tmm0
[ 	]*[a-f0-9]+:[ 	]*62 f2 6d 48 5e c1[ 	]+top4busd %zmm2,%zmm1,%tmm0
[ 	]*[a-f0-9]+:[ 	]*62 f2 6c 48 5e c1[ 	]+top4buud %zmm2,%zmm1,%tmm0
#pass
