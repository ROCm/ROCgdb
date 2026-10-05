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
[ 	]*[a-f0-9]+:[ 	]*62 62 6e 48 4a f5[ 	]+tcvtrowd2ps %edx,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 7e 48 07 f5 08[ 	]+tcvtrowd2ps \$0x8,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 62 6f 48 6d f5[ 	]+tcvtrowps2bf16h %edx,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 7f 48 07 f5 08[ 	]+tcvtrowps2bf16h \$0x8,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 62 6c 48 6d f5[ 	]+tcvtrowps2phh %edx,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 7c 48 07 f5 08[ 	]+tcvtrowps2phh \$0x8,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 62 6e 48 6d f5[ 	]+tcvtrowps2bf16l %edx,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 7e 48 77 f5 08[ 	]+tcvtrowps2bf16l \$0x8,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 62 6d 48 6d f5[ 	]+tcvtrowps2phl %edx,%tmm5,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 7f 48 77 f5 08[ 	]+tcvtrowps2phl \$0x8,%tmm5,%zmm30
#pass
