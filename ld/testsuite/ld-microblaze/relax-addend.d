#source: relax-addend.s
#source: relax-addend-support.s
#as: -EL
#ld: -EL -relax --gc-sections -T $srcdir/$subdir/relax-addend.ld
#objdump: -d
#name: MicroBlaze relaxation preserves R_MICROBLAZE_64 addends

# .text.aaa_relaxed contains an IMM that relaxation deletes.  .text.zzz_victim,
# a sibling section of the same object, refers to gvar + 0x18.  The linker must
# not let the deletion disturb that addend.
#
# The linker script pins .data, so gvar is at 0x90001000 and the reference must
# resolve to 0x90001018, encoded as IMM 0x9000 followed by LWI with 0x1018.
# A linker which corrupts the addend emits e8601014 instead.

.*: +file format .*
#...
90000000 <aaa_relaxed>:
[ 	]*90000000:[ 	]+3021ffe4[ 	]+addik[ 	]+r1, r1, -28
[ 	]*90000004:[ 	]+f9e10000[ 	]+swi[ 	]+r15, r1, 0
[ 	]*90000008:[ 	]+b9f40024[ 	]+brlid[ 	]+r15, 36
#...
9000001c <zzz_victim>:
[ 	]*9000001c:[ 	]+b0009000[ 	]+imm[ 	]+-28672
[ 	]*90000020:[ 	]+e8601018[ 	]+lwi[ 	]+r3, r0, 4120
#pass
