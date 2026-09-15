#source: ifunc-global.s
#ld: -relax -melf64alpha
#objdump: -dj.text

# The call must keep going through the GOT; relaxing it into a bsr would
# reach the resolver rather than the function it selects.
.*: +file format elf64-alpha.*

Disassembly of section \.text:
#...
[0-9a-f]+ <_start>:
#...
 +[0-9a-f]+:	[0-9a-f ]+	ldq	t12,-?[0-9]+\(gp\)
 +[0-9a-f]+:	[0-9a-f ]+	jsr	ra,\(t12\),[0-9a-f]+ <_start\+0x10>
#pass
