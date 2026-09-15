#source: ifunc-local.s
#ld: -relax -melf64alpha
#objdump: -dj.text

# As for a global IFUNC, the call must keep going through the GOT.  A local
# symbol reaches the check in elf64_alpha_relax_section by its symbol table
# entry rather than by a hash table entry.
.*: +file format elf64-alpha.*

Disassembly of section \.text:
#...
[0-9a-f]+ <_start>:
#...
 +[0-9a-f]+:	[0-9a-f ]+	ldq	t12,-?[0-9]+\(gp\)
 +[0-9a-f]+:	[0-9a-f ]+	jsr	ra,\(t12\),[0-9a-f]+ <_start\+0x10>
#pass
