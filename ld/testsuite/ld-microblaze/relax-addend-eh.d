#source: relax-addend-eh.s
#source: relax-addend-eh-support.s
#as: -EL
#ld: -EL -relax --gc-sections -T $srcdir/$subdir/relax-addend-eh.ld
#objdump: -d
#name: MicroBlaze relaxation preserves addends across .eh_frame editing

# Same defect as relax-addend.d, reached the other way.
#
# _bfd_elf_discard_section_eh_frame installs a locals-only symbol cache in
# symtab_hdr->contents when editing .eh_frame moves a local symbol defined
# inside it; the generic ELF emulation then calls lang_relax_sections from the
# same after_allocation.  dead_fn is unreferenced, so --gc-sections drops it,
# its FDE is removed, the section shrinks and ehlocal moves -- which is what
# makes the cache appear.  The linker script must KEEP .eh_frame or it is swept
# and none of this happens.
#
# gvar lands at 0x90000074, so the reference to gvar + 0x18 must be 0x9000008c,
# encoded as IMM 0x9000 followed by LWI with 0x008c.  A linker which corrupts
# the addend emits e8600088.

.*: +file format .*
#...
9000001c <zzz_victim>:
[ 	]*9000001c:[ 	]+b0009000[ 	]+imm[ 	]+-28672
[ 	]*90000020:[ 	]+e860008c[ 	]+lwi[ 	]+r3, r0, 140
#pass
