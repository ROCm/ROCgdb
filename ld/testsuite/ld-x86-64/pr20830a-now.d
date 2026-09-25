#name: PR ld/20830 (.plt.got, -z now)
#source: pr20830.s
#as: --64
#ld: -z now -melf_x86_64 -shared -z relro --ld-generated-unwind-info --hash-style=sysv -z max-page-size=0x200000 -z noseparate-code $NO_DT_RELR_LDFLAGS
#objdump: -dw -Wf

.*: +file format .*

Contents of the .eh_frame section:


0+ 0000000000000014 00000000 CIE
  Version:               1
  Augmentation:          "zR"
  Code alignment factor: 1
  Data alignment factor: -8
  Return address column: 16
  Augmentation data:     1b
  DW_CFA_def_cfa: r7 \(rsp\) ofs 8
  DW_CFA_offset: r16 \(rip\) at cfa-8
  DW_CFA_nop
  DW_CFA_nop

0+18 0+24 0+1c FDE cie=0+ pc=0+1b0\.\.0+1c0
  DW_CFA_def_cfa_offset: 16
  DW_CFA_advance_loc: 6 to 0+1b6
  DW_CFA_def_cfa_offset: 24
  DW_CFA_advance_loc: 10 to 0+1c0
  DW_CFA_def_cfa_expression \(DW_OP_breg7 \(rsp\): 8; DW_OP_breg16 \(rip\): 0; DW_OP_lit15; DW_OP_and; DW_OP_lit11; DW_OP_ge; DW_OP_lit3; DW_OP_shl; DW_OP_plus\)
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop

0+40 0+14 0+44 FDE cie=0+ pc=0+1c0\.\.0+1c8
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop

0+58 0+10 0+5c FDE cie=0+ pc=0+1c8\.\.0+1d4
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop


Disassembly of section .plt:

0+1b0 <.plt>:
 +[a-f0-9]+:	ff 35 32 fe 3f 00    	push   0x3ffe32\(%rip\)        # 3fffe8 <_GLOBAL_OFFSET_TABLE_\+0x8>
 +[a-f0-9]+:	ff 25 34 fe 3f 00    	jmp    \*0x3ffe34\(%rip\)        # 3ffff0 <_GLOBAL_OFFSET_TABLE_\+0x10>
 +[a-f0-9]+:	0f 1f 40 00          	nopl   0x0\(%rax\)

Disassembly of section .plt.got:

0+1c0 <func@plt>:
 +[a-f0-9]+:	ff 25 32 fe 3f 00    	jmp    \*0x3ffe32\(%rip\)        # 3ffff8 <func>
 +[a-f0-9]+:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0+1c8 <foo>:
 +[a-f0-9]+:	e8 f3 ff ff ff       	call   1c0 <func@plt>
 +[a-f0-9]+:	48 8b 05 24 fe 3f 00 	mov    0x3ffe24\(%rip\),%rax        # 3ffff8 <func>
#pass
