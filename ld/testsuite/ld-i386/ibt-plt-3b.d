#source: ibt-plt-3.s
#as: --32
#ld: -shared -m elf_i386 -z ibtplt --hash-style=sysv -z noseparate-code
#readelf: -wf -n

Contents of the .eh_frame section:

0+ 00000014 00000000 CIE
  Version:               1
  Augmentation:          "zR"
  Code alignment factor: 1
  Data alignment factor: -4
  Return address column: 8
  Augmentation data:     1b

  DW_CFA_def_cfa: r4 \(esp\) ofs 4
  DW_CFA_offset: r8 \(eip\) at cfa-4
  DW_CFA_nop
  DW_CFA_nop

0+18 0+20 0+1c FDE cie=0+ pc=0+140\.\.0+170
  DW_CFA_def_cfa_offset: 8
  DW_CFA_advance_loc: 6 to 0+146
  DW_CFA_def_cfa_offset: 12
  DW_CFA_advance_loc: 10 to 0+150
  DW_CFA_def_cfa_expression \(DW_OP_breg4 \(esp\): 4; DW_OP_breg8 \(eip\): 0; DW_OP_lit15; DW_OP_and; DW_OP_lit9; DW_OP_ge; DW_OP_lit2; DW_OP_shl; DW_OP_plus\)

0+3c 0+10 0+40 FDE cie=0+ pc=0+170\.\.0+190
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop

0+50 0+1c 0+54 FDE cie=0+ pc=0+190\.\.0+1ae
  DW_CFA_advance_loc: 1 to 0+191
  DW_CFA_def_cfa_offset: 8
  DW_CFA_offset: r3 \(ebx\) at cfa-8
  DW_CFA_advance_loc: 14 to 0+19f
  DW_CFA_def_cfa_offset: 16
  DW_CFA_advance_loc: 13 to 0+1ac
  DW_CFA_def_cfa_offset: 8
  DW_CFA_advance_loc: 1 to 0+1ad
  DW_CFA_restore: r3 \(ebx\)
  DW_CFA_def_cfa_offset: 4

0+70 0+10 0+74 FDE cie=0+ pc=0+1ae\.\.0+1b2
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop

