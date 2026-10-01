#source: ibt-plt-3.s
#as: --x32
#ld: -shared -m elf32_x86_64 -z ibt --hash-style=sysv -z max-page-size=0x200000 -z noseparate-code
#readelf: -wf -n

Contents of the .eh_frame section:

0+ 00000014 00000000 CIE
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

0+18 0+20 0+1c FDE cie=0+ pc=0+180\.\.0+1b0
  DW_CFA_def_cfa_offset: 16
  DW_CFA_advance_loc: 6 to 0+186
  DW_CFA_def_cfa_offset: 24
  DW_CFA_advance_loc: 10 to 0+190
  DW_CFA_def_cfa_expression \(DW_OP_breg7 \(rsp\): 8; DW_OP_breg16 \(rip\): 0; DW_OP_lit15; DW_OP_and; DW_OP_lit9; DW_OP_ge; DW_OP_lit3; DW_OP_shl; DW_OP_plus\)

0+3c 0+10 0+40 FDE cie=0+ pc=0+1b0\.\.0+1d0
  DW_CFA_nop
  DW_CFA_nop
  DW_CFA_nop

0+50 0+14 0+54 FDE cie=0+ pc=0+1d0\.\.0+1e2
  DW_CFA_advance_loc: 4 to 0+1d4
  DW_CFA_def_cfa_offset: 16
  DW_CFA_advance_loc: 9 to 0+1dd
  DW_CFA_def_cfa_offset: 8
  DW_CFA_nop


Displaying notes found in: .note.gnu.property
[ 	]+Owner[ 	]+Data size[ 	]+Description
  GNU                  0x0000000c	NT_GNU_PROPERTY_TYPE_0
      Properties: x86 feature: IBT

