#name: objdump SFrame with EH Frame
#source: sframe-func.s
#as: --gsframe
#objdump: --sframe -WF
#target: x86_64-*-*
#notarget: ![gas_sframe_check]

.*: +file format .*

Contents of the .eh_frame section:

[0-9a-f]+ [0-9a-f]+ [0-9a-f]+ CIE "zR" cf=1 df=-8 ra=16
#...
[0-9a-f]+ [0-9a-f]+ [0-9a-f]+ FDE cie=[0-9a-f]+ pc=[0-9a-f]+..0+0010
#...

Contents of the SFrame section .sframe:
  Header :

    Version: SFRAME_VERSION_3
    Flags: SFRAME_F_FDE_FUNC_START_PCREL
#...
    Num FDEs: 1
    Num FREs: 4

  Function Index :

    func idx \[0\]: pc = 0x0, size = 16 bytes
    STARTPC +CFA +FP +RA +
    0+0000 +sp\+8 +u +f +
    0+0004 +sp\+16 +c-16 +f +
    0+0008 +fp\+16 +c-16 +f +
    0+000c +sp\+8 +c-16 +f +
