#as: --gsframe
#objdump: --sframe=.sframe
#name: Signal Frame with unsupported CFI
#warning: \.cfi_def_cfa_offset with unsupported offset value
#...
Contents of the SFrame section .sframe:

  Header :

    Version: SFRAME_VERSION_3
    Flags: SFRAME_F_FDE_FUNC_START_PCREL
#?    CFA fixed FP offset: \-?\d+
#?    CFA fixed RA offset: \-?\d+
    Num FDEs: 1
    Num FREs: 0

  Function Index :
    func idx \[0\]: pc = 0x0, size = 4 bytes, attr = \"S\"
    STARTPC + CFA + FP + RA +
#pass
