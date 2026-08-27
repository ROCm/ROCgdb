#as: --no-link-relax
#source: pdiff-uleb128.s
#objdump: -r -s -j .debug_info -j .xt.prop
#name: uleb128 difference without link-relax

# Without link-relax, no R_XTENSA_PDIFF_ULEB128 is emitted.  The assembler
# falls back to marking the covered code no_transform via .xt.prop, and the
# uleb128 uses the minimal encoding (no spare byte).

.*: +file format .*xtensa.*

RELOCATION RECORDS FOR \[.xt.prop\]:
OFFSET +TYPE +VALUE
0+ R_XTENSA_32 +.text.*

Contents of section .debug_info:
 0000 c801.*
#...
Contents of section .xt.prop:
 0000 .*
