SCRIPT_NAME=elf
ELFSIZE=64
OUTPUT_FORMAT="elf64-hppa-linux"
NO_REL_RELOCS=yes
TEXT_START_ADDR=0x10000
TARGET_PAGE_SIZE=0x10000
MAXPAGESIZE="CONSTANT (MAXPAGESIZE)"
COMMONPAGESIZE="CONSTANT (COMMONPAGESIZE)"
if test "$LD_FLAG" = "N"; then
  unset DATA_SEGMENT_ALIGN
  unset DATA_SEGMENT_END
  unset DATA_SEGMENT_RELRO_END
else
  DATA_SEGMENT_ALIGN="ALIGN(${MAXPAGESIZE});\
  . = DATA_SEGMENT_ALIGN (${MAXPAGESIZE}, ${COMMONPAGESIZE})"
  DATA_SEGMENT_END=". = DATA_SEGMENT_END (.);"
  DATA_SEGMENT_RELRO_END=". = DATA_SEGMENT_RELRO_END (${SEPARATE_GOTPLT-0}, .);"
fi
DATA_SECTION_ALIGNMENT="${CREATE_SHLIB-${CREATE_PIE-ALIGN(8)}}"
ARCH=hppa
MACHINE=hppa2.0w
NOP=0x08000240
ENTRY="main"
TEMPLATE_NAME=elf
GENERATE_SHLIB_SCRIPT=yes

# We really want multiple .stub sections, one for each input .text section,
# but for now this is good enough.
OTHER_READONLY_SECTIONS="
  .PARISC.unwind ${RELOCATING-0} : { *(.PARISC.unwind) }"

# The PA64 ELF port treats .plt sections differently than most.  We also have
# to create a .opd section.  What most systems call the .got, we call the .dlt
OTHER_READWRITE_SECTIONS="
  .opd          ${RELOCATING-0} : { *(.opd) }
  ${RELOCATING+PROVIDE (__gp = .);}
  .plt          ${RELOCATING-0} : { *(.plt) }
  .dlt          ${RELOCATING-0} : { *(.dlt) }"

# The PA64 ELF port has an additional huge bss section.
OTHER_BSS_SECTIONS=".hbss         ${RELOCATING-0} : { *(.hbss) }"

#OTHER_SYMBOLS='PROVIDE (__TLS_SIZE = SIZEOF (.tbss));'
OTHER_SYMBOLS='
  PROVIDE (__TLS_SIZE = 0);
  PROVIDE (__TLS_INIT_SIZE = 0);
  PROVIDE (__TLS_INIT_START = 0);
  PROVIDE (__TLS_INIT_A = 0);
  PROVIDE (__TLS_PREALLOC_DTV_A = 0);'

# HPs use .dlt where systems use .got.  Sigh.
OTHER_GOT_RELOC_SECTIONS="
  .rela.dlt     ${RELOCATING-0} : { *(.rela.dlt) }
  .rela.opd     ${RELOCATING-0} : { *(.rela.opd) }"

DATA_PLT=
PLT_BEFORE_GOT=

# .dynamic should be at the start of the .text segment.
TEXT_DYNAMIC=
