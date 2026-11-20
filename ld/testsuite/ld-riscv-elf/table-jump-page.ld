SECTIONS
{
  .prefix 2 * page_size - 65 : { BYTE(0) }
  .riscv.jvt : { *(.riscv.jvt) }
  .pad : { . += 0x38; }
  /* Allocating one RV64 JVT entry moves .from from eight bytes before
     a page boundary to the boundary.  So .target moves by a full page,
     despite the two inputs being in an output not containing the JVT.  */
  .text : { *(.from) . = ALIGN(page_size); *(.target) }
  .fillers 4 * page_size + 0x10000 : { *(.fillers) }
  .far ADDR(.fillers) + 0x300000 : { *(.far) }
}
