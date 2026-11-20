SECTIONS
{
  __global_pointer$ = 0x2000;
  .prefix 0x7bf : { BYTE(0) }
  .riscv.jvt : { *(.riscv.jvt) }
  .pad : { . += 0x38; }
  /* Each target is 0x7f8 bytes above its base before planning and
     0x800 above it after the single RV64 JVT entry is allocated.  */
  .data : { *(.data.x0) . += 0x1fff; *(.data.gp) }
  .text 0x4000 : { *(.text) }
  .far 0x300000 : { *(.far) }
}
