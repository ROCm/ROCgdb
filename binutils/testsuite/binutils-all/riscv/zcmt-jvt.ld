SECTIONS
{
  .text 0x1000 : { *(.text) }
  .riscv.jvt 0x2000 : { *(.riscv.jvt) }
}
