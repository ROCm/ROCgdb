SECTIONS
{
  .riscv.jvt 0x10018 : SUBALIGN(1) { *(.riscv.jvt) }
  .text 0x11000 : { *(.text) }
  .far 0x300000 : { *(.far) }
}
