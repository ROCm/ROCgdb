SECTIONS
{
  .text 0x1000 : { *(.text) }
  .riscv.jvt 0x2000 :
  {
    *(.riscv.jvt)
    /* One RV64 entry fits before the following firmware byte.  */
    . = 8;
    BYTE(0x5a)
  }
  .far 0x300000 : { *(.far) }
}
