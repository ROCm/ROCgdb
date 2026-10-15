SECTIONS
{
  .text 0x00001000 : { *(.literal .text) }
  .rodata : { *(.rodata) }
}
