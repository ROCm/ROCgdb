SECTIONS
{
  .text 0x400000 : { *(.literal* .text*) }
  .rodata 0x400100 : { *(.rodata*) }
}
