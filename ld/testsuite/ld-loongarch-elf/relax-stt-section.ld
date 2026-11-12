SECTIONS
{
  . = 0x12000000;
  .text : { *(.text) }

  . = 0x12010000;
  .data : { *(.data) }
}
