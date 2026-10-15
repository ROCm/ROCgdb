OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS
{
  /DISCARD/ : { *(.discard.*) }

  . = 0x10000;
  .text : { *(.text) }
  . = 0x20000;
  .got  : { *(.got) *(.got.plt)}
  . = 0x12340000;
  .data : { *(.data) *(.data.*) }
}
