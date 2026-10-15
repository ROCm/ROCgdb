OUTPUT_ARCH (riscv)
ENTRY (_start)
SECTIONS
{
  . = 0x10000;
  .text : { *(.got_*) *(.text)}
  . = 0x20000;
  .got : { *(.got) *(.got.plt)}
  . = 0x12340000;
  .data : { *(.data) }
  .dynamic : { *(.dynamic) }
  .riscv.attributes 0 : { *(.riscv.attributes) }
}
