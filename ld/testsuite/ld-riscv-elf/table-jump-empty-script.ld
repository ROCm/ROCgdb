ENTRY (_start)
SECTIONS
{
  .text 0x1000 : { *(.text) }
  .riscv.jvt : { BYTE(0xaa) *(.riscv.jvt) BYTE(0x55) }
  .rodata : { *(.prefix) *(.suffix) }
}
