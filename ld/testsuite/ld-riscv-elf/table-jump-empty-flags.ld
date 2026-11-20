ENTRY (_start)
SECTIONS
{
  .text 0x1000 : { *(.text) }
  .riscv.jvt 0x3000 : { BYTE(0xaa) *(.riscv.jvt) *(.retained) BYTE(0x55) }
  /DISCARD/ : { *(.prefix) *(.suffix) }
}
