ENTRY (_start)
SECTIONS
{
  .text 0x1000 : { *(.text) }
  .out 0x3000 : { *(.riscv.jvt) *(.strings.*) *(.extra) }
}
