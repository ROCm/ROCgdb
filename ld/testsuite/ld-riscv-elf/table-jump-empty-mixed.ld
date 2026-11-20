ENTRY (_start)
MEMORY { rom (rx) : ORIGIN = 0x1000, LENGTH = 12 }
SECTIONS
{
  .text : { *(.text) } >rom
  .mixed : { *(.prefix) *(.riscv.jvt) *(.suffix) } >rom
}
