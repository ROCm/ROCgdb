MEMORY
{
  ROM (rx) : ORIGIN = 0x8000, LENGTH = 0x1000
  RAM (rw) : ORIGIN = 0x1000, LENGTH = 0x1000
}

SECTIONS
{
  .data : ALIGN(4) { *(.data) } >RAM AT>ROM
  .tdata : ALIGN(4) { *(.tdata) } >RAM AT>ROM
  .tbss (NOLOAD) : ALIGN(4) { *(.tbss) } >RAM
  .bss (NOLOAD) : ALIGN(4) { *(.bss) } >RAM
}
