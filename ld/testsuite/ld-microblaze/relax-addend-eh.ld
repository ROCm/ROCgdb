ENTRY(_start)
SECTIONS
{
  . = 0x90000000;
  .text : { *(.text) *(.text.*) }
  .eh_frame : { KEEP (*(.eh_frame)) }
  .data : { *(.data) *(.data.*) }
  /DISCARD/ : { *(.comment) *(.note*) }
}
