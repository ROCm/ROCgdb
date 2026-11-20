ENTRY(_start)
SECTIONS
{
  . = 0x1000;
  .riscv.jvt : { *(.riscv.jvt) }
  . = 0x3000;
  .text : { *(.text.startup) *(.text .text.*) }
  . = 0x8000;
  __global_pointer$ = .;
  .sdata : { *(.sdata .sdata.*) }
  .data : { *(.data .data.*) }
}
