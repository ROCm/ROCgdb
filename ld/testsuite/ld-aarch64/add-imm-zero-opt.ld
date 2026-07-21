OUTPUT_ARCH(aarch64)
SECTIONS
{
  . = 0x10000;
  .text : { *(.text) }
  . = got_base;
  .got : { *(.got) *(.got.plt) }
  . = 0x30000;
  .tdata : { *(.tdata) }
  .tbss : { *(.tbss) }
  .data :
  {
    *(.data)
    *(.data.*)
  }
}
