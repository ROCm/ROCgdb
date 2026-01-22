SECTIONS
{
  . = . + SIZEOF_HEADERS;
  .text : { *(.text) *(.plt) *(.rodata) *(.got*) }
  .note : { *(.note) *(.note.*) }
  /DISCARD/ : { *(*) }
}
