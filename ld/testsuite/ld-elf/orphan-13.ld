SECTIONS
{
  . = SIZEOF_HEADERS;
  .text : { *(.text) }
  .data : { *(.data) }
  .foo (NOLOAD) : {}
}
