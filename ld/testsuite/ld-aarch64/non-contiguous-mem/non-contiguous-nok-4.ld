MEMORY
{
  RAML    (rwx) : ORIGIN = 0x8000000,          LENGTH = 2K
  VECTORS (xrw) : ORIGIN = (0x8000000 + 2048), LENGTH = 2K
  RAMU    (xrw) : ORIGIN = 0x40000000,         LENGTH = 64K
}

SECTIONS
{
  RAM_REGION_START = ORIGIN(RAML);
  .boot : {
    *(.bootloader) *(.bootloader.*);
  } >RAML

  .vectors : AT ( ADDR (.boot) + SIZEOF (.boot) ) {
    *(.vectors)
  } >VECTORS

  .text : AT ( ADDR (.vectors) + SIZEOF (.vectors) )
  { _text_start = . ;
    *(.code) *(.code.*) ;
    _text_end = . ;
  } > RAMU
}
