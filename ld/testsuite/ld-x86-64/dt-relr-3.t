PHDRS
{
  text PT_LOAD FILEHDR PHDRS FLAGS (5);
}

SECTIONS
{
  . = SIZEOF_HEADERS;
  .text : { *(.text) } :text
}
