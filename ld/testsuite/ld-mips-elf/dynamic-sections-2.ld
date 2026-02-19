SECTIONS
{
  .MIPS.abiflags : { *(.MIPS.abiflags) }
  .dynamic : { *(.dynamic) }
  .hash : { *(.hash) }
  .data : { *(.data) }
  .dynsym : { *(.dynsym) }
  .dynstr : { *(.dynstr) }
  .rel.dyn : { *(.rel.dyn) }
  .got : { *(.got) }
  /DISCARD/ : { *(*) }
}
