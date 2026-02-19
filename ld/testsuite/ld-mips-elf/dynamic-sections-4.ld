SECTIONS
{
  .MIPS.abiflags : { *(.MIPS.abiflags) }
  .dynsym : { *(.dynsym) }
  .dynstr : { *(.dynstr) }
  .dynamic : { *(.dynamic) }
  .data : { *(.data) }
  .hash : { *(.hash) }
  .rel.dyn : { *(.rel.dyn) }
  .got : { *(.got) }
  /DISCARD/ : { *(*) }
}
