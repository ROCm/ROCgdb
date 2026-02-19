SECTIONS
{
  .MIPS.abiflags : { *(.MIPS.abiflags) }
  .dynamic : { *(.dynamic) }
  .hash : { *(.hash) }
  .dynsym : { *(.dynsym) }
  .dynstr : { *(.dynstr) }
  .rel.dyn : { *(.rel.dyn) }
  .data : { *(.data) }
  .got : { *(.got) }
  /DISCARD/ : { *(*) }
}
