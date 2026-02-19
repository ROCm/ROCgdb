SECTIONS
{
  .MIPS.abiflags : { *(.MIPS.abiflags) }
  .dynsym : { *(.dynsym) }
  .dynstr : { *(.dynstr) }
  .dynamic : { *(.dynamic) }
  .hash : { *(.hash) }
  .rel.dyn : { *(.rel.dyn) }
  .data : { *(.data) }
  .got : { *(.got) }
  /DISCARD/ : { *(*) }
}
