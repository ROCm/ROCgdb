/* Script for ld testsuite */
OUTPUT_ARCH(aarch64)
ENTRY(_start)
SECTIONS
{
  /* Read-only sections, merged into text segment: */
  PROVIDE (__executable_start = 0x8000); . = 0x10000;
  .text           :
  {
    *(.before)
    *(.text)
    *(.after)
  } =0
  .rodata :
  {
    *(.rodata)
  }
  . = 0x20000;
  .got            : { *(.got) *(.got.plt)}
  . = 0x12340000;
  .far : { *(.far) }
  .ARM.attributes 0 : { *(.ARM.attributes) }
  /* Providing a .data section triggers a non-executable segment to be
     generated, which gives somewhere else for the linker to put the TLS
     segment without creating a RWX permissioned one. */
  .data : { *(.data) }
  .data.unaligned_local : { *(.data.unaligned_local) }
  .data.unaligned_hidden : { *(.data.unaligned_hidden) }
  .data.unaligned_global : { *(.data.unaligned_global) }
  .data.unaligned_DYNAMIC : { *(.data.unaligned_DYNAMIC) }
  .data.aligned_local : { *(.data.aligned_local) }
  .data.aligned_hidden : { *(.data.aligned_hidden) }
  .data.aligned_global : { *(.data.aligned_global) }
  .data.aligned_global_abs : { *(.data.aligned_global_abs) }
  .data.aligned_weak_undef : { *(.data.aligned_weak_undef) }
  .data.aligned_DYNAMIC : { *(.data.aligned_DYNAMIC) }
  .dynamic : { *(.dynamic) }
}
