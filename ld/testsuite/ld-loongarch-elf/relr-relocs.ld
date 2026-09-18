/* Script for DT_RELR tests */
OUTPUT_ARCH(loongarch64)
ENTRY(_start)
SECTIONS
{
	PROVIDE (__executable_start = 0x8000); . = 0x10000;
	.text :
	{
		*(.before)
		*(.text)
		*(.after)
	} =0
	. = 0x20000;
	.got :
	{
		*(.got)
		*(.got.plt)
	}
	. = 0x12340000;
	.data :
	{
		*(.data)
	}
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
