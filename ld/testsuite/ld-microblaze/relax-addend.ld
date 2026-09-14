/* Fixed layout so that the expected dump is deterministic.  gvar is the first
   thing in .data, which is pinned, so the reference to gvar + 0x18 has a known
   value regardless of how the sections happen to be ordered.  */
ENTRY(_start)
SECTIONS
{
  . = 0x90000000;
  .text : {
    *(.text.aaa_relaxed)
    *(.text.zzz_victim)
    *(.text.zz_support)
    *(.text)
    *(.text.*)
  }
  . = 0x90001000;
  .data : {
    *(.data)
    *(.data.*)
  }
  . = 0x90001100;
  .checkdata : { KEEP (*(.checkdata)) }
  /DISCARD/ : { *(.comment) *(.note*) }
}
