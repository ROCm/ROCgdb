#as:
#readelf: -x .rodata -r

Relocation section '.rel.rodata' at offset 0xd4 contains 4 entries:
 Offset     Info    Type            Sym.Value  Sym. Name
0+  00000104 R_386_PLT32       00000000   foo1
0+4  00000204 R_386_PLT32       00000000   foo2
0+8  00000304 R_386_PLT32       00000000   foo3
0+c  00000404 R_386_PLT32       00000000   foo4

Hex dump of section '.rodata':
 NOTE: This section has relocations against it, but these have NOT been applied to this dump.
  0x00000000 00000000 04000000 08000000 0c000000 ................
#pass
