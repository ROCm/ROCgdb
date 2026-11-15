#as: -k
#objdump: -r
#name: VAX ELF PIC address of a function uses the GOT

.*:     file format elf32-vax

RELOCATION RECORDS FOR \[\.text\]:
OFFSET +TYPE +VALUE
00000004 R_VAX_GOT32       fn
0000000b R_VAX_GOT32       fn
00000012 R_VAX_PLT32       fn
00000019 R_VAX_PLT32       fn
0000001f R_VAX_PLT32       fn
00000025 R_VAX_PLT32       fn
