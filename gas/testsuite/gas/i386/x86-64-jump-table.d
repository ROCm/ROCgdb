#as:
#readelf: -rsW
#name: x86-64 PIC jump table
#notarget: *-*-solaris*

#...
Relocation section '.rela.text' at offset 0x[0-9a-f]+ contains 3 entries:
 +Offset +Info +Type +Sym.* Value +Symbol's Name \+ Addend
0+8  0+40+2 +R_X86_64_PC32 +0+ +bar0 - 4
0+f  0+20+2 +R_X86_64_PC32 +0+ +.rodata - 4
0+20  0+50+4 +R_X86_64_PLT32 +0+ +bar2 - 4
#...
Relocation section '.rela.rodata' at offset 0x[0-9a-f]+ contains 5 entries:
 +Offset +Info +Type +Sym.* Value +Symbol's Name \+ Addend
0+  0+40+4 +R_X86_64_PLT32 +0+ +bar0 \+ 0
0+4  0+60+4 +R_X86_64_PLT32 +0+ +bar1 \+ 4
0+8  0+10+2 +R_X86_64_PC32 +0+ +.text \+ 27
0+c  0+70+4 +R_X86_64_PLT32 +0+ +bar3 \+ c
0+10+  0+80+4 +R_X86_64_PLT32 +0+ +bar4 \+ 10
#...
 +[0-9]+: 0+ +0 SECTION LOCAL +DEFAULT +1 .text
 +[0-9]+: 0+ +0 SECTION LOCAL +DEFAULT +5 .rodata
 +[0-9]+: 0+ +36 FUNC +GLOBAL +DEFAULT +1 foo
 +[0-9]+: 0+ +0 NOTYPE +GLOBAL DEFAULT +UND bar0
 +[0-9]+: 0+ +0 NOTYPE +GLOBAL DEFAULT +UND bar2
 +[0-9]+: 0+ +0 NOTYPE +GLOBAL DEFAULT +UND bar1
 +[0-9]+: 0+ +0 NOTYPE +GLOBAL DEFAULT +UND bar3
 +[0-9]+: 0+ +0 NOTYPE +GLOBAL DEFAULT +UND bar4
#pass
