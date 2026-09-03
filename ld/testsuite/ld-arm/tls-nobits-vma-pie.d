#source: tls-nobits-vma.s
#ld: -pie -T tls-nobits-vma.ld
#readelf: -hSW

ELF Header:
#...
  Type: +EXEC \(Executable file\)
#...

Section Headers:
  \[Nr\] Name +Type +Addr +Off +Size +ES Flg Lk Inf Al
  \[ 0\] +NULL +0+ +0+ +0+ +0+ +0 +0 +0
  \[ 1\] \.interp +PROGBITS +00008000 +[0-9a-f]+ +000011 +00 +A +0 +0 +1
  \[ 2\] \.dynsym +DYNSYM +00008014 +[0-9a-f]+ +000010 +10 +A +3 +1 +4
  \[ 3\] \.dynstr +STRTAB +00008024 +[0-9a-f]+ +000001 +00 +A +0 +0 +1
  \[ 4\] \.hash +HASH +00008028 +[0-9a-f]+ +000010 +04 +A +2 +0 +4
  \[ 5\] \.data +PROGBITS +00001000 +[0-9a-f]+ +000004 +00 +WA +0 +0 +4
  \[ 6\] \.dynamic +DYNAMIC +00001004 +[0-9a-f]+ +000068 +08 +WA +3 +0 +4
  \[ 7\] \.got\.plt +PROGBITS +0000106c +[0-9a-f]+ +00000c +04 +WA +0 +0 +4
  \[ 8\] \.tdata +PROGBITS +00001078 +[0-9a-f]+ +000004 +00 WAT +0 +0 +4
  \[ 9\] \.tbss +NOBITS +0000107c +[0-9a-f]+ +000004 +00 WAT +0 +0 +4
  \[10\] \.bss +NOBITS +00001080 +[0-9a-f]+ +000008 +00 +WA +0 +0 +4
  \[11\] \.ARM\.attributes +ARM_ATTRIBUTES +0+ +[0-9a-f]+ +[0-9a-f]+ +00 +0 +0 +1
  \[12\] \.symtab +SYMTAB +0+ +[0-9a-f]+ +[0-9a-f]+ +10 +13 +[0-9]+ +4
  \[13\] \.strtab +STRTAB +0+ +[0-9a-f]+ +[0-9a-f]+ +00 +0 +0 +1
  \[14\] \.shstrtab +STRTAB +0+ +[0-9a-f]+ +[0-9a-f]+ +00 +0 +0 +1
Key to Flags:
#...
