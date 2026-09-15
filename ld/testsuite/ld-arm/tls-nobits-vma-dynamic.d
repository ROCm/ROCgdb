#source: tls-nobits-vma.s
#ld: -shared -T tls-nobits-vma.ld
#readelf: -hSW

ELF Header:
#...
  Type: +DYN \(Shared object file\)
#...

Section Headers:
  \[Nr\] Name +Type +Addr +Off +Size +ES Flg Lk Inf Al
  \[ 0\] +NULL +0+ +0+ +0+ +0+ +0 +0 +0
  \[ 1\] \.dynsym +DYNSYM +00008000 +[0-9a-f]+ +000010 +10 +A +2 +1 +4
  \[ 2\] \.dynstr +STRTAB +00008010 +[0-9a-f]+ +000001 +00 +A +0 +0 +1
  \[ 3\] \.hash +HASH +00008014 +[0-9a-f]+ +000010 +04 +A +1 +0 +4
  \[ 4\] \.data +PROGBITS +00001000 +[0-9a-f]+ +000004 +00 +WA +0 +0 +4
  \[ 5\] \.dynamic +DYNAMIC +00001004 +[0-9a-f]+ +000058 +08 +WA +2 +0 +4
  \[ 6\] \.got\.plt +PROGBITS +0000105c +[0-9a-f]+ +00000c +04 +WA +0 +0 +4
  \[ 7\] \.tdata +PROGBITS +00001068 +[0-9a-f]+ +000004 +00 WAT +0 +0 +4
  \[ 8\] \.tbss +NOBITS +0000106c +[0-9a-f]+ +000004 +00 WAT +0 +0 +4
  \[ 9\] \.bss +NOBITS +00001070 +[0-9a-f]+ +000008 +00 +WA +0 +0 +4
  \[10\] \.ARM\.attributes +ARM_ATTRIBUTES +0+ +[0-9a-f]+ +[0-9a-f]+ +00 +0 +0 +1
  \[11\] \.symtab +SYMTAB +0+ +[0-9a-f]+ +[0-9a-f]+ +10 +12 +[0-9]+ +4
  \[12\] \.strtab +STRTAB +0+ +[0-9a-f]+ +[0-9a-f]+ +00 +0 +0 +1
  \[13\] \.shstrtab +STRTAB +0+ +[0-9a-f]+ +[0-9a-f]+ +00 +0 +0 +1
Key to Flags:
#...
