#source: tls-nobits-vma.s
#ld: -T tls-nobits-vma.ld
#readelf: -hSW

ELF Header:
#...
  Type: +EXEC \(Executable file\)
#...

Section Headers:
  \[Nr\] Name +Type +Addr +Off +Size +ES Flg Lk Inf Al
  \[ 0\] +NULL +0+ +0+ +0+ +0+ +0 +0 +0
  \[ 1\] \.data +PROGBITS +00001000 +[0-9a-f]+ +000004 +00 +WA +0 +0 +4
  \[ 2\] \.tdata +PROGBITS +00001004 +[0-9a-f]+ +000004 +00 WAT +0 +0 +4
  \[ 3\] \.tbss +NOBITS +00001008 +[0-9a-f]+ +000004 +00 WAT +0 +0 +4
  \[ 4\] \.bss +NOBITS +0000100c +[0-9a-f]+ +000008 +00 +WA +0 +0 +4
  \[ 5\] \.ARM\.attributes +ARM_ATTRIBUTES +0+ +[0-9a-f]+ +[0-9a-f]+ +00 +0 +0 +1
  \[ 6\] \.symtab +SYMTAB +0+ +[0-9a-f]+ +[0-9a-f]+ +10 +7 +[0-9]+ +4
  \[ 7\] \.strtab +STRTAB +0+ +[0-9a-f]+ +[0-9a-f]+ +00 +0 +0 +1
  \[ 8\] \.shstrtab +STRTAB +0+ +[0-9a-f]+ +[0-9a-f]+ +00 +0 +0 +1
Key to Flags:
#...
