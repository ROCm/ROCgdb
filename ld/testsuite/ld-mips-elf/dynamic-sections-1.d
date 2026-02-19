#name: Dynamic segment sections 1
#ld: -shared -T dynamic-sections-1.ld
#readelf: -Wl
#target: [check_shared_lib_support]
#source: dynamic-sections.s

Elf file type is DYN \(Shared object file\)
Entry point 0x0
There are 4 program headers, starting at offset .*

Program Headers:
  Type           Offset +VirtAddr +PhysAddr +FileSiz +MemSiz +Flg +Align
  ABIFLAGS       .*
  LOAD           .*
  DYNAMIC        [^ ]+ +[^ ]+ +[^ ]+ +(0x[0-9a-f]+) +\1 +.*
  NULL           .*

 Section to Segment mapping:
  Segment Sections\.\.\.
   00     .*
   01     .*
   02     \.dynamic 
   03     
