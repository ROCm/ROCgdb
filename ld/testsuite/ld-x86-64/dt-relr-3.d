#source: simple.s
#as: --64
#ld: -pie -melf_x86_64 $DT_RELR_LDFLAGS -T dt-relr-3.t
#readelf: -l --wide
#target: x86_64-*-linux*

#...
 Section to Segment mapping:
  Segment Sections\.\.\.
   00 +\.text .*
#pass
