#source: simple.s
#as: --x32
#ld: -pie -melf32_x86_64 $DT_RELR_LDFLAGS -T dt-relr-3.t
#readelf: -l --wide
#target: x86_64-*-linux*

#...
 Section to Segment mapping:
  Segment Sections\.\.\.
   00 +\.text .*
#pass
