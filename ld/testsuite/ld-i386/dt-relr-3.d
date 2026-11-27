#source: ../ld-x86-64/simple.s
#as: --32
#ld: -pie -melf_i386 $DT_RELR_LDFLAGS -T ../ld-x86-64/dt-relr-3.t
#readelf: -l --wide
#target: x86_64-*-linux* i?86-*-linux-gnu i?86-*-gnu*

#...
 Section to Segment mapping:
  Segment Sections\.\.\.
   00 +\.text .*
#pass
