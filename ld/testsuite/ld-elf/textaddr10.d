#source: maxpage1.s
#ld: -shared -z max-page-size=0x800000 -z noseparate-code
#readelf: -l --wide
#target: *-*-linux-gnu *-*-gnu* arm*-*-uclinuxfdpiceabi
#xfail: ![check_shared_lib_support]

#...
  LOAD +0x0* 0x0* 0x0* 0x[0-9a-f]+ 0x[0-9a-f]+ R E 0x800000
#pass
