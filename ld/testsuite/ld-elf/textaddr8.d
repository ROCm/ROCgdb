#source: maxpage1.s
#ld: -z max-page-size=0x800000 --image-base 0x40000 -z separate-code
#warning: image base \(0x40000\) < maximum page size \(0x800000\)
#readelf: -l --wide
#target: *-*-linux-gnu *-*-gnu* arm*-*-uclinuxfdpiceabi

#...
  LOAD +0x0* 0x0*800000 0x[0-9a-f]+ 0x[0-9a-f]+ 0x[0-9a-f]+ R [E ] 0x800000
#pass
