#source: gc-prop.s
#ld: -T gc-prop.t --gc-sections
#map: gc-prop.map
#readelf: -SW
#...
 +\[ *[0-9]+\] \.xt\.prop +PROGBITS +[0-9a-f]+ +[0-9a-f]+ +0*[1-9a-f][0-9a-f]* .*
#pass
