#name: diagnostics for visibility directives
#readelf: -s -W
#warning_output: visibility.l
#target: [supports_gnu_unique]

#...
 +[0-9]+: +0+ +0 +(NOTYPE|OBJECT) +GLOBAL +INTERNAL +[1-9] +gd
 +[0-9]+: +0+1 +0 +(NOTYPE|OBJECT) +WEAK +INTERNAL +[1-9] +wd
 +[0-9]+: +0+2 +0 +OBJECT +UNIQUE +HIDDEN +[1-9] +gu
 +[0-9]+: +0+ +0 +(NOTYPE|OBJECT) +GLOBAL +INTERNAL +UND +ge
 +[0-9]+: +0+ +0 +(NOTYPE|OBJECT) +WEAK +HIDDEN +UND +we
#pass
