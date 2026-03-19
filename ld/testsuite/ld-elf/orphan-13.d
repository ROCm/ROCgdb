#source: orphan-13a.s
#source: orphan-13b.s
#source: orphan-13c.s
#ld: -T orphan-13.ld
#readelf: -S --wide

#...
  \[[ 0-9]+\] \.foo +NOBITS +[0-9a-f]+ +[0-9a-f]+ +0+30 +0+ +A +0 +0 +[0-9]+
  \[[ 0-9]+\] [._][^f].*
#pass
