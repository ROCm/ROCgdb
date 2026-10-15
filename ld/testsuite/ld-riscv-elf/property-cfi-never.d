#name: -z zicfilp=never -z zicfiss=never
#source: property1.s
#as: -march=rv64g -mlittle-endian
#ld: -shared -melf64lriscv -z zicfilp=never -z zicfiss=never
#readelf: -SW

#failif
#...
.*\.note\.gnu\.property .*
#...
