#name: -z zicfilp= with an unknown value
#source: property1.s
#as: -march=rv64g -mlittle-endian
#ld: -shared -melf64lriscv -z zicfilp=foo
#error: .*: error: unrecognized value '-z zicfilp=foo'
