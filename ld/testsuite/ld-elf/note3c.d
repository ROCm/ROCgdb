#name: .note.GNU-stack ld -r SHT_PROGBITS + SHT_NOTE (positive)
#source: note3.s
#source: empty.s --noexecstack
#as: --generate-missing-build-notes=no
#ld: -r
#readelf: -SW

#...
 *\[[ 0-9]+\] \.note\.GNU-stack +PROGBITS .*
#pass
