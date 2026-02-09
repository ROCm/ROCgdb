#name: .note.GNU-stack ld -r SHT_NOTE + SHT_PROGBITS (positive)
#source: empty.s --noexecstack
#source: note3.s
#as: --generate-missing-build-notes=no
#ld: -r
#readelf: -SW

#...
 *\[[ 0-9]+\] \.note\.GNU-stack +NOTE .*
#pass
