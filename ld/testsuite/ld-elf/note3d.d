#name: .note.GNU-stack ld -r SHT_PROGBITS + SHT_NOTE (negative)
#source: note3.s
#source: empty.s --noexecstack
#as: --generate-missing-build-notes=no
#ld: -r
#readelf: -SW
#failif
# cr16 and crx force entirely empty scripts for -r, while hppa-elf re-uses
# a significantly shrunk down script also for -r.
#xfail: cr16-*-elf* crx-*-elf* hppa-*-*elf* hppa*-*-lites*

#...
 *\[[ 0-9]+\] \.note\.GNU-stack +NOTE .*
#pass
