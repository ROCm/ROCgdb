#name: .note.GNU-stack ld -r SHT_NOTE + SHT_PROGBITS (negative)
#source: empty.s --noexecstack
#source: note3.s
#as: --generate-missing-build-notes=no
#ld: -r
#readelf: -SW
#failif
# cr16 and crx force entirely empty scripts for -r, while hppa-elf re-uses
# a significantly shrunk down script also for -r.
#xfail: cr16-*-elf* crx-*-elf* hppa-*-*elf* hppa*-*-lites*

#...
 *\[[ 0-9]+\] \.note\.GNU-stack +PROGBITS .*
#pass
