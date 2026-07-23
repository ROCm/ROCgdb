#objdump: -h -t
#name: PE large normal object after raising the normal section limit

# This only becomes meaningful once normal PE/COFF is allowed past the
# old 32767-section cutoff.  Before that, BFD errors out or auto-promotes
# to bigobj before the normal 16-bit symbol-section encoding is exercised.

.*: *file format pe-(aarch64-little|i386|x86-64)

Sections:
#...
40002 +\.data\$a39999  .*
                  CONTENTS, ALLOC, LOAD, DATA

SYMBOL TABLE:
#...
.*\(sec 40003\).*\(scl +2\).*\) 0x[0-9a-f]+ a39999$
