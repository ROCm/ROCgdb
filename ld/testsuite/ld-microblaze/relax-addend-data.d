#source: relax-addend.s
#source: relax-addend-support.s
#as: -EL
#ld: -EL -relax --gc-sections -T $srcdir/$subdir/relax-addend.ld
#readelf: -x .checkdata
#name: MicroBlaze relaxation preserves R_MICROBLAZE_32 addends

# The same check for the R_MICROBLAZE_32 arm of the same loop, using a data
# word rather than an instruction so that readelf alone can verify it.
# .checkdata holds a single word initialised to gvar + 0x18; with .data pinned
# by the linker script that is 0x90001018, little endian 18 10 00 90.
# A linker which corrupts the addend stores 14 10 00 90.

#...
Hex dump of section '.checkdata':
[ 	]*0x90001100 18100090 .*
#pass
