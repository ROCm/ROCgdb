#source: eqv-dot.s
#objdump: -s -j .data
#name: eqv involving dot (PDP11)
# Special for PDP11 which is little-endian for octets in shorts
# but big-endian for shorts in longs per register assignments for
# mul/div and in by convention in memory (at least for Unix).

.*: .*

Contents of section \.data:
 0000 0+0000 0+0100 0+0200 0+0800  .*
 0010 0+0c00 0+1000 0+0c00 0+1800  .*
#pass
