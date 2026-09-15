#source: relax-uleb128-2.s
#source: lcall2.s
#ld: -T lcall.t
#objdump: -s -j .debug_info
#name: uleb128 difference shrinking below 128

# The difference is 156 before relaxation and 108 after it, so the value no
# longer needs two bytes.  The assembler reserved one spare byte (three
# total), and the linker must keep that fixed width with uleb128 padding.
#...
 0000 ec8000.*
#...
