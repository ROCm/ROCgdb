#source: relax-uleb128.s
#source: lcall2.s
#ld: -T lcall.t
#objdump: -s -j .debug_info
#name: uleb128 difference after longcall relax

# Longcall relaxes from 6 bytes to 3; with the trailing nop the difference
# is 5.  Assembler reserved one spare byte (two total).
#...
 0000 8500.*
#...
