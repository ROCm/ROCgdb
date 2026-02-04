#name: rela-abs-relative (big endian)
#source: rela-abs-relative.s
#alltargets: [check_shared_lib_support] aarch64_be-*-*
#ld: -shared -Ttext-segment=0x100000 -Tdata=0x200000 -Trelocs.ld
#objdump: -sR -j .data

#...
0+200008 R_AARCH64_RELATIVE  \*ABS\*\+0x0*100ca

Contents of section \.data:
 200000 00000000 cafecafe 00000000 000100ca .*
 200010 00000000 deaddead                   .*
