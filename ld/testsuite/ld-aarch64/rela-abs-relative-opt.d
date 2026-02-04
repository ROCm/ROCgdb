#name: rela-abs-relative --no-apply-dynamic-relocs
#source: rela-abs-relative.s
#target: [check_shared_lib_support]
#ld: -shared -Ttext-segment=0x100000 -Tdata=0x200000 -Trelocs.ld --no-apply-dynamic-relocs
#notarget: aarch64_be-*-*
#objdump: -sR -j .data

#...
0+200008 R_AARCH64_RELATIVE  \*ABS\*\+0x0*100ca

Contents of section \.data:
 200000 fecafeca 00000000 00000000 00000000 .*
 200010 addeadde 00000000                   .*
