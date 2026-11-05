#as: -march=rv64gc
#source: dis-arch-option.s RUN_OBJCOPY
#objcopy_objects: --strip-all
#objdump: -d -M arch=rv64gcv
# With no mapping symbols left, "arch=" overrides the ISA recorded in
# the architecture attribute.

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <.text>:
[ 	]+0:[ 	]+022180d7[ 	]+vadd.vv[ 	]+v1,v2,v3
[ 	]+4:[ 	]+022180d7[ 	]+vadd.vv[ 	]+v1,v2,v3
