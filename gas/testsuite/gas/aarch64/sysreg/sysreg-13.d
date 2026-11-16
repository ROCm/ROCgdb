#as: -I$srcdir/$subdir
#objdump: -dr

.*:     file format .*

[^:]+:

0+ <.*>:
.*:	d5390060 	mrs	x0, ccontidr_el1
