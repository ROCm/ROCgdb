#as: -I$srcdir/$subdir
#objdump: -dr

.*:     file format .*

[^:]+:

0+ <.*>:
.*:	d51b2280 	msr	tpcr0_el0, x0
.*:	d53b2280 	mrs	x0, tpcr0_el0
.*:	d5182280 	msr	tpcr0_el1, x0
.*:	d5382280 	mrs	x0, tpcr0_el1
.*:	d51d2280 	msr	tpcr0_el12, x0
.*:	d53d2280 	mrs	x0, tpcr0_el12
.*:	d51c2280 	msr	tpcr0_el2, x0
.*:	d53c2280 	mrs	x0, tpcr0_el2
.*:	d51b22a0 	msr	tpcr1_el0, x0
.*:	d53b22a0 	mrs	x0, tpcr1_el0
.*:	d51822a0 	msr	tpcr1_el1, x0
.*:	d53822a0 	mrs	x0, tpcr1_el1
.*:	d51d22a0 	msr	tpcr1_el12, x0
.*:	d53d22a0 	mrs	x0, tpcr1_el12
.*:	d51c22a0 	msr	tpcr1_el2, x0
.*:	d53c22a0 	mrs	x0, tpcr1_el2
