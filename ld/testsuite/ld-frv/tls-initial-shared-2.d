#name: FRV TLS relocs with addends, shared linking with static TLS
#source: tls-2.s
#as: --defsym static_tls=1
#objdump: -DR -z -j .text -j .got -j .plt
#ld: -shared tmpdir/tls-1-dep.so --version-script tls-1-shared.lds

.*:     file format elf.*frv.*

Disassembly of section \.plt:

[0-9a-f ]+<\.plt>:
[0-9a-f ]+:	92 c8 f0 60 	ldi @\(gr15,96\),gr9
[0-9a-f ]+:	c0 3a 40 00 	bralr
[0-9a-f ]+:	90 cc ff e8 	lddi @\(gr15,-24\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc ff f0 	lddi @\(gr15,-16\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	92 c8 f0 88 	ldi @\(gr15,136\),gr9
[0-9a-f ]+:	c0 3a 40 00 	bralr

Disassembly of section \.text:

[0-9a-f ]+<_start>:
[0-9a-f ]+:	92 c8 f0 48 	ldi @\(gr15,72\),gr9
[0-9a-f ]+:	92 c8 f0 88 	ldi @\(gr15,136\),gr9
[0-9a-f ]+:	92 c8 f0 14 	ldi @\(gr15,20\),gr9
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 80 	ldi @\(gr15,128\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 8c 	ldi @\(gr15,140\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 64 	ldi @\(gr15,100\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 4c 	ldi\.p @\(gr15,76\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 54 	ldi\.p @\(gr15,84\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 18 	ldi\.p @\(gr15,24\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc f8 14 	setlos 0xf+814,gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc 08 14 	setlos 0x814,gr9
[0-9a-f ]+:	92 f8 00 00 	sethi hi\(0x0\),gr9
[0-9a-f ]+:	92 f4 f8 14 	setlo 0xf814,gr9
[0-9a-f ]+:	92 c8 f0 40 	ldi @\(gr15,64\),gr9
[0-9a-f ]+:	92 c8 f0 60 	ldi @\(gr15,96\),gr9
[0-9a-f ]+:	92 c8 f0 20 	ldi @\(gr15,32\),gr9
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 90 	ldi @\(gr15,144\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 78 	ldi @\(gr15,120\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 68 	ldi @\(gr15,104\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 58 	ldi\.p @\(gr15,88\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 30 	ldi\.p @\(gr15,48\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 24 	ldi\.p @\(gr15,36\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc f8 24 	setlos 0xf+824,gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc 08 24 	setlos 0x824,gr9
[0-9a-f ]+:	92 f8 00 00 	sethi hi\(0x0\),gr9
[0-9a-f ]+:	92 f4 f8 24 	setlo 0xf824,gr9
[0-9a-f ]+:	92 c8 f0 38 	ldi @\(gr15,56\),gr9
[0-9a-f ]+:	fe 3f ff bd 	call .*
[0-9a-f ]+:	92 c8 f0 0c 	ldi @\(gr15,12\),gr9
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 7c 	ldi @\(gr15,124\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 84 	ldi @\(gr15,132\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 5c 	ldi @\(gr15,92\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 3c 	ldi\.p @\(gr15,60\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 50 	ldi\.p @\(gr15,80\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 10 	ldi\.p @\(gr15,16\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc 00 04 	setlos 0x4,gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc 10 04 	setlos 0x1004,gr9
[0-9a-f ]+:	92 f8 00 01 	sethi 0x1,gr9
[0-9a-f ]+:	92 f4 00 04 	setlo 0x4,gr9
[0-9a-f ]+:	92 c8 f0 2c 	ldi @\(gr15,44\),gr9
[0-9a-f ]+:	fe 3f ff a1 	call .*
[0-9a-f ]+:	92 c8 f0 28 	ldi @\(gr15,40\),gr9
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 74 	ldi @\(gr15,116\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 6c 	ldi @\(gr15,108\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	00 88 00 00 	nop\.p
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 70 	ldi @\(gr15,112\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 34 	ldi\.p @\(gr15,52\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 1c 	ldi\.p @\(gr15,28\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	12 c8 f0 44 	ldi\.p @\(gr15,68\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 48 	ldi @\(gr15,72\),gr9
[0-9a-f ]+:	92 c8 f0 14 	ldi @\(gr15,20\),gr9
[0-9a-f ]+:	92 c8 f0 40 	ldi @\(gr15,64\),gr9
[0-9a-f ]+:	92 c8 f0 20 	ldi @\(gr15,32\),gr9
[0-9a-f ]+:	92 c8 f0 38 	ldi @\(gr15,56\),gr9
[0-9a-f ]+:	92 c8 f0 0c 	ldi @\(gr15,12\),gr9
[0-9a-f ]+:	92 c8 f0 2c 	ldi @\(gr15,44\),gr9
[0-9a-f ]+:	92 c8 f0 28 	ldi @\(gr15,40\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 88 	ldi @\(gr15,136\),gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 c8 f0 60 	ldi @\(gr15,96\),gr9

Disassembly of section \.got:

[0-9a-f ]+<\.got>:
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 10 11 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 17 f1 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	x
[0-9a-f ]+:	00 00 10 01 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 10 01 .*

[0-9a-f ]+<_GLOBAL_OFFSET_TABLE_>:
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f ]+:	00 01 07 f1 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 01 07 f3 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 01 00 01 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 01 00 03 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 10 03 .*
[0-9a-f	]+: R_FRV_TLSOFF	x
[0-9a-f ]+:	00 01 00 11 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 01 00 13 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 01 00 01 .*
[0-9a-f	]+: R_FRV_TLSOFF	x
[0-9a-f ]+:	00 00 00 01 .*
[0-9a-f	]+: R_FRV_TLSOFF	x
[0-9a-f ]+:	00 00 10 13 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 00 03 .*
[0-9a-f	]+: R_FRV_TLSOFF	x
[0-9a-f ]+:	00 00 07 f1 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 07 f3 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 00 11 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 01 00 03 .*
[0-9a-f	]+: R_FRV_TLSOFF	x
[0-9a-f ]+:	00 00 00 01 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 00 03 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 17 f3 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 10 03 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 00 13 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 01 07 f2 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 10 11 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 01 00 02 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 01 00 12 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 10 02 .*
[0-9a-f	]+: R_FRV_TLSOFF	x
[0-9a-f ]+:	00 01 00 02 .*
[0-9a-f	]+: R_FRV_TLSOFF	x
[0-9a-f ]+:	00 00 00 02 .*
[0-9a-f	]+: R_FRV_TLSOFF	x
[0-9a-f ]+:	00 00 10 12 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 07 f2 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 00 02 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 17 f2 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 10 01 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 10 02 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
[0-9a-f ]+:	00 00 00 12 .*
[0-9a-f	]+: R_FRV_TLSOFF	\.tbss
