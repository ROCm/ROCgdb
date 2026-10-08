#name: FRV TLS relocs with addends, shared linking
#source: tls-2.s
#objdump: -DR -z -j .text -j .got -j .plt
#ld: -shared tmpdir/tls-1-dep.so --version-script tls-1-shared.lds

.*:     file format elf.*frv.*

Disassembly of section \.plt:

[0-9a-f ]+<\.plt>:
[0-9a-f ]+:	90 cc f0 10 	lddi @\(gr15,16\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc f0 20 	lddi @\(gr15,32\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc f0 40 	lddi @\(gr15,64\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc f0 48 	lddi @\(gr15,72\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc f0 58 	lddi @\(gr15,88\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc f0 60 	lddi @\(gr15,96\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc ff a0 	lddi @\(gr15,-96\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc ff b8 	lddi @\(gr15,-72\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc ff c8 	lddi @\(gr15,-56\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc ff d8 	lddi @\(gr15,-40\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc ff e0 	lddi @\(gr15,-32\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)
[0-9a-f ]+:	90 cc ff f0 	lddi @\(gr15,-16\),gr8
[0-9a-f ]+:	80 30 80 00 	jmpl @\(gr8,gr0\)

Disassembly of section \.text:

[0-9a-f ]+<_start>:
[0-9a-f ]+:	fe 3f ff f6 	call .*
[0-9a-f ]+:	fe 3f ff fb 	call .*
[0-9a-f ]+:	fe 3f ff e8 	call .*
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 b0 	setlo 0xb0,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 c0 	setlo 0xc0,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 78 	setlo 0x78,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc ff c0 	lddi\.p @\(gr15,-64\),gr8
[0-9a-f ]+:	9c fc ff c0 	setlos 0xf+c0,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc ff e8 	lddi\.p @\(gr15,-24\),gr8
[0-9a-f ]+:	9c fc ff e8 	setlos 0xf+e8,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc f0 28 	lddi\.p @\(gr15,40\),gr8
[0-9a-f ]+:	9c fc 00 28 	setlos 0x28,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc f8 14 	setlos 0xf+814,gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc 08 14 	setlos 0x814,gr9
[0-9a-f ]+:	92 f8 00 00 	sethi hi\(0x0\),gr9
[0-9a-f ]+:	92 f4 f8 14 	setlo 0xf814,gr9
[0-9a-f ]+:	fe 3f ff e0 	call .*
[0-9a-f ]+:	fe 3f ff cd 	call .*
[0-9a-f ]+:	fe 3f ff ce 	call .*
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 c8 	setlo 0xc8,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 a0 	setlo 0xa0,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 80 	setlo 0x80,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc ff f8 	lddi\.p @\(gr15,-8\),gr8
[0-9a-f ]+:	9c fc ff f8 	setlos 0xf+8,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc f0 68 	lddi\.p @\(gr15,104\),gr8
[0-9a-f ]+:	9c fc 00 68 	setlos 0x68,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc f0 50 	lddi\.p @\(gr15,80\),gr8
[0-9a-f ]+:	9c fc 00 50 	setlos 0x50,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc f8 24 	setlos 0xf+824,gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc 08 24 	setlos 0x824,gr9
[0-9a-f ]+:	92 f8 00 00 	sethi hi\(0x0\),gr9
[0-9a-f ]+:	92 f4 f8 24 	setlo 0xf824,gr9
[0-9a-f ]+:	fe 3f ff b8 	call .*
[0-9a-f ]+:	fe 3f ff bb 	call .*
[0-9a-f ]+:	fe 3f ff aa 	call .*
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 a8 	setlo 0xa8,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 b8 	setlo 0xb8,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 70 	setlo 0x70,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc ff a8 	lddi\.p @\(gr15,-88\),gr8
[0-9a-f ]+:	9c fc ff a8 	setlos 0xf+a8,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc ff d0 	lddi\.p @\(gr15,-48\),gr8
[0-9a-f ]+:	9c fc ff d0 	setlos 0xf+d0,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc f0 18 	lddi\.p @\(gr15,24\),gr8
[0-9a-f ]+:	9c fc 00 18 	setlos 0x18,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc 00 04 	setlos 0x4,gr9
[0-9a-f ]+:	80 88 00 00 	nop
[0-9a-f ]+:	92 fc 10 04 	setlos 0x1004,gr9
[0-9a-f ]+:	92 f8 00 01 	sethi 0x1,gr9
[0-9a-f ]+:	92 f4 00 04 	setlo 0x4,gr9
[0-9a-f ]+:	fe 3f ff 96 	call .*
[0-9a-f ]+:	fe 3f ff 97 	call .*
[0-9a-f ]+:	fe 3f ff 9e 	call .*
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 98 	setlo 0x98,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 88 	setlo 0x88,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	1c f8 00 00 	sethi\.p hi\(0x0\),gr14
[0-9a-f ]+:	9c f4 00 90 	setlo 0x90,gr14
[0-9a-f ]+:	90 08 f1 4e 	ldd @\(gr15,gr14\),gr8
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc f0 30 	lddi\.p @\(gr15,48\),gr8
[0-9a-f ]+:	9c fc 00 30 	setlos 0x30,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc f0 38 	lddi\.p @\(gr15,56\),gr8
[0-9a-f ]+:	9c fc 00 38 	setlos 0x38,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)
[0-9a-f ]+:	10 cc ff b0 	lddi\.p @\(gr15,-80\),gr8
[0-9a-f ]+:	9c fc ff b0 	setlos 0xf+b0,gr14
[0-9a-f ]+:	82 30 80 00 	calll @\(gr8,gr0\)

Disassembly of section \.got:

[0-9a-f ]+<.*>:
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 07 f1 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 07 f3 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	x
[0-9a-f ]+:	00 01 00 03 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 00 01 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 00 03 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 17 f1 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 17 f3 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	x
[0-9a-f ]+:	00 01 00 01 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 10 01 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 10 03 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 00 11 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 00 13 .*

[0-9a-f ]+<_GLOBAL_OFFSET_TABLE_>:
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 01 07 f1 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 01 07 f3 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 01 00 01 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 01 00 03 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	x
[0-9a-f ]+:	00 00 00 03 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	x
[0-9a-f ]+:	00 00 10 03 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 10 11 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 01 00 11 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 01 00 13 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	x
[0-9a-f ]+:	00 00 00 01 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	x
[0-9a-f ]+:	00 00 10 01 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 10 13 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 01 07 f2 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 01 00 02 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 01 00 12 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	x
[0-9a-f ]+:	00 00 10 02 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	x
[0-9a-f ]+:	00 01 00 02 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	x
[0-9a-f ]+:	00 00 00 02 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 10 12 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 07 f2 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 00 02 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 17 f2 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 10 02 .*
[0-9a-f ]+:	00 00 00 00 .*
[0-9a-f	]+: R_FRV_TLSDESC_VALUE	\.tbss
[0-9a-f ]+:	00 00 00 12 .*
