#readelf: -Ah
#name: ELF m6201 markings
#as: -32 -march=m6201
#source: empty.s

ELF Header:
#...
  Flags: +0x9......., .*mips32r6.*
#...

MIPS ABI Flags Version: 0

ISA: MIPS32r6
GPR size: 32
CPR1 size: 64
CPR2 size: 0
FP ABI: .*
ISA Extension: None
ASEs:
	MCU \(MicroController\) ASE
FLAGS 1: .*
FLAGS 2: .*
