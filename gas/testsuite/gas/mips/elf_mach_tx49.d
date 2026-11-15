#readelf: -Ah
#name: ELF TX49 markings
#as: -32 -march=tx49
#source: empty.s

ELF Header:
#...
  Flags: +0x..86...., .*tx49.*
#...

MIPS ABI Flags Version: 0

ISA: MIPS3
GPR size: 32
CPR1 size: 32
CPR2 size: 0
FP ABI: .*
ISA Extension: Toshiba TX49
ASEs:
	None
FLAGS 1: .*
FLAGS 2: .*
