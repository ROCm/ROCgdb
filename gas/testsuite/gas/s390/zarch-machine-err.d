#name: s390x machine error
#as: -march=z900
#error: 4: Error: Unrecognized opcode: `ppa'.*
#error: 6: Error: Unrecognized opcode: `ppa'.*
#error: 8: Error: Unrecognized opcode: `ppa'.*
#objdump: -dr
