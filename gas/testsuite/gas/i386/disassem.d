#as: --divide
#objdump: -drw
#name: opcodes with invalid modrm byte

.*: +file format .*


Disassembly of section \.text:

0+ <\.text>:
[ 	]*[a-f0-9]+:[ 	]*ff[ 	]+ljmp[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*ef[ 	]*out    %eax,\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*ff[ 	]+lcall[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*d8 90 90 90 90 90[ 	]*fcoms  -0x6f6f6f70\(%eax\)
[ 	]*[a-f0-9]+:[ 	]*c5 ec 4a[ 	]+kaddw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ec 4a[ 	]+kaddw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ec 4a[ 	]+kaddw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ed 4a[ 	]+kaddb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ed 4a[ 	]+kaddb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ed 4a[ 	]+kaddb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 4a[ 	]+kaddq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 4a[ 	]+kaddq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 4a[ 	]+kaddq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 4a[ 	]+kaddd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 4a[ 	]+kaddd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 4a[ 	]+kaddd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ec 41[ 	]+kandw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ec 41[ 	]+kandw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ec 41[ 	]+kandw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ed 41[ 	]+kandb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ed 41[ 	]+kandb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ed 41[ 	]+kandb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 41[ 	]+kandq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 41[ 	]+kandq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 41[ 	]+kandq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 41[ 	]+kandd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 41[ 	]+kandd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 41[ 	]+kandd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ec 42[ 	]+kandnw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ec 42[ 	]+kandnw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ec 42[ 	]+kandnw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ed 42[ 	]+kandnb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ed 42[ 	]+kandnb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ed 42[ 	]+kandnb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 42[ 	]+kandnq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 42[ 	]+kandnq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 42[ 	]+kandnq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 42[ 	]+kandnd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 42[ 	]+kandnd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 42[ 	]+kandnd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ec 4b[ 	]+kunpckwd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ec 4b[ 	]+kunpckwd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ec 4b[ 	]+kunpckwd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ed 4b[ 	]+kunpckbw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ed 4b[ 	]+kunpckbw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ed 4b[ 	]+kunpckbw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 4b[ 	]+kunpckdq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 4b[ 	]+kunpckdq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 4b[ 	]+kunpckdq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 f8 44[ 	]+knotw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 f8 44[ 	]+knotw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 f8 44[ 	]+knotw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 f9 44[ 	]+knotb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 f9 44[ 	]+knotb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 f9 44[ 	]+knotb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f8 44[ 	]+knotq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f8 44[ 	]+knotq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f8 44[ 	]+knotq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 44[ 	]+knotd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 44[ 	]+knotd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 44[ 	]+knotd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ec 45[ 	]+korw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ec 45[ 	]+korw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ec 45[ 	]+korw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ed 45[ 	]+korb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ed 45[ 	]+korb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ed 45[ 	]+korb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 45[ 	]+korq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 45[ 	]+korq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 45[ 	]+korq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 45[ 	]+kord[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 45[ 	]+kord[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 45[ 	]+kord[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 f8 98[ 	]+kortestw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 f8 98[ 	]+kortestw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 f8 98[ 	]+kortestw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 f9 98[ 	]+kortestb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 f9 98[ 	]+kortestb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 f9 98[ 	]+kortestb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f8 98[ 	]+kortestq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f8 98[ 	]+kortestq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f8 98[ 	]+kortestq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 98[ 	]+kortestd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 98[ 	]+kortestd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 98[ 	]+kortestd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ec 46[ 	]+kxnorw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ec 46[ 	]+kxnorw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ec 46[ 	]+kxnorw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ed 46[ 	]+kxnorb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ed 46[ 	]+kxnorb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ed 46[ 	]+kxnorb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 46[ 	]+kxnorq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 46[ 	]+kxnorq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 46[ 	]+kxnorq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 46[ 	]+kxnord[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 46[ 	]+kxnord[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 46[ 	]+kxnord[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ec 47[ 	]+kxorw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ec 47[ 	]+kxorw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ec 47[ 	]+kxorw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 ed 47[ 	]+kxorb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 ed 47[ 	]+kxorb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 ed 47[ 	]+kxorb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 47[ 	]+kxorq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 47[ 	]+kxorq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ec 47[ 	]+kxorq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 47[ 	]+kxord[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 47[ 	]+kxord[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 ed 47[ 	]+kxord[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 f8 99[ 	]+ktestw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 f8 99[ 	]+ktestw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 f8 99[ 	]+ktestw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 f9 99[ 	]+ktestb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 f9 99[ 	]+ktestb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 f9 99[ 	]+ktestb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f8 99[ 	]+ktestq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f8 99[ 	]+ktestq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f8 99[ 	]+ktestq[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 99[ 	]+ktestd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 99[ 	]+ktestd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 99[ 	]+ktestd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 30 8f[ 	]+kshiftrw[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 30 6a[ 	]+kshiftrw[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 30 04[ 	]+kshiftrw[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 30 8f[ 	]+kshiftrb[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 30 6a[ 	]+kshiftrb[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 30 04[ 	]+kshiftrb[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 31 8f[ 	]+kshiftrq[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 31 6a[ 	]+kshiftrq[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 31 04[ 	]+kshiftrq[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 31 8f[ 	]+kshiftrd[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 31 6a[ 	]+kshiftrd[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 31 04[ 	]+kshiftrd[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 32 8f[ 	]+kshiftlw[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 32 6a[ 	]+kshiftlw[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 32 04[ 	]+kshiftlw[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 32 8f[ 	]+kshiftlb[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 32 6a[ 	]+kshiftlb[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 32 04[ 	]+kshiftlb[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 33 8f[ 	]+kshiftlq[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 33 6a[ 	]+kshiftlq[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 f9 33 04[ 	]+kshiftlq[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 33 8f[ 	]+kshiftld[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 33 6a[ 	]+kshiftld[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c4 e3 79 33 04[ 	]+kshiftld[ 	]*\$0x[0-9a-f]*,\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*07[ 	]+pop[ 	]+%es
[ 	]*[a-f0-9]+:[ 	]*c5 f8 92[ 	]+kmovw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 f8 92[ 	]+kmovw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 f8 92[ 	]+kmovw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 f9 92[ 	]+kmovb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 f9 92[ 	]+kmovb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 f9 92[ 	]+kmovb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 fb 92[ 	]+kmovd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 fb 92[ 	]+kmovd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 fb 92[ 	]+kmovd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 92[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 92[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 92[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 f8 93[ 	]+kmovw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 f8 93[ 	]+kmovw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 f8 93[ 	]+kmovw[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 f9 93[ 	]+kmovb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 f9 93[ 	]+kmovb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 f9 93[ 	]+kmovb[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c5 fb 93[ 	]+kmovd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c5 fb 93[ 	]+kmovd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c5 fb 93[ 	]+kmovd[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 93[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*9b[ 	]*fwait
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 93[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*6f[ 	]*outsl  %ds:\(%esi\),\(%dx\)
[ 	]*[a-f0-9]+:[ 	]*c4 e1 f9 93[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*3f[ 	]*aas
[ 	]*[a-f0-9]+:[ 	]*c4 e2 01 1c[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*41[ 	]*inc[ 	]*%ecx
[ 	]*[a-f0-9]+:[ 	]*37[ 	]*aaa
[ 	]*[a-f0-9]+:[ 	]*c4 e2 7f cc[ 	]+vsha512msg1[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*71 20[ 	]+jno.*
[ 	]*[a-f0-9]+:[ 	]*c4 e2 7f cd[ 	]+vsha512msg2[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*71 20[ 	]+jno.*
[ 	]*[a-f0-9]+:[ 	]*c4 e2 6f cb[ 	]+vsha512rnds2[ 	]*\(bad\),.*
[ 	]*[a-f0-9]+:[ 	]*71 20[ 	]+jno.*
[ 	]*[a-f0-9]+:[ 	]*62 f2 ad 08 1c[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*01 01[ 	]*add[ 	]*%eax,\(%ecx\)
[ 	]*[a-f0-9]+:[ 	]*62 f3 7d 28 1b[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*c8 25 62 f3[ 	]*enter[ ]*\$0x6225,\$0xf3
[ 	]*[a-f0-9]+:[ 	]*62 f3 75 08 23[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*c2 25 62[ 	]*ret[ ]*\$0x6225
[ 	]*[a-f0-9]+:[ 	]*62 f2 7d 28 5b[ 	]*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*41[ 	]*inc[ 	]*%ecx
[ 	]*[a-f0-9]+:[ 	]*37[ 	]*aaa
#pass
