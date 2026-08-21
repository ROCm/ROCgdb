# Skip annotate which reserved hint values such as 0xf, 0x1f and 0x7000.
#name: Check annotate of dbar
#source: annotate-dbar.s
#objdump: -d -M annotate

#...
Disassembly of section \.text:

0+ <.text>:
   0:	38720000 	dbar        	0x0	#  RW|RW
   4:	38720001 	dbar        	0x1	#  RW|R 
   8:	38720002 	dbar        	0x2	#  RW| W
   c:	38720003 	dbar        	0x3	#  RW|  
  10:	38720004 	dbar        	0x4	#  R |RW
  14:	38720005 	dbar        	0x5	#  R |R 
  18:	38720006 	dbar        	0x6	#  R | W
  1c:	38720007 	dbar        	0x7	#  R |  
  20:	38720008 	dbar        	0x8	#   W|RW
  24:	38720009 	dbar        	0x9	#   W|R 
  28:	3872000a 	dbar        	0xa	#   W| W
  2c:	3872000b 	dbar        	0xb	#   W|  
  30:	3872000c 	dbar        	0xc	#    |RW
  34:	3872000d 	dbar        	0xd	#    |R 
  38:	3872000e 	dbar        	0xe	#    | W
  3c:	3872000f 	dbar        	0xf
  40:	38720010 	dbar        	0x10	# CRW|RW
  44:	38720011 	dbar        	0x11	# CRW|R 
  48:	38720012 	dbar        	0x12	# CRW| W
  4c:	38720013 	dbar        	0x13	# CRW|  
  50:	38720014 	dbar        	0x14	# CR |RW
  54:	38720015 	dbar        	0x15	# CR |R 
  58:	38720016 	dbar        	0x16	# CR | W
  5c:	38720017 	dbar        	0x17	# CR |  
  60:	38720018 	dbar        	0x18	# C W|RW
  64:	38720019 	dbar        	0x19	# C W|R 
  68:	3872001a 	dbar        	0x1a	# C W| W
  6c:	3872001b 	dbar        	0x1b	# C W|  
  70:	3872001c 	dbar        	0x1c	# C  |RW
  74:	3872001d 	dbar        	0x1d	# C  |R 
  78:	3872001e 	dbar        	0x1e	# C  | W
  7c:	3872001f 	dbar        	0x1f
  80:	38720700 	dbar        	0x700	# SA-RAR
  84:	38727000 	dbar        	0x7000
