#as: -mrelax
#objdump: -d
#warning_output: negative_right_shift.l

#...
.*03400000 	nop
.*00000001 	\.word		0x00000001
.*00000001 	\.word		0x00000001
#...
.*02bff9ac 	addi.w      	\$t0, \$t1, -2
.*02fff9ac 	addi.d      	\$t0, \$t1, -2
.*13fff9ac 	addu16i.d   	\$t0, \$t1, -2
.*15ffffcc 	lu12i.w     	\$t0, -2
.*17ffffcc 	lu32i.d     	\$t0, -2
.*033ff9ac 	lu52i.d     	\$t0, \$t1, -2
.*023ff9ac 	slti        	\$t0, \$t1, -2
.*027ff9ac 	sltui       	\$t0, \$t1, -2
.*19ffffcc 	pcaddi      	\$t0, -2
.*1dffffcc 	pcaddu12i   	\$t0, -2
.*1fffffcc 	pcaddu18i   	\$t0, -2
.*1bffffcc 	pcalau12i   	\$t0, -2
.*02bffdac 	addi.w      	\$t0, \$t1, -1
.*02fffdac 	addi.d      	\$t0, \$t1, -1
.*13fffdac 	addu16i.d   	\$t0, \$t1, -1
.*15ffffec 	lu12i.w     	\$t0, -1
.*17ffffec 	lu32i.d     	\$t0, -1
.*033ffdac 	lu52i.d     	\$t0, \$t1, -1
.*023ffdac 	slti        	\$t0, \$t1, -1
.*027ffdac 	sltui       	\$t0, \$t1, -1
.*19ffffec 	pcaddi      	\$t0, -1
.*1dffffec 	pcaddu12i   	\$t0, -1
.*1fffffec 	pcaddu18i   	\$t0, -1
.*1bffffec 	pcalau12i   	\$t0, -1
