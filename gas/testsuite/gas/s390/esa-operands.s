.text
foo:
	.insn e,0x0101					# pr
	.insn i,0x0a00,1				# svc 1
	.insn ri,0xa70a0000,%r1,-32767			# ahi %r1,-32767
	.insn rr,0x1800,%r1,%r2				# lr %r1,%r2
	.insn rre,0xb25e0000,%r1,%r2			# srst %r1,%r2
	.insn rrf,0xb35b0000,%f1,%f2,%f3,4		# didbr %f1,%f3,%f2,4
	.insn rs,0xba000000,%r1,%r2,3(%r4)		# cs %r1,%r2,3(%r4)
	.insn rsi,0x84000000,%r1,%r2,test_rsi		# brxh %r1,%r2,test_rsi
	.insn rx,0x58000000,%r1,2(%r3,%r4)		# l %r1,2(%r3,%r4)
	.insn rxe,0xed000000001a,%f1,2(%r3)		# adb %f1,2(%r3)
	.insn rxf,0xed000000001e,%f1,%f2,3(%r4,%r5)	# madb %f1,%f2,3(%r4,%r5)
	.insn s,0xb2330000,1(%r2)			# ssch 1(%r2)
	.insn si,0x92000000,1(%r2),3			# mvi 1(%r2),3
	.insn ss,0xd20000000000,1(2,%r3),4(%r5),6
	.insn sse,0xe50100000000,1(%r2),3(%r4)
