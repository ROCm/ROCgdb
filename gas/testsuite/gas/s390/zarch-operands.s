.text
foo:
	.insn e,0x0101					# pr
	.insn i,0x0a00,1				# svc 1
	.insn ie,0xb2fa0000,1,2				# niai 1,2
	.insn mii,0xc50000000000,1,test_mii,test_mii	# bprp 1,test_mii,test_mii
	.insn ri,0xa70a0000,%r1,1			# ahi %r1,1
	.insn ri_a,0xa70a0000,%r1,-1			# ahi %r1,-1
	.insn ri_b,0xa7060000,%r1,test_ri_b		# brct %r1,test_ri_b
	.insn ri_c,0xa7040000,15,test_ri_c		# brc 15,test_ri_c
	.insn rie,0xec0000000045,%r1,%r2,test_rie
	.insn ril,0xc00500000000,%r14,test_ril
	.insn rse,0xeb000000000d,%r1,%r2,3(%r4)
	.insn ssf,0xc80000000000,1(%r2),3(%r4),%r5
