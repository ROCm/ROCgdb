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
	.insn rie,0xec0000000045,%r1,%r2,test_rie	# brxlg %r1,%r2,test_rie
	.insn rie_a,0xec0000000072,%r1,-1,7		# cit %r1,-1,7
	.insn rie_b,0xec0000000076,%r1,%r2,7,.		# crj %r1,%r2,7,.
	.insn rie_c,0xec000000007e,%r1,-1,7,.		# cij %r1,-1,7,.
	.insn rie_d,0xec00000000d8,%r1,-1,%r3		# ahik %r1,%r3,-1
	.insn rie_e,0xec0000000044,%r1,.,%r3		# brxhg %r1,%r3,.
	.insn rie_f,0xec0000000054,%r1,%r2,3,4,5	# rnsbg %r1,%r2,3,4,5
	.insn rie_g,0xec0000000042,%r1,-1,15		# lochi %r1,-1,15
	.insn ril,0xc00500000000,%r14,test_ril		# brasl %r14,test_ril
	.insn rilu,0xc20900000000,%r1,1			# afi %r1,1
	.insn ril_a,0xc20900000000,%r1,-1		# afi %r1,-1
	.insn ril_b,0xc00500000000,%r14,.		# brasl %r14,.
	.insn ril_c,0xc00400000000,0b1111,.		# brcl 0b1111,.
	.insn ris,0xec00000000fe,%r1,-2,3,444(%r4)	# cib %r1,-2,3,444(%r4)
	.insn rr,0x1a00,%r1,%r2				# ar %r1,%r2
	.insn rr,0x0400,%r1				# spm %r1
	.insn rse,0xeb000000000d,%r1,%r2,3(%r4)
	.insn ssf,0xc80000000000,1(%r2),3(%r4),%r5
