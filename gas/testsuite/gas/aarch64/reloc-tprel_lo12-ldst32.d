#as: -mabi=lp64
#objdump: -dr
# This test is only valid on ELF based ports.
#notarget: *-*-*coff *-*-pe *-*-wince *-*-*aout* *-*-netbsd

.*:     file format .*

Disassembly of section \.text:

0000000000000000 <.*>:
   0:	b980009b 	ldrsw	x27, \[x4\]
			0: R_AARCH64_TLSLE_LDST32_TPREL_LO12	sym
