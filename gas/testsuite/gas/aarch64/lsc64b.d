#name: LSC64B instructions
#objdump: -dr

.*:     file format .*

Disassembly of section \.text:

0+ <.*>:
.*:	f8bfd000 	lda64b	x0, \[x0\]
.*:	f8bfd3e0 	lda64b	x0, \[sp\]
.*:	f8bfd020 	lda64b	x0, \[x1\]
.*:	f8bfd020 	lda64b	x0, \[x1\]
.*:	f8bfd02e 	lda64b	x14, \[x1\]
.*:	f8bfd036 	lda64b	x22, \[x1\]
.*:	f87f9000 	stl64b	x0, \[x0\]
.*:	f87f93e0 	stl64b	x0, \[sp\]
.*:	f87f9020 	stl64b	x0, \[x1\]
.*:	f87f9020 	stl64b	x0, \[x1\]
.*:	f87f902e 	stl64b	x14, \[x1\]
.*:	f87f9036 	stl64b	x22, \[x1\]
.*:	f860b000 	stl64bv	x0, x0, \[x0\]
.*:	f87fb000 	stl64bv	xzr, x0, \[x0\]
.*:	f860b3e0 	stl64bv	x0, x0, \[sp\]
.*:	f861b040 	stl64bv	x1, x0, \[x2\]
.*:	f860b04e 	stl64bv	x0, x14, \[x2\]
.*:	f860b056 	stl64bv	x0, x22, \[x2\]
.*:	f860a000 	stl64bv0	x0, x0, \[x0\]
.*:	f87fa000 	stl64bv0	xzr, x0, \[x0\]
.*:	f860a3e0 	stl64bv0	x0, x0, \[sp\]
.*:	f861a040 	stl64bv0	x1, x0, \[x2\]
.*:	f860a04e 	stl64bv0	x0, x14, \[x2\]
.*:	f860a056 	stl64bv0	x0, x22, \[x2\]
.*:	d940581f 	ldap	xzr, x0, \[x0\]
.*:	d940781f 	ldapp	xzr, x0, \[x0\]
.*:	d9005800 	stlp	x0, x0, \[x0\]
