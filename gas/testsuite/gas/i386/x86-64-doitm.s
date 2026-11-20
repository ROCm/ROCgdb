# Check 64bit DOITM instructions

	.text
_start:
	rddoitm	%r12
	rddoitm	%rbx
	rddoitm	%r22
	setdoitm
	wrdoitm	%r14
	wrdoitm	%rcx
	wrdoitm	%r24

_intel:
	.intel_syntax noprefix
	rddoitm	r12
	rddoitm	rbx
	rddoitm	r22
	setdoitm
	wrdoitm	r14
	wrdoitm	rcx
	wrdoitm	r24
