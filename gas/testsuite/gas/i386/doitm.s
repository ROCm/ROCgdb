# Check 32bit DOITM instructions

	.text
_start:
	rddoitm	%eax
	setdoitm
	wrdoitm	%ebx

_intel:
	.intel_syntax noprefix
	rddoitm	eax
	setdoitm
	wrdoitm	ebx
