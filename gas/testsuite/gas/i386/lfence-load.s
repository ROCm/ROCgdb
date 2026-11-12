	.text
_start:
	vldmxcsr (%ebp)
	lgdt (%ebp)
	lidt (%ebp)
	vmptrld (%ebp)
	vmptrst (%ebp)
	vmclear (%ebp)
	vmxon (%ebx)
	vmread %ecx, (%ebx)
	invpcid (%ebp), %edx
	invlpg (%ebp)
	clflush (%ebp)
	clflushopt (%ebp)
	clwb (%ebp)
	cldemote (%ebp)
	bndmk (%ebp), %bnd1
	bndcl (%ebp), %bnd1
	bndcu (%ebp), %bnd1
	bndcn (%ebp), %bnd1
	bndstx %bnd1, (%ebp)
	bndldx (%ebp), %bnd1
	prefetcht0 (%ebp)
	prefetcht1 (%ebp)
	prefetcht2 (%ebp)
	prefetchw (%ebp)
	prefetchrst2 (%ebp)
	pop %ds
	popf
	popa
	xlatb (%ebx)
	lldt (%edx)
	verr (%edx)
	fsts (%ebp)
	flds (%ebp)
	fistl (%ebp)
	fists (%ebp)
	fildl (%ebp)
	filds (%ebp)
	fsave (%ebp)
	frstor (%ebp)
	filds (%ebp)
	fisttps (%ebp)
	fldenv (%ebp)
	fstenv (%ebp)
	fadds  (%ebp)
	fadds  (%esp)
	fadd  %st(3),%st
	fadds (%ecx)
	filds (%ecx)
	fists (%ecx)
	xrstor (%ecx)
	prefetchnta (%ecx)
	cmpxchg8b (%ecx)
	incl	%ecx
	lgdt (%eax)
	pfcmpeq 2(%esi),%mm4
	popl (%eax)
	popl %eax
	rclw	(%ecx)
	rcrb	$0,(%ecx)
	testl	$1,(%ecx)
	incl	(%ecx)
	notl	(%ecx)
	divl	(%ecx)
	mull	(%ecx)
	idivl	(%ecx)
	imull	(%ecx)
	leal	(%eax,%eax,2), %eax
	leave
	outsb
	lodsb
	rep movsl
	rep scasl
	rep cmpsl
	rep lodsl
	addl $1, (%eax)
	btl $1, (%eax)
	bt %eax, (%ecx)
	shld $0, %eax, (%ecx)
	shld $1, %eax, (%ecx)
	shrd %cl, %eax, (%ecx)
	cmpxchg %eax, (%ebx)
	xadd %eax,(%ebx)
	xadd %eax,%ebx
	xchg %eax,(%ebx)
	xchg %eax,%ebx
	cmp    %eax,0x40(%ebp)
	cmp    0x40(%ebp),%eax
	add %eax,0x40(%ebp)
	add (%eax),%eax
	test %eax,0x40(%ebp)
	test 0x40(%ebp),%eax
	arpl	%eax, (%ecx)
