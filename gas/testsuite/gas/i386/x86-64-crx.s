.text
_start:
	movq	%cr8, %rax
	movq	%cr8, %rdi
	movq	%rax, %cr8
	movq	%rdi, %cr8

	lock; mov %cr0, %rcx
	lock; mov %cr8, %rcx

.att_syntax noprefix
	movq	cr8, rax
	movq	cr8, rdi
	movq	rax, cr8
	movq	rdi, cr8

.intel_syntax noprefix
	mov	rax, cr8
	mov	rdi, cr8
	mov	cr8, rax
	mov	cr8, rdi
