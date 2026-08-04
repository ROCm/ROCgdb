# Check R_X86_64_PLT32 relocation in jump table.

        .text
        .p2align 4
        .globl	foo
        .type	foo, @function
foo:
	.cfi_startproc
        cmpl	$4, %edi
        ja	.L1
        leaq	.L4(%rip), %rdx
        movl	%edi, %edi
        movslq	(%rdx,%rdi,4), %rax
        addq	%rdx, %rax
        jmp	*%rax
.L1:
	ret
.Lbar2:
	jmp	bar2@PLT
	.cfi_endproc
        .size	foo, .-foo
        .section	.rodata
        .p2align 2
.L4:
        .long	bar0@plt-.L4
        .long	bar1@PLT-.L4
        .long	.Lbar2-.L4
        .long	bar3@PLT-.L4
        .long	bar4@plt-.L4
        .section	.note.GNU-stack,"",@progbits
