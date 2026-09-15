# Check RMPCHKD instruction

	.text
att:
        rmpchkd
        rmpchkd %rax, %rcx
        rmpchkd %eax, %rcx

	.intel_syntax noprefix
intel:
        rmpchkd
        rmpchkd rax, rcx
        rmpchkd eax, rcx
