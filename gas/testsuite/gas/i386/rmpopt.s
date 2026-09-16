# Check RMPOPT instruction

	.text
att:
        rmpopt
        rmpopt %rax, %rcx
        rmpopt %eax, %rcx

	.intel_syntax noprefix
intel:
        rmpopt
        rmpopt rax, rcx
        rmpopt eax, rcx
