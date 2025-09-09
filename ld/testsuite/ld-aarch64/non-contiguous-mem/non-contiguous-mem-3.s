	.section .boot, "ax", %progbits
	.p2align 11  /* 2KB alignment (2^11) */
	# Fit in RAML
	.global bootloader
	.type bootloader, %function
bootloader:
	nop
	nop
	bl code2
	.size bootloader, . - bootloader


	/* Interrupt vector table.  */
	.section  .vectors,"ax"
	.p2align 11  /* 2KB alignment (2^11) */
	.global vector_table1
vector_table1:
	b irqhandler1
	.size vector_table1, . - vector_table1

	.global irqhandler1
	.type irqhandler1, %function
irqhandler1:
	eret
	.size irqhandler1, . - irqhandler1


	.section .code.2, "ax", %progbits
	# Fit in RAMU
	.global code2
	.type code2, %function
code2:
	nop
	nop
	bl code3
	.size code2, . - code2


	/* Interrupt vector table.  */
	.section  .vectors
	.global vector_table3
vector_table3:
	b irqhandler3
	.size vector_table3, . - vector_table3

	.global irqhandler3
	.type irqhandler3, %function
irqhandler3:
	eret
	.size irqhandler3, . - irqhandler3

	.section .code.3, "ax", %progbits
	# Fit in RAMU
	.global code3
	.type code3, %function
code3:
	nop
	bl code4
	.size code3, . - code3

	.section .code.4, "ax", %progbits
	# Fit in RAMU
	.global code4
	.type code4, %function
code4:
$x:
	# fill with NOPs
	.fill 20, 4, 0xd503201f
	.size code4, . - code4

	.global foo
	.type foo, %function
foo:
	ret
	.size foo, . - foo
