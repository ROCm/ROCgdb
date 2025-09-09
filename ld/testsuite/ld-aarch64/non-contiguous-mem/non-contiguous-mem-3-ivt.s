	/* Interrupt vector table.  */
	.section  .vectors,"ax"
	.p2align 11  /* 2KB alignment (2^11) */
	.global vector_table2
vector_table2:
	b irqhandler2
	b irqhandler1
	.size vector_table2, . - vector_table2

	.global irqhandler2
	.type irqhandler2, %function
irqhandler2:
	b foo
	eret
	.size irqhandler2, . - irqhandler2
