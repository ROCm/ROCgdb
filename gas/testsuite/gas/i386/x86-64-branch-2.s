	.text
	data16 jmp foo
	data16 rex.w jmp foo

bar:
	mov %eax, %ebx

	data16 call foo
	data16 rex.w call foo

	retw
	retw $8

	data16 je foo
	data16 rex.w jne foo
