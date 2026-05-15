	.intel_syntax noprefix

	a = a
	mov eax, [a]

	b = c
	c = b
	mov eax, [b]
	mov eax, [c]

	d = e
	e = d
	x = d
	d = 1
	e = 2
	mov eax, [x]
