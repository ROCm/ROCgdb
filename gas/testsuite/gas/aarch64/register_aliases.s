	# test register aliases.
	lr	.req 	x30
	fp 	.req 	x29
	ip0	.req 	x16
	ip1	.req 	x17
	zero	.req	xzr
	add	ip0, ip0, lr
	str 	ip0, [fp]
	ldr	ip1, [fp]
	str 	IP0, [fp]
	ldr	IP1, [fp]
	str	zero, [x0]

	# Removal of user-defined aliases.
	foo	.req	x13
	.unreq foo

	# Removing builtin register aliases should not segfault.
	.unreq lr
	.unreq fp
	.unreq ip0
	.unreq ip1
