	.data
	.long 0
x:	.long 1, 2, . - x
 y = . - x
 z == . - x
	.long y
	.long z
	.long y
	.long z
