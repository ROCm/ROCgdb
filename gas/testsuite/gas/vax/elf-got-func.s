# Taking the address of a global function in PIC code must go through
# the GOT even if the function is defined in this file; only calls and
# jumps may use the PLT.
	.text
	.globl	fn
	.type	fn, @function
fn:
	.word	0
	movab	fn, %r0
	pushab	fn
	calls	$0, fn
	callg	(%ap), fn
	jsb	fn
	jmp	fn
	ret
	.size	fn, . - fn
