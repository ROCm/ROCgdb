/* Trivial object used by stub-file-retarget.exp.

   The object is deliberately not an AVR or m68hc11 object.  The bugs
   under test happen before the input architecture matters: the
   emulation hook dereferences a NULL stub_file, and the link dies in
   the hook.  An x86-64 object keeps the test buildable by any
   assembler.  */

	.text
	.globl	_start
	.type	_start, @function
_start:
	nop
	.size	_start, .-_start
