    .section .text

    .globl _start
	.type	_start, %function
_start:
.ifdef TEST_CM_JT
    .rept 5
    tail foo_local
    tail foo_global
    .endr
.endif

.ifdef TEST_CM_JALT
    .rept 70
    call foo_global
    call foo_local
    .endr
.endif

.ifdef TEST_CM_JT_JALT
    .rept 5
    tail foo_global
    .endr
    .rept 140
    call foo_global
    .endr
.endif

	# Keep paired JAL candidates out of C.J range.  Zcmt profiling runs
	# after ordinary shortening, so near paired JALs should become C.J
	# and no longer be Zcmt candidates.
.ifdef TEST_CM_JT
    .space 4096
.endif
.ifdef TEST_CM_JT_JALT
    .space 4096
.endif

    .globl  foo_global
	.type	foo_global, %function
foo_global:
    ret

	.type	foo_local, %function
foo_local:
    ret
