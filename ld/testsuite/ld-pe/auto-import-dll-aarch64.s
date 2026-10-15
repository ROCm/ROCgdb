	.data
	.global	dyn_v
	.align	3
	.def	dyn_v;	.scl	2;	.type	0;	.endef
dyn_v:
	.space	8

	.text
	.align	2
	.global	init_dyn_v
	.def	init_dyn_v;	.scl	2;	.type	32;	.endef
init_dyn_v:
	ret
