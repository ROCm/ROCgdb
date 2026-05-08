	.data
	.global gd
	.internal gd
	.hidden gd
	.protected gd
gd:	.dc.b 0

	.weak wd
	.protected wd
	.hidden wd
	.internal wd
wd:	.dc.b 0

	.type gu, %gnu_unique_object
	.hidden gu
	.hidden gu
gu:	.dc.b 0

	.global ge
	.hidden ge
	.protected ge
	.internal ge

	.weak we
	.hidden we
	.protected we
	.p2align 3
	.dc.a we
