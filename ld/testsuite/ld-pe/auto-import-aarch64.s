	.text
	.global main
main:
	adrp	x0, dyn_v
	add	x0, x0, :lo12:dyn_v

	adrp	x0, dyn_v
	ldr	x0, [x0, :lo12:dyn_v]

	adrp	x0, dyn_v + 2
	add	x0, x0, :lo12:dyn_v + 2

	adrp	x0, dyn_v + 8
	ldr	x0, [x0, :lo12:dyn_v + 8]

	adrp	x0, dyn_v + 4095
	add	x0, x0, :lo12:dyn_v + 4095

	adrp	x0, dyn_v + 4088
	ldr	x0, [x0, :lo12:dyn_v + 4088]

	adrp	x0, dyn_v + 4096
	add	x0, x0, :lo12:dyn_v + 4096

	adrp	x0, dyn_v + 4096
	ldr	x0, [x0, :lo12:dyn_v + 4096]

	adrp	x0, dyn_v + 900012
	add	x0, x0, :lo12:dyn_v + 900012

	adrp	x0, dyn_v + 900016
	ldr	x0, [x0, :lo12:dyn_v + 900016]

	adrp	x0, dyn_v + 1048575
	add	x0, x0, :lo12:dyn_v + 1048575

	adrp	x0, dyn_v + 1048568
	ldr	x0, [x0, :lo12:dyn_v + 1048568]

	adrp	x0, dyn_v - 2
	add	x0, x0, :lo12:dyn_v - 2

	adrp	x0, dyn_v - 8
	ldr	x0, [x0, :lo12:dyn_v - 8]

	adrp	x0, dyn_v - 4095
	add	x0, x0, :lo12:dyn_v - 4095

	adrp	x0, dyn_v - 4088
	ldr	x0, [x0, :lo12:dyn_v - 4088]

	adrp	x0, dyn_v - 4096
	add	x0, x0, :lo12:dyn_v - 4096

	adrp	x0, dyn_v - 4096
	ldr	x0, [x0, :lo12:dyn_v - 4096]

	adrp	x0, dyn_v - 900012
	add	x0, x0, :lo12:dyn_v - 900012

	adrp	x0, dyn_v - 900016
	ldr	x0, [x0, :lo12:dyn_v - 900016]

	adrp	x0, dyn_v - 1048576
	add	x0, x0, :lo12:dyn_v - 1048576

	adrp	x0, dyn_v - 1048576
	ldr	x0, [x0, :lo12:dyn_v - 1048576]

	adrp	x1, dyn_v + 2
	add	x1, x1, :lo12:dyn_v + 2

	adrp	x1, dyn_v + 8
	ldr	x1, [x1, :lo12:dyn_v + 8]

	adrp	x1, dyn_v + 2
	add	x0, x1, :lo12:dyn_v + 2

	adrp	x1, dyn_v + 8
	ldr	x0, [x1, :lo12:dyn_v + 8]

	adrp	x1, dyn_v - 2
	add	x1, x1, :lo12:dyn_v - 2

	adrp	x1, dyn_v - 8
	ldr	x1, [x1, :lo12:dyn_v - 8]

	b init_dyn_v
	bl init_dyn_v
