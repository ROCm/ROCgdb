	.section .bss
local:
	.zero 1
.Ltemp:
	.zero 1

	.data
	.long local + 16
	.long .Ltemp + 16
