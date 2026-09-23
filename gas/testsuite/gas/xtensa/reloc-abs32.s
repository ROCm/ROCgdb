	.text
	.global gsym
	.skip 2
lsym:
	.skip 2
gsym:
	.skip 2

	.section .rodata
	.word lsym
	.word lsym + 16
	.word gsym
	.word gsym + 8
