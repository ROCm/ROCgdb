	.intel_syntax noprefix

	.data
0:	.long 0
1:	.long 0

	.text
.Lorg:
	mov	eax, 0b - .Lorg
	mov	eax, 1b - .Lorg

	lea	edx, 0b - .Lorg[ecx]
	lea	edx, 1b - .Lorg[ecx]
	lea	edx, 0b - .Lorg[ecx][4]

	lea	edx, 0b - .Lorg[ecx*4]
	lea	edx, 0b - .Lorg[4*ecx]
	lea	edx, 0b - .Lorg[2*2*ecx]
	lea	edx, 0b - .Lorg[2*ecx*2]
	lea	edx, 0b - .Lorg[ecx*2*2]

	lea	edx, [(0b - .Lorg) + ecx]
	lea	edx, [0b - .Lorg + ecx]
	lea	edx, [ecx + (0b - .Lorg)]
	lea	edx, [ecx + 0b - .Lorg]
	lea	edx, [0b + ecx - .Lorg]

	lea	edx, [(0b - .Lorg) + ecx*4]
	lea	edx, [0b - .Lorg + ecx*4]
	lea	edx, [ecx*4 + (0b - .Lorg)]
	lea	edx, [ecx*4 + 0b - .Lorg]
	lea	edx, [0b + ecx*4 - .Lorg]

	lea	edx, 0b - .Lorg[ecx][esi]
	lea	edx, 0b - .Lorg[ecx][esi*4]
	lea	edx, 0b - .Lorg[ecx][esi*4][4]
