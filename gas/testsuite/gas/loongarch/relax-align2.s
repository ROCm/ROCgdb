# Make align symbol .Lla-relax-align in same section with .align

.section ".text", "ax"
call .text
nop
.align 4
nop
.align 4, , 4

.section ".text2", "ax"
call .text
nop
.align 4
nop
.align 4, , 4
