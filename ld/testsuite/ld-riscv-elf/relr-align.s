# Test DT_RELR with differently aligned relative relocs.

.macro data value
.ifdef __64_bit__
	.quad \value
.else
	.long \value
.endif
.endm

.text
.global _start
_start:
foo:

.data
.p2align 3
double_0:
data foo
data foo
.byte 0
double_1:
data foo
data foo
.byte 0
double_2:
data foo
data foo
.byte 0
.byte 0
.byte 0
.byte 0
.byte 0
.byte 0
single_0:
data foo
.byte 0
single_1:
data foo
.byte 0
single_2:
data foo
.byte 0
.byte 0
.byte 0
.byte 0
.byte 0
.byte 0
big:
data foo
data 1
data 2
data 3
data 4
data 5
data 6
data 7
data 8
data 9
data 10
data 11
data 12
data 13
data 14
data 15
data 16
data 17
data 18
data 19
data 20
data 21
data 22
data 23
data 24
data 25
data 26
data 27
data 28
data 29
data 30
data 31
data foo + 32
data 33
data 34
data 35
data 36
data 37
data 38
data 39
data 40
data 41
data 42
data 43
data 44
data 45
data 46
data 47
data 48
data 49
data 50
data 51
data 52
data 53
data 54
data 55
data 56
data 57
data 58
data 59
data 60
data 61
data 62
data foo + 63
data foo + 64
