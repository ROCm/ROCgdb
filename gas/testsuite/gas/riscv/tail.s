foo:
.option push
.option arch, rv32i
	tail foo
	tail foo, t2
	tail foo, a0
.option pop
.option push
.option arch, rv32i_zicfilp
	tail foo
	tail foo, t1
.option pop
