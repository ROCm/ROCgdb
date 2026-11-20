ENTRY(_start)
SECTIONS
{
	.got  0x1000 : {
		*(.got)
	}
	.riscv.jvt : {
		*(.riscv.jvt)
	}
	.data 0x2000: {
		*(.data)
	}
	.text 0x3000 : {
		*(.text)
	}
}
