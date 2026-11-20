ENTRY(_start)
SECTIONS
{
	.text.target 0x0 : {
		*(.text.target)
	}
	.riscv.jvt 0x1000 : {
		*(.riscv.jvt)
	}
	.text 0x3000 : {
		*(.text.start)
	}
}
