ENTRY(_start)
SECTIONS
{
	.riscv.jvt 0x1000 : AT(0x8000) {
		*(.riscv.jvt)
	}
	.text 0x3000 : AT(0x9000) {
		*(.text)
	}
}
