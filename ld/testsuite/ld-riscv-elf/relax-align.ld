ENTRY(_start)
SECTIONS {
	.text 0x0001000 : {
		*(.text*)
	}
}
