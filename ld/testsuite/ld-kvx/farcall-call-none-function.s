	.global _start
	.global bar

# We will place the section .text at 0x1000.

	.text

_start:
	call bar
	;;
	ret
	;;
	
# We will place the section .foo at 0x10001000.

	.section .foo, "xa"
bar:
	ret
	;;
