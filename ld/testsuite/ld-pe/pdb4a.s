/* Two modules with 32768 source file references each, so that the total
   number of references in the DBI source-info substream exceeds 65535.  */

	.equ CV_SIGNATURE_C13, 4
	.equ DEBUG_S_STRINGTABLE, 0xf3
	.equ DEBUG_S_FILECHKSMS, 0xf4
	.equ CHKSUM_TYPE_NONE, 0

	.equ NUM_CHKSMS, 32768

	.section ".debug$S", "rn"
	.long CV_SIGNATURE_C13
	.long DEBUG_S_STRINGTABLE
	.long .strings_end - .strings_start

.strings_start:

	.asciz ""

.src1:
	.asciz "foo"

.strings_end:

	.balign 4

	.long DEBUG_S_FILECHKSMS
	.long .chksms_end - .chksms_start

.chksms_start:

	.rept NUM_CHKSMS
	.long .src1 - .strings_start
	.byte 0 /* checksum length */
	.byte CHKSUM_TYPE_NONE
	.short 0 /* padding */
	.endr

.chksms_end:

	.text

	.global main
main:
	.long 0x12345678
