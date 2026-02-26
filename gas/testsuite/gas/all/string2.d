#as: -f
#objdump : -s -j .data -j "\$DATA\$"
#name : .ascii w/ angle brackets
# These have their own "stringer", not recognizing '<' / '>' as brackets.
#xfail: tic4x-* tic54x-*

.*: .*

Contents of section (\.data|\$DATA\$):
 0000 213f5c21 213f5c21 .*
#pass
