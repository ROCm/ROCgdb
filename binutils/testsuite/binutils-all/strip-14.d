#PROG: strip
#strip: -g
#error: \A[^\n]*: relocation 0 has invalid symbol index 1048560\Z
#notarget: rx-*
# The RX target text section is called "P" rather than ".text", so
# creating .rela.text by hand is wrong.
