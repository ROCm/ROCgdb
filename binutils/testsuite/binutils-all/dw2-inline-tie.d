#source: dw2-inline-tie.S
#addr2line: -f 0x0 0x4 -e
#name: addr2line, inlined subroutine covering its caller exactly

# Both the DW_TAG_subprogram and the DW_TAG_inlined_subroutine inside it
# cover the same address range, so the function lookup has to break a tie.
# The inlined routine must win, at both addresses, on every run.

inlined_fn
.*dw2-inline-tie\.c:17
inlined_fn
.*dw2-inline-tie\.c:17
