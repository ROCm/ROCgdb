# Test for conditional pseudo-jump instruction in pseudo-c syntax
    .text
    may_goto 1f
1:
    may_goto 2f
    may_goto 1b
2:
    may_goto 1b
