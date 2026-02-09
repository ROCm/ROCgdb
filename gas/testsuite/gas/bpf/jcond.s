# Test for conditional pseudo-jump instruction
    .text
    jcond 1f
1:
    jcond 2f
    jcond 1b
2:
    jcond 1b
