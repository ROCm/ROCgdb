  .text
  .globl foo
  .type foo, @function
foo:
  ret

  .section .nonalloc,"",@progbits
  bl foo
