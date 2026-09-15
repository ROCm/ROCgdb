  .text
1:
  call36 fff
  .half 0x2eef - 1b
  .word 0x2eef - 1b
  .dword 0x12345678 - 1b
  .short 0x2eef - 1b
  .long 0x2eef - 1b
  .byte 0x2e - 1b
  .2byte 0x2eef - 1b
  .4byte 0x2eef - 1b
  .8byte 0x12345678 - 1b
