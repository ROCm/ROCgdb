#objdump: -s -j .xdata
#name: PEP aarch64 SEH large function

.*:     file format pe-aarch64-little

Contents of section .xdata:

# .xdata SEH record
# 0x0813ffff  code-words: 1
#             epilogue-count: 0
#             single-epilogue-in-header: 0
#             exception-data: 1
#             version: 0
#             function-size: 1048572
# 0x01         .seh_stackalloc 16
# 0xe4        end
# 0xe3        nop
# 0xe3        nop
# 0x00000000  seh-handler
# 0x00000000  fragment-offset
# 0x0000003c  seh-handle-data-address
# 0x0833ffff  code-words: 1
#             epilogue-count: 0
#             single-epilogue-in-header: 1
#             exception-data: 1
#             version: 0
#             function-size: 1048572
# 0xe5        end_c
# 0x01        .seh_stackalloc 16
# 0xe4        end
# 0xe3        nop
# 0x00000000  seh-handler
# 0x000ffffc  fragment-offset
# 0x0000003c  seh-handle-data-address
# 0x083002ca  code-words: 1
#             epilogue-count: 0
#             single-epilogue-in-header: 1
#             exception-data: 1
#             version: 0
#             function-size: 2856
# 0xe5        end_c
# 0x01         .seh_stackalloc 16
# 0xe4        end
# 0xe3        nop
# 0x00000000  seh-handler
# 0x001ffff8  fragment-offset
# 0x0000003c  seh-handle-data-address
# 0x00000001  long 1 (seh_handlerdata)

 0000 ffff1308 01e4e3e3 00000000 00000000  .*
 0010 3c000000 ffff1308 e501e4e3 00000000  .*
 0020 fcff0f00 3c000000 ca021008 e501e4e3  .*
 0030 00000000 f8ff1f00 3c000000 01000000  .*