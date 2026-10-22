#as: -march=rv32i -march-attr -misa-spec=20191213
#source: imply-zicsr-zifencei.s
#error: .*Assembler messages:
.*Error: unrecognized opcode `csrr t0,mscratch', extension `zicsr' required
.*Error: unrecognized opcode `csrwi mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrsi mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrci mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrw mscratch,t0', extension `zicsr' required
.*Error: unrecognized opcode `csrw mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrs mscratch,t0', extension `zicsr' required
.*Error: unrecognized opcode `csrs mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrc mscratch,t0', extension `zicsr' required
.*Error: unrecognized opcode `csrc mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrrwi t0,mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrrsi t0,mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrrci t0,mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrrw t0,mscratch,t0', extension `zicsr' required
.*Error: unrecognized opcode `csrrw t0,mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrrs t0,mscratch,t0', extension `zicsr' required
.*Error: unrecognized opcode `csrrs t0,mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `csrrc t0,mscratch,t0', extension `zicsr' required
.*Error: unrecognized opcode `csrrc t0,mscratch,0x0', extension `zicsr' required
.*Error: unrecognized opcode `fence.i', extension `zifencei' required
