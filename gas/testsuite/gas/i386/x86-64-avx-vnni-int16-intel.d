#objdump: -dw -Mintel
#name: x86_64 AVX-VNNI-INT16 insns (Intel disassembly)
#source: x86-64-avx-vnni-int16.s

.*: +file format .*

Disassembly of section \.text:

0+ <_start>:
\s*[a-f0-9]+:\s*c4 e2 56 d2 f4\s+vpdpwsud ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 52 d2 f4\s+vpdpwsud xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 56 d2 b4 f5 00 00 00 10\s+vpdpwsud ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 56 d2 31\s+vpdpwsud ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 56 d2 b1 e0 0f 00 00\s+vpdpwsud ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 56 d2 b2 00 f0 ff ff\s+vpdpwsud ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 52 d2 b4 f5 00 00 00 10\s+vpdpwsud xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 52 d2 31\s+vpdpwsud xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 52 d2 b1 f0 07 00 00\s+vpdpwsud xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 52 d2 b2 00 f8 ff ff\s+vpdpwsud xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 56 d3 f4\s+vpdpwsuds ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 52 d3 f4\s+vpdpwsuds xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 56 d3 b4 f5 00 00 00 10\s+vpdpwsuds ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 56 d3 31\s+vpdpwsuds ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 56 d3 b1 e0 0f 00 00\s+vpdpwsuds ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 56 d3 b2 00 f0 ff ff\s+vpdpwsuds ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 52 d3 b4 f5 00 00 00 10\s+vpdpwsuds xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 52 d3 31\s+vpdpwsuds xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 52 d3 b1 f0 07 00 00\s+vpdpwsuds xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 52 d3 b2 00 f8 ff ff\s+vpdpwsuds xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 55 d2 f4\s+vpdpwusd ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 51 d2 f4\s+vpdpwusd xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 55 d2 b4 f5 00 00 00 10\s+vpdpwusd ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 55 d2 31\s+vpdpwusd ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 55 d2 b1 e0 0f 00 00\s+vpdpwusd ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 55 d2 b2 00 f0 ff ff\s+vpdpwusd ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 51 d2 b4 f5 00 00 00 10\s+vpdpwusd xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 51 d2 31\s+vpdpwusd xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 51 d2 b1 f0 07 00 00\s+vpdpwusd xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 51 d2 b2 00 f8 ff ff\s+vpdpwusd xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 55 d3 f4\s+vpdpwusds ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 51 d3 f4\s+vpdpwusds xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 55 d3 b4 f5 00 00 00 10\s+vpdpwusds ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 55 d3 31\s+vpdpwusds ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 55 d3 b1 e0 0f 00 00\s+vpdpwusds ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 55 d3 b2 00 f0 ff ff\s+vpdpwusds ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 51 d3 b4 f5 00 00 00 10\s+vpdpwusds xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 51 d3 31\s+vpdpwusds xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 51 d3 b1 f0 07 00 00\s+vpdpwusds xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 51 d3 b2 00 f8 ff ff\s+vpdpwusds xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 54 d2 f4\s+vpdpwuud ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 50 d2 f4\s+vpdpwuud xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 54 d2 b4 f5 00 00 00 10\s+vpdpwuud ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 54 d2 31\s+vpdpwuud ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 54 d2 b1 e0 0f 00 00\s+vpdpwuud ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 54 d2 b2 00 f0 ff ff\s+vpdpwuud ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 50 d2 b4 f5 00 00 00 10\s+vpdpwuud xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 50 d2 31\s+vpdpwuud xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 50 d2 b1 f0 07 00 00\s+vpdpwuud xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 50 d2 b2 00 f8 ff ff\s+vpdpwuud xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 54 d3 f4\s+vpdpwuuds ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 50 d3 f4\s+vpdpwuuds xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 54 d3 b4 f5 00 00 00 10\s+vpdpwuuds ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 54 d3 31\s+vpdpwuuds ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 54 d3 b1 e0 0f 00 00\s+vpdpwuuds ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 54 d3 b2 00 f0 ff ff\s+vpdpwuuds ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 50 d3 b4 f5 00 00 00 10\s+vpdpwuuds xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 50 d3 31\s+vpdpwuuds xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 50 d3 b1 f0 07 00 00\s+vpdpwuuds xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 50 d3 b2 00 f8 ff ff\s+vpdpwuuds xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 56 d2 f4\s+vpdpwsud ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 52 d2 f4\s+vpdpwsud xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 56 d2 b4 f5 00 00 00 10\s+vpdpwsud ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 56 d2 31\s+vpdpwsud ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 56 d2 b1 e0 0f 00 00\s+vpdpwsud ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 56 d2 b2 00 f0 ff ff\s+vpdpwsud ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 52 d2 b4 f5 00 00 00 10\s+vpdpwsud xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 52 d2 31\s+vpdpwsud xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 52 d2 b1 f0 07 00 00\s+vpdpwsud xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 52 d2 b2 00 f8 ff ff\s+vpdpwsud xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 56 d3 f4\s+vpdpwsuds ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 52 d3 f4\s+vpdpwsuds xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 56 d3 b4 f5 00 00 00 10\s+vpdpwsuds ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 56 d3 31\s+vpdpwsuds ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 56 d3 b1 e0 0f 00 00\s+vpdpwsuds ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 56 d3 b2 00 f0 ff ff\s+vpdpwsuds ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 52 d3 b4 f5 00 00 00 10\s+vpdpwsuds xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 52 d3 31\s+vpdpwsuds xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 52 d3 b1 f0 07 00 00\s+vpdpwsuds xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 52 d3 b2 00 f8 ff ff\s+vpdpwsuds xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 55 d2 f4\s+vpdpwusd ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 51 d2 f4\s+vpdpwusd xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 55 d2 b4 f5 00 00 00 10\s+vpdpwusd ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 55 d2 31\s+vpdpwusd ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 55 d2 b1 e0 0f 00 00\s+vpdpwusd ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 55 d2 b2 00 f0 ff ff\s+vpdpwusd ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 51 d2 b4 f5 00 00 00 10\s+vpdpwusd xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 51 d2 31\s+vpdpwusd xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 51 d2 b1 f0 07 00 00\s+vpdpwusd xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 51 d2 b2 00 f8 ff ff\s+vpdpwusd xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 55 d3 f4\s+vpdpwusds ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 51 d3 f4\s+vpdpwusds xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 55 d3 b4 f5 00 00 00 10\s+vpdpwusds ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 55 d3 31\s+vpdpwusds ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 55 d3 b1 e0 0f 00 00\s+vpdpwusds ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 55 d3 b2 00 f0 ff ff\s+vpdpwusds ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 51 d3 b4 f5 00 00 00 10\s+vpdpwusds xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 51 d3 31\s+vpdpwusds xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 51 d3 b1 f0 07 00 00\s+vpdpwusds xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 51 d3 b2 00 f8 ff ff\s+vpdpwusds xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 54 d2 f4\s+vpdpwuud ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 50 d2 f4\s+vpdpwuud xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 54 d2 b4 f5 00 00 00 10\s+vpdpwuud ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 54 d2 31\s+vpdpwuud ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 54 d2 b1 e0 0f 00 00\s+vpdpwuud ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 54 d2 b2 00 f0 ff ff\s+vpdpwuud ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 50 d2 b4 f5 00 00 00 10\s+vpdpwuud xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 50 d2 31\s+vpdpwuud xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 50 d2 b1 f0 07 00 00\s+vpdpwuud xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 50 d2 b2 00 f8 ff ff\s+vpdpwuud xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
\s*[a-f0-9]+:\s*c4 e2 54 d3 f4\s+vpdpwuuds ymm6,ymm5,ymm4
\s*[a-f0-9]+:\s*c4 e2 50 d3 f4\s+vpdpwuuds xmm6,xmm5,xmm4
\s*[a-f0-9]+:\s*c4 a2 54 d3 b4 f5 00 00 00 10\s+vpdpwuuds ymm6,ymm5,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 54 d3 31\s+vpdpwuuds ymm6,ymm5,YMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 54 d3 b1 e0 0f 00 00\s+vpdpwuuds ymm6,ymm5,YMMWORD PTR \[rcx\+0xfe0\]
\s*[a-f0-9]+:\s*c4 e2 54 d3 b2 00 f0 ff ff\s+vpdpwuuds ymm6,ymm5,YMMWORD PTR \[rdx-0x1000\]
\s*[a-f0-9]+:\s*c4 a2 50 d3 b4 f5 00 00 00 10\s+vpdpwuuds xmm6,xmm5,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\]
\s*[a-f0-9]+:\s*c4 c2 50 d3 31\s+vpdpwuuds xmm6,xmm5,XMMWORD PTR \[r9\]
\s*[a-f0-9]+:\s*c4 e2 50 d3 b1 f0 07 00 00\s+vpdpwuuds xmm6,xmm5,XMMWORD PTR \[rcx\+0x7f0\]
\s*[a-f0-9]+:\s*c4 e2 50 d3 b2 00 f8 ff ff\s+vpdpwuuds xmm6,xmm5,XMMWORD PTR \[rdx-0x800\]
