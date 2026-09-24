#...
Contents of section .got:
 20000 00003412 00000000 00000100 00000000.*
 20010 04000100 00000000 00003412 00000000.*
 20020 00000000 00000000 00000000 00000000.*
 20030 00000000 00000000.*
#...
Disassembly of section .text.got_local:

000000000001000c <.text.got_local>:
[ 	]+1000c:[ 	]+[0-9a-f]+[ 	]+auipc.*
[ 	]+10010:[ 	]+[0-9a-f]+[ 	]+ld[ 	]+.*# 20008 <.*>

Disassembly of section .text.got_hidden:

0000000000010014 <.text.got_hidden>:
[ 	]+10014:[ 	]+[0-9a-f]+[ 	]+auipc.*
[ 	]+10018:[ 	]+[0-9a-f]+[ 	]+ld[ 	]+.*# 20010 <.*>

Disassembly of section .text.got_global:

000000000001001c <.text.got_global>:
[ 	]+1001c:[ 	]+[0-9a-f]+[ 	]+auipc.*
[ 	]+10020:[ 	]+[0-9a-f]+[ 	]+ld[ 	]+.*# 20020 <.*>

Disassembly of section .text.got_global_abs:

0000000000010024 <.text.got_global_abs>:
[ 	]+10024:[ 	]+[0-9a-f]+[ 	]+auipc.*
[ 	]+10028:[ 	]+[0-9a-f]+[ 	]+ld[ 	]+.*# 20028 <.*>

Disassembly of section .text.got_weak_undef:

000000000001002c <.text.got_weak_undef>:
[ 	]+1002c:[ 	]+[0-9a-f]+[ 	]+auipc.*
[ 	]+10030:[ 	]+[0-9a-f]+[ 	]+ld[ 	]+.*# 20030 <.*>

Disassembly of section .text.got_DYNAMIC:

0000000000010034 <.text.got_DYNAMIC>:
[ 	]+10034:[ 	]+[0-9a-f]+[ 	]+auipc.*
[ 	]+10038:[ 	]+[0-9a-f]+[ 	]+ld[ 	]+.*# 20018 <.*>
#...
