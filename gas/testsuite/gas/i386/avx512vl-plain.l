.*: Assembler messages:
.*:5: Error: .*
.*:8: Error: .*
.*:9: Error: .*
.*:22: Error: .*
GAS LISTING .*


[ 	]*1[ 	]+\.text
[ 	]*2[ 	]+\.arch generic32
[ 	]*3[ 	]+\.arch \.avx512vl
[ 	]*4[ 	]+_start:
[ 	]*5[ 	]+\{evex\} vaesenc[ 	]+%ymm1, %ymm2, %ymm3
[ 	]*6[ 	]+\?\?\?\? 62F27D08[ 	]+vcvtph2ps[ 	]+%xmm1, %xmm2
[ 	]*6[ 	]+13D1
[ 	]*7[ 	]+\?\?\?\? 62F26D08[ 	]+vfmadd132ps[ 	]+%xmm1, %xmm2, %xmm3
[ 	]*7[ 	]+98D9
[ 	]*8[ 	]+vgf2p8mulb[ 	]+%ymm1, %ymm2, %ymm3\{%k4\}
[ 	]*9[ 	]+\{evex\} vpclmulqdq[ 	]+\$0, %ymm1, %ymm2, %ymm3
[ 	]*10[ 	]*
[ 	]*11[ 	]+\.arch \.vaes
[ 	]*12[ 	]+\?\?\?\? 62F26D28[ 	]+\{evex\} vaesenc[ 	]+%ymm1, %ymm2, %ymm3
[ 	]*12[ 	]+DCD9
[ 	]*13[ 	]*
[ 	]*14[ 	]+\.arch \.gfni
[ 	]*15[ 	]+\?\?\?\? 62F26D2C[ 	]+vgf2p8mulb[ 	]+%ymm1, %ymm2, %ymm3\{%k4\}
[ 	]*15[ 	]+CFD9
[ 	]*16[ 	]*
[ 	]*17[ 	]+\.arch \.vpclmulqdq
[ 	]*18[ 	]+\?\?\?\? 62F36D28[ 	]+\{evex\} vpclmulqdq[ 	]+\$0, %ymm1, %ymm2, %ymm3
[ 	]*18[ 	]+44D900
[ 	]*19[ 	]*
[ 	]*20[ 	]+\.arch generic32
[ 	]*21[ 	]+\.arch \.avx512f
[ 	]*22[ 	]+\{evex\}[ 	]+vpermd[ 	]+%ymm1, %ymm2, %ymm3
[ 	]*23[ 	]+\?\?\?\? C4E26D36[ 	]+vpermd[ 	]+%ymm1, %ymm2, %ymm3
#pass
