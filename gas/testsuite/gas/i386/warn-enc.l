.*: Assembler messages:
.*:7: Warning: vfmadd132ps: will use AVX512VL encoding; use \{evex\} to silence
.*:8: Warning: vfnmadd132pd: will use AVX512VL encoding; use \{evex\} to silence
.*:9: Warning: vfmsub213ps: will use AVX512VL encoding; use \{evex\} to silence
.*:10: Warning: vfnmsub213pd: will use AVX512VL encoding; use \{evex\} to silence
.*:11: Warning: vfmaddsub231ps: will use AVX512VL encoding; use \{evex\} to silence
.*:12: Warning: vfmsubadd231pd: will use AVX512VL encoding; use \{evex\} to silence
.*:15: Warning: vcvtph2ps: will use AVX512VL encoding; use \{evex\} to silence
.*:16: Warning: vcvtph2ps: will use AVX512VL encoding; use \{evex\} to silence
.*:17: Warning: vcvtps2ph: will use AVX512VL encoding; use \{evex\} to silence
.*:18: Warning: vcvtps2ph: will use AVX512VL encoding; use \{evex\} to silence
.*:21: Warning: vpdpbusd: will use AVX-VNNI encoding; use \{vex\} to silence
.*:22: Warning: vpdpwssd: will use AVX-VNNI encoding; use \{vex\} to silence
.*:25: Warning: vpdpbusd: will use AVX-VNNI encoding; use \{vex\} to silence
.*:26: Warning: vpdpwssd: will use AVX-VNNI encoding; use \{vex\} to silence
.*:29: Warning: vpdpbssd: will use AVX10 encoding; use \{evex\} to silence
.*:30: Warning: vpdpbsuds: will use AVX10 encoding; use \{evex\} to silence
.*:31: Warning: vpdpbuud: will use AVX10 encoding; use \{evex\} to silence
.*:34: Warning: vpdpwsud: will use AVX10 encoding; use \{evex\} to silence
.*:35: Warning: vpdpwusds: will use AVX10 encoding; use \{evex\} to silence
.*:36: Warning: vpdpwuud: will use AVX10 encoding; use \{evex\} to silence
.*:7: Warning: vfmadd132ps: will use AVX512VL encoding; use \{evex\} to silence
.*:8: Warning: vfnmadd132pd: will use AVX512VL encoding; use \{evex\} to silence
.*:9: Warning: vfmsub213ps: will use AVX512VL encoding; use \{evex\} to silence
.*:10: Warning: vfnmsub213pd: will use AVX512VL encoding; use \{evex\} to silence
.*:11: Warning: vfmaddsub231ps: will use AVX512VL encoding; use \{evex\} to silence
.*:12: Warning: vfmsubadd231pd: will use AVX512VL encoding; use \{evex\} to silence
.*:15: Warning: vcvtph2ps: will use AVX512VL encoding; use \{evex\} to silence
.*:16: Warning: vcvtph2ps: will use AVX512VL encoding; use \{evex\} to silence
.*:17: Warning: vcvtps2ph: will use AVX512VL encoding; use \{evex\} to silence
.*:18: Warning: vcvtps2ph: will use AVX512VL encoding; use \{evex\} to silence
.*:29: Error: no VEX/XOP encoding for `vpdpbssd'
.*:30: Error: no VEX/XOP encoding for `vpdpbsuds'
.*:31: Error: no VEX/XOP encoding for `vpdpbuud'
.*:34: Error: no VEX/XOP encoding for `vpdpwsud'
.*:35: Error: no VEX/XOP encoding for `vpdpwusds'
.*:36: Error: no VEX/XOP encoding for `vpdpwuud'
.*:21: Error: .* for `vpdpbusd'
.*:22: Error: .* for `vpdpwssd'
.*:25: Error: .* for `vpdpbusd'
.*:26: Error: .* for `vpdpwssd'
[ 	]*[0-9]+[ 	]+\.text
[ 	]*[0-9]+[ 	]*
[ 	]*[0-9]+[ 	]+_start:
[ 	]*[0-9]+[ 	]+\.irp pfx, "", \{vex\}, \{evex\}
[ 	]*[0-9]+[ 	]*
[ 	]*[0-9]+[ 	]+\.arch default; \.arch \.nofma
[ 	]*[0-9]+[ 	]+\\pfx	vfmadd132ps	%xmm1, %xmm2, %xmm3
[ 	]*[0-9]+[ 	]+\\pfx	vfnmadd132pd	%ymm1, %ymm2, %ymm3
[ 	]*[0-9]+[ 	]+\\pfx	vfmsub213ps	%xmm1, %xmm2, %xmm3
[ 	]*[0-9]+[ 	]+\\pfx	vfnmsub213pd	%ymm1, %ymm2, %ymm3
[ 	]*[0-9]+[ 	]+\\pfx	vfmaddsub231ps	%xmm1, %xmm2, %xmm3
[ 	]*[0-9]+[ 	]+\\pfx	vfmsubadd231pd	%ymm1, %ymm2, %ymm3
[ 	]*[0-9]+[ 	]*
[ 	]*[0-9]+[ 	]+\.arch default; \.arch \.nof16c
[ 	]*[0-9]+[ 	]+\\pfx	vcvtph2ps	%xmm1, %xmm2
[ 	]*[0-9]+[ 	]+\\pfx	vcvtph2ps	%xmm1, %ymm2
[ 	]*[0-9]+[ 	]+\\pfx	vcvtps2ph	\$0, %xmm1, %xmm2
[ 	]*[0-9]+[ 	]+\\pfx	vcvtps2ph	\$0, %ymm1, %xmm2
[ 	]*[0-9]+[ 	]*
[ 	]*[0-9]+[ 	]+\.arch default; \.arch \.noavx512_vnni
[ 	]*[0-9]+[ 	]+\\pfx	vpdpbusd	%xmm1, %xmm2, %xmm3
[ 	]*[0-9]+[ 	]+\\pfx	vpdpwssd	%ymm1, %ymm2, %ymm3
[ 	]*[0-9]+[ 	]*
[ 	]*[0-9]+[ 	]+\.arch default; \.arch \.noavx512vl
[ 	]*[0-9]+[ 	]+\\pfx	vpdpbusd	%xmm1, %xmm2, %xmm3
[ 	]*[0-9]+[ 	]+\\pfx	vpdpwssd	%ymm1, %ymm2, %ymm3
[ 	]*[0-9]+[ 	]*
[ 	]*[0-9]+[ 	]+\.arch default; \.arch \.noavx_vnni_int8
[ 	]*[0-9]+[ 	]+\\pfx	vpdpbssd	%xmm1, %xmm2, %xmm3
[ 	]*[0-9]+[ 	]+\\pfx	vpdpbsuds	%xmm1, %xmm2, %xmm3
[ 	]*[0-9]+[ 	]+\\pfx	vpdpbuud	%ymm1, %ymm2, %ymm3
[ 	]*[0-9]+[ 	]*
[ 	]*[0-9]+[ 	]+\.arch default; \.arch \.noavx_vnni_int16
[ 	]*[0-9]+[ 	]+\\pfx	vpdpwsud	%xmm1, %xmm2, %xmm3
[ 	]*[0-9]+[ 	]+\\pfx	vpdpwusds	%ymm1, %ymm2, %ymm3
[ 	]*[0-9]+[ 	]+\\pfx	vpdpwuud	%ymm1, %ymm2, %ymm3
[ 	]*[0-9]+[ 	]*
[ 	]*[0-9]+[ 	]+\.endr
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.nofma
[ 	]*[0-9]+ .... 62F26D08 	>  vfmadd132ps %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+98D9
[ 	]*[0-9]+ .... 62F2ED28 	>  vfnmadd132pd %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+9CD9
[ 	]*[0-9]+ .... 62F26D08 	>  vfmsub213ps %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+AAD9
[ 	]*[0-9]+ .... 62F2ED28 	>  vfnmsub213pd %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+AED9
[ 	]*[0-9]+ .... 62F26D08 	>  vfmaddsub231ps %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+B6D9
[ 	]*[0-9]+ .... 62F2ED28 	>  vfmsubadd231pd %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+B7D9
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.nof16c
[ 	]*[0-9]+ .... 62F27D08 	>  vcvtph2ps %xmm1,%xmm2
.*Warning: .*
[ 	]*[0-9]+[ 	]+13D1
[ 	]*[0-9]+ .... 62F27D28 	>  vcvtph2ps %xmm1,%ymm2
.*Warning: .*
[ 	]*[0-9]+[ 	]+13D1
[ 	]*[0-9]+ .... 62F37D08 	>  vcvtps2ph \$0,%xmm1,%xmm2
.*Warning: .*
[ 	]*[0-9]+[ 	]+1DCA00
[ 	]*[0-9]+ .... 62F37D28 	>  vcvtps2ph \$0,%ymm1,%xmm2
.*Warning: .*
[ 	]*[0-9]+[ 	]+1DCA00
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx512_vnni
[ 	]*[0-9]+ .... C4E26950 	>  vpdpbusd %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+D9
[ 	]*[0-9]+ .... C4E26D52 	>  vpdpwssd %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+D9
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx512vl
[ 	]*[0-9]+ .... C4E26950 	>  vpdpbusd %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+D9
[ 	]*[0-9]+ .... C4E26D52 	>  vpdpwssd %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+D9
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx_vnni_int8
[ 	]*[0-9]+ .... 62F26F08 	>  vpdpbssd %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+50D9
[ 	]*[0-9]+ .... 62F26E08 	>  vpdpbsuds %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+51D9
[ 	]*[0-9]+ .... 62F26C28 	>  vpdpbuud %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+50D9
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx_vnni_int16
[ 	]*[0-9]+ .... 62F26E08 	>  vpdpwsud %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+D2D9
[ 	]*[0-9]+ .... 62F26D28 	>  vpdpwusds %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+D3D9
[ 	]*[0-9]+ .... 62F26C28 	>  vpdpwuud %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+D2D9
#...
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.nofma
[ 	]*[0-9]+ .... 62F26D08 	>  \{vex\} vfmadd132ps %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+98D9
[ 	]*[0-9]+ .... 62F2ED28 	>  \{vex\} vfnmadd132pd %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+9CD9
[ 	]*[0-9]+ .... 62F26D08 	>  \{vex\} vfmsub213ps %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+AAD9
[ 	]*[0-9]+ .... 62F2ED28 	>  \{vex\} vfnmsub213pd %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+AED9
[ 	]*[0-9]+ .... 62F26D08 	>  \{vex\} vfmaddsub231ps %xmm1,%xmm2,%xmm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+B6D9
[ 	]*[0-9]+ .... 62F2ED28 	>  \{vex\} vfmsubadd231pd %ymm1,%ymm2,%ymm3
.*Warning: .*
[ 	]*[0-9]+[ 	]+B7D9
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.nof16c
[ 	]*[0-9]+ .... 62F27D08 	>  \{vex\} vcvtph2ps %xmm1,%xmm2
.*Warning: .*
[ 	]*[0-9]+[ 	]+13D1
[ 	]*[0-9]+ .... 62F27D28 	>  \{vex\} vcvtph2ps %xmm1,%ymm2
.*Warning: .*
[ 	]*[0-9]+[ 	]+13D1
[ 	]*[0-9]+ .... 62F37D08 	>  \{vex\} vcvtps2ph \$0,%xmm1,%xmm2
.*Warning: .*
[ 	]*[0-9]+[ 	]+1DCA00
[ 	]*[0-9]+ .... 62F37D28 	>  \{vex\} vcvtps2ph \$0,%ymm1,%xmm2
.*Warning: .*
[ 	]*[0-9]+[ 	]+1DCA00
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx512_vnni
[ 	]*[0-9]+ .... C4E26950 	>  \{vex\} vpdpbusd %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+D9
[ 	]*[0-9]+ .... C4E26D52 	>  \{vex\} vpdpwssd %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+D9
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx512vl
[ 	]*[0-9]+ .... C4E26950 	>  \{vex\} vpdpbusd %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+D9
[ 	]*[0-9]+ .... C4E26D52 	>  \{vex\} vpdpwssd %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+D9
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx_vnni_int8
[ 	]*[0-9]+[ 	]+>  \{vex\} vpdpbssd %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+>  \{vex\} vpdpbsuds %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+>  \{vex\} vpdpbuud %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx_vnni_int16
[ 	]*[0-9]+[ 	]+>  \{vex\} vpdpwsud %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+>  \{vex\} vpdpwusds %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+>  \{vex\} vpdpwuud %ymm1,%ymm2,%ymm3
#...
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.nofma
[ 	]*[0-9]+ .... 62F26D08 	>  \{evex\} vfmadd132ps %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+98D9
[ 	]*[0-9]+ .... 62F2ED28 	>  \{evex\} vfnmadd132pd %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+9CD9
[ 	]*[0-9]+ .... 62F26D08 	>  \{evex\} vfmsub213ps %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+AAD9
[ 	]*[0-9]+ .... 62F2ED28 	>  \{evex\} vfnmsub213pd %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+AED9
[ 	]*[0-9]+ .... 62F26D08 	>  \{evex\} vfmaddsub231ps %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+B6D9
[ 	]*[0-9]+ .... 62F2ED28 	>  \{evex\} vfmsubadd231pd %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+B7D9
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.nof16c
[ 	]*[0-9]+ .... 62F27D08 	>  \{evex\} vcvtph2ps %xmm1,%xmm2
[ 	]*[0-9]+[ 	]+13D1
[ 	]*[0-9]+ .... 62F27D28 	>  \{evex\} vcvtph2ps %xmm1,%ymm2
[ 	]*[0-9]+[ 	]+13D1
[ 	]*[0-9]+ .... 62F37D08 	>  \{evex\} vcvtps2ph \$0,%xmm1,%xmm2
[ 	]*[0-9]+[ 	]+1DCA00
[ 	]*[0-9]+ .... 62F37D28 	>  \{evex\} vcvtps2ph \$0,%ymm1,%xmm2
[ 	]*[0-9]+[ 	]+1DCA00
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx512_vnni
[ 	]*[0-9]+[ 	]+>  \{evex\} vpdpbusd %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+>  \{evex\} vpdpwssd %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx512vl
[ 	]*[0-9]+[ 	]+>  \{evex\} vpdpbusd %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+>  \{evex\} vpdpwssd %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx_vnni_int8
[ 	]*[0-9]+ .... 62F26F08 	>  \{evex\} vpdpbssd %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+50D9
[ 	]*[0-9]+ .... 62F26E08 	>  \{evex\} vpdpbsuds %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+51D9
[ 	]*[0-9]+ .... 62F26C28 	>  \{evex\} vpdpbuud %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+50D9
[ 	]*[0-9]+[ 	]+> *
[ 	]*[0-9]+[ 	]+>  \.arch default
[ 	]*[0-9]+[ 	]+>  \.arch \.noavx_vnni_int16
[ 	]*[0-9]+ .... 62F26E08 	>  \{evex\} vpdpwsud %xmm1,%xmm2,%xmm3
[ 	]*[0-9]+[ 	]+D2D9
[ 	]*[0-9]+ .... 62F26D28 	>  \{evex\} vpdpwusds %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+D3D9
[ 	]*[0-9]+ .... 62F26C28 	>  \{evex\} vpdpwuud %ymm1,%ymm2,%ymm3
[ 	]*[0-9]+[ 	]+D2D9
#pass
