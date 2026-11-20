#source: table-jump.s
#as: -march=rv64imafd_zicsr_zifencei_zca_zcd_zcmt --defsym TEST_CM_JT_JALT=1
#ld: -Ttable-jump.ld --relax-tbljal
#readelf: -SWs

#failif
#...
.*(__jvt_base\$|\.riscv\.jvt).*
#pass
