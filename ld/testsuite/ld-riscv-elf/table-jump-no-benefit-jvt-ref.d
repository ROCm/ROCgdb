#source: table-jump-no-benefit-jvt-ref.s
#as: -march=rv64ic_zcmt
#ld: --relax-tbljal
#error: .*undefined reference to `__jvt_base\$'
