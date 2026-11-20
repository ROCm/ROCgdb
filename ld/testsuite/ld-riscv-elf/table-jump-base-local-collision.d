#source: table-jump-base-local-collision.s
#as: -march=rv64ic_zcmt
#ld: -Ttable-jump.ld --relax-tbljal
#error: .*__jvt_base\$ is reserved for Zcmt table-jump relaxation
