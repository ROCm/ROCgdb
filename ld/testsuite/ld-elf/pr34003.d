#name: --rosegment (has at least one R, one RX and one RW segments)
#source: pr22393-1.s
#ld: -shared -z separate-code -z relro --rosegment
#readelf: -l --wide
#notarget: ![check_relro_support]
#target: i?86-*-* x86_64-*-*

#...
[ ]+LOAD[ 	]+0x[0-9a-f x]+R[ ]+0x.*
#...
[ ]+LOAD[ 	]+0x[0-9a-f x]+R E[ ]+0x.*
#...
[ ]+LOAD[ 	]+0x[0-9a-f x]+RW[ ]+0x.*
#...
