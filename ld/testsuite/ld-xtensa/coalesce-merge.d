#source: coalesce-merge-1.s
#source: coalesce-merge-2.s
#ld: -e probe -T coalesce-merge.t
#objdump: -d
#...
.*<probe>:
#...
.*l32r	a2, 400004 .*\(400102 .*
.*l32r	a3, 400008 .*\(40010a .*
.*l32r	a4, 40000c .*\(400100 .*
.*l32r	a5, 40000c .*\(400100 .*
.*l32r	a6, 40000c .*\(400100 .*
#...
