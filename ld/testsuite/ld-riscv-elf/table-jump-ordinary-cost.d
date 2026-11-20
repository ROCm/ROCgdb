#source: table-jump-ordinary-cost.s
#as: -march=rv64ic_zcmt -mabi=lp64 -mlittle-endian
#ld: -melf64lriscv --relax-tbljal
#objdump: -d -M no-aliases

# With ordinary encodings these groups save 0 and 6 bytes.  Neither
# pays for an 8-byte entry even when the cross-input calls were
# still eight bytes long at recording time.
#...
.* <_start>:
.*[ 	]c\.j[ 	]+[0-9a-f]+ <short_target>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <short_target>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <short_target>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <short_target>
.*[ 	]c\.j[ 	]+[0-9a-f]+ <short_target>
.*[ 	]jal[ 	]+zero,[0-9a-f]+ <medium_target>
.*[ 	]jal[ 	]+zero,[0-9a-f]+ <medium_target>
.*[ 	]jal[ 	]+zero,[0-9a-f]+ <medium_target>
#pass
