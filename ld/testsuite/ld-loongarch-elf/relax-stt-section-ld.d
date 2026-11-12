#name: LoongArch relax STT_SECTION addend executable
#source: relax-stt-section-a.s
#source: relax-stt-section-b.s
#ld: -r
#ld_after_inputfiles: && cp tmpdir/dump tmpdir/relax-stt-section-ld-r.o && $LD -o tmpdir/dump tmpdir/relax-stt-section-ld-r.o -e 0 -T $srcdir/$subdir/relax-stt-section.ld
#objdump: -d -M no-aliases

#...
.*[ \t]+bl[ \t]+.*# 12000000 <a-0x8>
#...
.*[ \t]+bl[ \t]+.*# 12010000 <b\+0xffe8>
#...
.*[ \t]+bl[ \t]+.*# 12000008 <a>
#...
.*[ \t]+bl[ \t]+.*# 12000010 <a\+0x8>
#...
.*[ \t]+bl[ \t]+.*# 12010004 <b\+0xffec>
#...
.*[ \t]+bl[ \t]+.*# 12000018 <b>
#pass
