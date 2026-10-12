#as: -EL -mdialect=pseudoc
#objdump: -dr -M dec,pseudoc
#source: acqrel-pseudoc.s
#name: BPF load-acquire and store-release instructions, pseudoc syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	d3 21 00 00 00 01 00 00 	w1=load_acquire\(\(u8\*\)\(r2\+0\)\)
   8:	cb 21 02 00 00 01 00 00 	w1=load_acquire\(\(u16\*\)\(r2\+2\)\)
  10:	c3 21 fc ff 00 01 00 00 	w1=load_acquire\(\(u32\*\)\(r2-4\)\)
  18:	db 21 ff 7f 00 01 00 00 	r1=load_acquire\(\(u64\*\)\(r2\+32767\)\)
  20:	d3 21 00 00 10 01 00 00 	store_release\(\(u8\*\)\(r1\+0\),w2\)
  28:	cb 21 02 00 10 01 00 00 	store_release\(\(u16\*\)\(r1\+2\),w2\)
  30:	c3 21 fc ff 10 01 00 00 	store_release\(\(u32\*\)\(r1-4\),w2\)
  38:	db 21 ff 7f 10 01 00 00 	store_release\(\(u64\*\)\(r1\+32767\),r2\)
