.text
foo:
	ag	%r9,4095(%r5,%r10)
.machine push
.machine z10
	asi	5555(%r6),-42
.machine pop
	agf	%r9,4095(%r5,%r10)
	.machine push
	.machine "z900+htm"
	ppa	%r1,%r2,3
	.machine "zEC12"
	ppa	%r1,%r2,3
	.machine "z15+nohtm"
	ppa	%r1,%r2,3
	.machine pop
