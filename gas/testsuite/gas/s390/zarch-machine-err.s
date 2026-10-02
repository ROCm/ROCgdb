	.text
	.machine push
	.machine "z196"
	ppa	%r1,%r2,3
	.machine "zEC12+nohtm"
	ppa	%r1,%r2,3
	.machine "z14+nohtm"
	ppa	%r1,%r2,3
	.machine pop
