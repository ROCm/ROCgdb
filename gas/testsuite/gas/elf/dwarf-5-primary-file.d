#source: dwarf-primary-file.s
#as: --gdwarf-5
#readelf: -W -wl -wi
#name: DWARF5 primary source file

#...
[ \t]+0[ \t]+.*primary\.S
[ \t]+1[ \t]+.*primary\.S
[ \t]+2[ \t]+.*included\.h
#...
.*DW_AT_name.*: primary\.S
#pass
