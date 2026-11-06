#source: dwarf-primary-file.s
#as: --gdwarf-4
#readelf: -W -wl -wi
#name: DWARF4 primary source file

#...
[ \t]+1[ \t]+.*primary\.S
[ \t]+2[ \t]+.*included\.h
#...
.*DW_AT_name.*: primary\.S
#pass
