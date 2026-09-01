#source: textrel.s
#ld: -shared -melf64alpha -z notext
#readelf: -d

# Only DT_TEXTREL, with no DT_FLAGS carrying the SYMBOLIC and STATIC_TLS
# bits of the value of DT_TEXTREL.
#...
 +0x0+16 +\(TEXTREL\) +0x0
 +0x0+0 +\(NULL\) +0x0
