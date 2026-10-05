#name: -z zicfiss=always with one input missing CFI_SS
#source: property1.s
#source: property2.s
#as: -march=rv64g -mlittle-endian
#ld: -shared -melf64lriscv -z zicfiss=always -z zicfiss-report=warning
#warning: .*property2.o: warning: -z zicfiss-report: file does not have GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS property$
#readelf: -n

Displaying notes found in: .note.gnu.property
[ 	]+Owner[ 	]+Data size[ 	]+Description
[ 	]+GNU[ 	]+0x00000010[ 	]+NT_GNU_PROPERTY_TYPE_0
[ 	]+Properties: RISC-V AND feature: CFI_SS
