#name: -z zicfilp=unlabeled with a single input missing CFI_LP_UNLABELED
#source: property2.s
#as: -march=rv64g -mlittle-endian
#ld: -shared -melf64lriscv -z zicfilp=unlabeled -z zicfilp-unlabeled-report=warning
#warning: .*property2.o: warning: -z zicfilp-unlabeled-report: file does not have GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED property$
#readelf: -n

Displaying notes found in: .note.gnu.property
[ 	]+Owner[ 	]+Data size[ 	]+Description
[ 	]+GNU[ 	]+0x00000010[ 	]+NT_GNU_PROPERTY_TYPE_0
[ 	]+Properties: RISC-V AND feature: CFI_LP_UNLABELED
