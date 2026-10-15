#name: -z zicfiss-report=error
#source: property1.s
#source: property3.s
#as: -march=rv64g -mlittle-endian
#ld: -shared -melf64lriscv -z zicfiss-report=error
#error: .*property3.o: error: -z zicfiss-report: file does not have GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS property$
