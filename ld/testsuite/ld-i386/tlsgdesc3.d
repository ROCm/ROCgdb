#source: tlsgdesc2.s
#name: TLS GDesc call (indirect CALL)
#as: --32
#ld: -shared -melf_i386
#error: .*: relocation R_386_TLS_DESC_CALL against `foo' must be used in indirect CALL with EAX register only
