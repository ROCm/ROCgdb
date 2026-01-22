#name: .note.GNU-stack using SHT_NOTE
#source: empty.s
#source: property-or-1.s
#as: --noexecstack --generate-missing-build-notes=no
#ld: -shared --script note2.t
#readelf: --notes
#target: [check_shared_lib_support]
# Assembly source file for the HPPA assembler is renamed and modifed by
# sed.  mn10300 has relocations in .note.gnu.property section which
# elf_parse_notes doesn't support.
#notarget: am33_2.0-*-* hppa*-*-hpux* mn10300-*-*

#...
Displaying notes found in: .note
[ 	]+Owner[ 	]+Data size[ 	]+Description
  GNU                  0x[0-9a-f]+	NT_GNU_PROPERTY_TYPE_0
      Properties: UINT32_OR .*
#pass
