/* Companion stub for the YAML-authored PDB coverage fixtures.

   The YAML-generated PDBs are attached to a real PE via the CodeView RSDS
   record; GDB validates the PDB GUID/age against this executable before
   using it (gdb/pdb/pdb.c pdb_validate_guid).  This stub is only that PE:
   its debug info is discarded and replaced by the hand-authored PDB, so all
   it needs is a .text section for the fixtures' symbol offsets to resolve
   against (gdb/pdb/pdb.c map_section_offset_to_pc uses the PE's sections).  */

int
main (void)
{
  return 0;
}
