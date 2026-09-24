/* Shards for the cooked index

   Copyright (C) 2022-2026 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#ifndef GDB_DWARF2_COOKED_INDEX_SHARD_H
#define GDB_DWARF2_COOKED_INDEX_SHARD_H

#include "dwarf2/cooked-index-entry.h"
#include "dwarf2/types.h"
#include "gdbsupport/gdb_obstack.h"
#include "addrmap.h"
#include "gdbsupport/iterator-range.h"
#include "gdbsupport/string-set.h"
#include "complaints.h"

/* An index of interesting DIEs.  This is "cooked", in contrast to a
   mapped .debug_names or .gdb_index, which are "raw".  An entry in
   the index is of type cooked_index_entry.

   Operations on the index are described below.  They are chosen to
   make it relatively simple to implement the symtab "quick"
   methods.  */
class cooked_index_shard
{
public:
  cooked_index_shard () = default;
  DISABLE_COPY_AND_ASSIGN (cooked_index_shard);

  /* Create a new cooked_index_entry and register it with this object.
     Entries are owned by this object.  The new item is returned.  */
  cooked_index_entry *add (sect_offset die_offset, enum dwarf_tag tag,
			   cooked_index_flag flags, enum language lang,
			   cooked_index_entry_name_ref name,
			   cooked_index_entry_ref parent_entry,
			   dwarf2_per_cu *per_cu);

  /* Add a copy of NAME to the index.  Return a pointer to the
     copy.  */
  const char *add (std::string_view name)
  {
    return m_names.insert (name);
  }

  /* Install a new fixed addrmap from the given mutable addrmap.  */
  void install_addrmap (addrmap_mutable *map)
  {
    gdb_assert (m_addrmap == nullptr);
    m_addrmap = new (&m_storage) addrmap_fixed (&m_storage, map);
  }

  friend class cooked_index;

  /* A simple range over part of m_entries.  */
  using range
    = iterator_range<std::vector<cooked_index_entry *>::const_iterator>;

  /* Return a range of all the entries.  */
  range all_entries () const
  {
    return { m_entries.cbegin (), m_entries.cend () };
  }

  /* Look up an entry by name.  Returns a range of all matching
     results.  If COMPLETING is true, then a larger range, suitable
     for completion, will be returned.  */
  range find (const std::string &name, bool completing) const;

private:

  /* Return the entry that is believed to represent the program's
     "main".  This will return NULL if no such entry is available.  */
  const cooked_index_entry *get_main () const
  {
    return m_main;
  }

  /* Look up ADDR in the address map, and return either the
     corresponding CU, or nullptr if the address could not be
     found.  */
  dwarf2_per_cu *lookup (unrelocated_addr addr)
  {
    if (m_addrmap == nullptr)
      return nullptr;

    return (static_cast<dwarf2_per_cu *> (m_addrmap->find ((CORE_ADDR) addr)));
  }

  /* Create a new cooked_index_entry and register it with this object.
     Entries are owned by this object.  The new item is returned.  */
  cooked_index_entry *create (sect_offset die_offset,
			      enum dwarf_tag tag,
			      cooked_index_flag flags,
			      enum language lang,
			      cooked_index_entry_name_ref name,
			      cooked_index_entry_ref parent_entry,
			      dwarf2_per_cu *per_cu);

  /* When GNAT emits mangled ("encoded") names in the DWARF, and does
     not emit the module structure, we still need this structuring to
     do lookups.  This function recreates that information for an
     existing entry, modifying ENTRY as appropriate.  Any new entries
     are added to NEW_ENTRIES.  */
  void handle_gnat_encoded_entry
       (cooked_index_entry *entry, htab_t gnat_entries,
	std::vector<cooked_index_entry *> &new_entries);

  /* Use SIG_NAMES to resolve the deferred names of entries in this shard.

     Return true if any deferred name could not be resolved.  */
  bool resolve_deferred_names (const signature_to_name_map &sig_names);

  /* Use PARENT_MAPS to resolve the deferred parent links of entries in this
     shard.  */
  void resolve_deferred_parents (const parent_map_map *parent_maps);

  /* Remove index entries that have no name (for which we failed to
     resolve the name in resolve_deferred_names).  Break any parent link
     pointing to an entry with no name.  */
  void prune_nameless_entries ();

  /* Compute the canonical name for the entries in this shard.

     Due to how Ada name lookups work, this function may also create new index
     entries with full names.  */
  void canonicalize_names ();

  /* Called after each step of the finalization process.  Store
     COMPLAINTS so they can be reported later on the main thread.  */
  void merge_finalize_complaints (complaint_collection &&complaints)
  {
    if (m_finalize_complaints.empty ())
      m_finalize_complaints = std::move (complaints);
    else
      {
	/* The current version of gdb::unordered_set doesn't support
	   the merge method that std::unordered_set supports.  If we
	   update gdb::unordered_set then we could switch this to use
	   merge().  */
	m_finalize_complaints.insert (complaints.begin (), complaints.end ());
      }
  }

  /* Return the set of complaints emitted during the finalization
     process.  We move these complaints out of the shard as these are
     only emitted once, and don't need to be stored beyond that.  */
  complaint_collection release_finalize_complaints ()
  {
    return std::move (m_finalize_complaints);
  }

  /* Storage for the entries.  */
  auto_obstack m_storage;

  /* List of all entries.  */
  std::vector<cooked_index_entry *> m_entries;

  /* If we found an entry with 'is_main' set, store it here.  */
  cooked_index_entry *m_main = nullptr;

  /* The addrmap.  This maps address ranges to dwarf2_per_cu objects.  */
  addrmap_fixed *m_addrmap = nullptr;

  /* Storage for canonical names.  */
  gdb::string_set m_names;

  /* True if at least one entry in this shard has a name that requires
     deferred resolution.  */
  bool m_have_deferred_names = false;

  /* True if at least one entry in this shard has a parent link that requires
     deferred resolution.  */
  bool m_have_deferred_parents = false;

  /* Any complaints emitted while finalizing the index are stored
     here until they can be emitted on the main thread.  */
  complaint_collection m_finalize_complaints;
};

using cooked_index_shard_up = std::unique_ptr<cooked_index_shard>;

#endif /* GDB_DWARF2_COOKED_INDEX_SHARD_H */
