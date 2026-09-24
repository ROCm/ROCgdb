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

#include "dwarf2/cooked-index-shard.h"
#include "dwarf2/tag.h"
#include "dwarf2/index-common.h"
#include "dwarf2/read.h"
#include "dwarf2/error.h"
#include "cp-support.h"
#include "c-lang.h"
#include "ada-lang.h"

/* Return true if a plain "main" could be the main program for this
   language.  Languages that are known to use some other mechanism are
   excluded here.  */

static bool
language_may_use_plain_main (enum language lang)
{
  /* No need to handle "unknown" here.  */
  return (lang == language_c
	  || lang == language_objc
	  || lang == language_cplus
	  || lang == language_m2
	  || lang == language_asm
	  || lang == language_opencl
	  || lang == language_minimal);
}

/* See cooked-index-shard.h.  */

cooked_index_entry *
cooked_index_shard::create (sect_offset die_offset,
			    enum dwarf_tag tag,
			    cooked_index_flag flags,
			    enum language lang,
			    cooked_index_entry_name_ref name,
			    cooked_index_entry_ref parent_entry,
			    dwarf2_per_cu *per_cu)
{
  if (tag == DW_TAG_module || tag == DW_TAG_namespace)
    flags &= ~IS_STATIC;
  else if (lang == language_cplus
	   && (tag == DW_TAG_class_type
	       || tag == DW_TAG_interface_type
	       || tag == DW_TAG_structure_type
	       || tag == DW_TAG_union_type
	       || tag == DW_TAG_enumeration_type
	       || tag == DW_TAG_enumerator))
    flags &= ~IS_STATIC;
  else if (tag_is_type (tag))
    flags |= IS_STATIC;

  return new (&m_storage) cooked_index_entry (die_offset, tag, flags,
					      lang, name, parent_entry,
					      per_cu);
}

/* See cooked-index-shard.h.  */

cooked_index_entry *
cooked_index_shard::add (sect_offset die_offset, enum dwarf_tag tag,
			 cooked_index_flag flags, enum language lang,
			 cooked_index_entry_name_ref name,
			 cooked_index_entry_ref parent_entry,
			 dwarf2_per_cu *per_cu)
{
  cooked_index_entry *result = create (die_offset, tag, flags, lang, name,
				       parent_entry, per_cu);
  m_entries.push_back (result);

  if (result->parent_is_deferred ())
    m_have_deferred_parents = true;

  if (result->name_is_deferred ())
    m_have_deferred_names = true;

  /* An explicitly-tagged main program should always override the
     implicit "main" discovery.  */
  if ((flags & IS_MAIN) != 0)
    m_main = result;
  /* The language check here is subtle: it exists solely to work
     around a bug in .gdb_index.  That index does not record
     languages, but it might emit an entry for "main".  However,
     recognizing this "main" as being the main program would be wrong
     -- for example, an Ada program has a C "main" but this is not the
     desired target of the "start" command.  Requiring the language to
     be set here avoids over-eagerly setting the "main" when using
     .gdb_index.  Should .gdb_index ever be removed (PR symtab/31363),
     the language_unknown check here could also be removed.

     Note that this explicit check isn't truly needed (it is covered
     by language_may_use_plain_main as well), but it's handy as a spot
     to document.  */
  else if (lang != language_unknown
	   && !result->parent_is_deferred ()
	   && result->get_parent () == nullptr
	   && m_main == nullptr
	   && language_may_use_plain_main (lang)
	   && !result->name_is_deferred ()
	   && streq (result->name (), "main"))
    m_main = result;

  return result;
}

/* See cooked-index-shard.h.  */

void
cooked_index_shard::handle_gnat_encoded_entry
     (cooked_index_entry *entry,
      htab_t gnat_entries,
      std::vector<cooked_index_entry *> &new_entries)
{
  /* We decode Ada names in a particular way: operators and wide
     characters are left as-is.  This is done to make name matching a
     bit simpler; and for wide characters, it means the choice of Ada
     source charset does not affect the indexer directly.  */
  std::string canonical = ada_decode (entry->name (), false, false);
  if (canonical.empty ())
    {
      entry->canonical = entry->name ();
      return;
    }
  std::vector<std::string_view> names = split_name (canonical.c_str (),
						    split_style::DOT_STYLE);
  std::string_view tail = names.back ();
  names.pop_back ();

  const cooked_index_entry *parent = nullptr;
  for (const auto &name : names)
    {
      uint32_t hashval = dwarf5_djb_hash (name);
      void **slot = htab_find_slot_with_hash (gnat_entries, &name,
					      hashval, INSERT);
      /* CUs are processed in order, so we only need to check the most
	 recent entry.  */
      cooked_index_entry *last = (cooked_index_entry *) *slot;
      if (last == nullptr || last->per_cu != entry->per_cu)
	{
	  const char *new_name = m_names.insert (name);
	  last = create (entry->die_offset, DW_TAG_module,
			 IS_SYNTHESIZED, language_ada, new_name, parent,
			 entry->per_cu);
	  last->canonical = last->name ();
	  new_entries.push_back (last);
	  *slot = last;
	}

      parent = last;
    }

  entry->set_parent (parent);
  entry->canonical = m_names.insert (tail);
}

/* Hash a cooked index entry by name pointer value.

   We can use pointer equality here because names come from .debug_str, which
   will normally be unique-ified by the linker.  Also, duplicates are relatively
   harmless -- they just mean a bit of extra memory is used.  */

struct cooked_index_entry_name_ptr_hash
{
  using is_avalanching = void;

  std::uint64_t operator () (const cooked_index_entry *entry) const noexcept
  {
    return ankerl::unordered_dense::hash<const char *> () (entry->name ());
  }
};

/* Compare cooked index entries by name pointer value.  */

struct cooked_index_entry_name_ptr_eq
{
  bool operator () (const cooked_index_entry *a,
		    const cooked_index_entry *b) const noexcept
  {
    return a->name () == b->name ();
  }
};

/* See cooked-index-shard.h.  */

bool
cooked_index_shard::resolve_deferred_names
	(const signature_to_name_map &sig_names)
{
  bool have_nameless_entries = false;
  for (cooked_index_entry *entry : m_entries)
    {
      if (!entry->name_is_deferred ())
	continue;

      const ULONGEST signature = entry->get_deferred_name ();

      if (const auto it = sig_names.find (signature);
	  it != sig_names.end ())
	entry->resolve_name (it->second);
      else
	{
	  have_nameless_entries = true;
	  complaint (_(DWARF_ERROR_PREFIX
		       "Cannot find signatured DIE %s referenced from DIE "
		       "at %s [in module %s]"),
		     hex_string (signature),
		     sect_offset_str (entry->die_offset),
		     entry->per_cu->per_bfd ()->filename ());
	}
    }

  return have_nameless_entries;
}

/* See cooked-index-shard.h.  */

void
cooked_index_shard::resolve_deferred_parents
	(const parent_map_map *parent_maps)
{
  gdb_assert (m_have_deferred_parents);

  for (cooked_index_entry *entry : m_entries)
    if (entry->parent_is_deferred ())
      {
	const cooked_index_entry *new_parent
	  = parent_maps->find (entry->get_deferred_parent ());
	entry->resolve_parent (new_parent);
      }
}

/* See cooked-index-shard.h.  */

void
cooked_index_shard::prune_nameless_entries ()
{
  for (cooked_index_entry *entry : m_entries)
    {
      /* Remove a parent reference if the parent has no name.  This
	 leaves ENTRY as an orphan, but this only happens if the DWARF
	 is corrupted and we failed to find a name for the parent.  We
	 can safely check the parent's name at this point because all
	 deferred names and parent links will have been resolved in
	 all shards before this is called on any shard.  */
      if (const cooked_index_entry *parent = entry->get_parent ();
	  parent != nullptr && parent->name_is_deferred ())
	entry->set_parent (nullptr);
    }

  /* If we failed to resolve the name of an entry via its signature
     then remove the entry from the m_entries vector.  This should be
     rare, and should only happen when we have corrupted DWARF.  The
     entries still live on the obstack, so parent pointers are still
     valid, but removing entries from the index means we don't try to
     search them when looking for index hits.  */
  m_entries.erase (std::remove_if (m_entries.begin (), m_entries.end (),
				   [] (const cooked_index_entry *e)
				   {
				     return e->name_is_deferred ();
				   }),
		   m_entries.end ());
}

/* See cooked-index-shard.h.  */

void
cooked_index_shard::canonicalize_names ()
{
  gdb::unordered_set<const cooked_index_entry *,
		     cooked_index_entry_name_ptr_hash,
		     cooked_index_entry_name_ptr_eq> seen_names;

  auto hash_entry = [] (const void *e)
    {
      const cooked_index_entry *entry = (const cooked_index_entry *) e;
      return dwarf5_djb_hash (entry->canonical);
    };

  auto eq_entry = [] (const void *a, const void *b) -> int
    {
      const cooked_index_entry *ae = (const cooked_index_entry *) a;
      const std::string_view *sv = (const std::string_view *) b;
      return (strlen (ae->canonical) == sv->length ()
	      && strncasecmp (ae->canonical, sv->data (), sv->length ()) == 0);
    };

  htab_up gnat_entries (htab_create_alloc (10, hash_entry, eq_entry,
					   nullptr, xcalloc, xfree));
  std::vector<cooked_index_entry *> new_gnat_entries;

  for (cooked_index_entry *entry : m_entries)
    {
      /* Deferred parents should not reach this point.  */
      gdb_assert (!entry->parent_is_deferred ());

      /* Entries without a name are filtered out during the call to
	 prune_nameless_entries.  */
      gdb_assert (!entry->name_is_deferred ());

      /* Note that this code must be kept in sync with
	 cooked_index::get_main -- if canonicalization is required
	 here, then a check might be required there.  */
      gdb_assert (entry->canonical == nullptr);
      if ((entry->flags & IS_LINKAGE) != 0)
	entry->canonical = entry->name ();
      else if (entry->lang == language_ada)
	{
	  /* Newer versions of GNAT emit DW_TAG_module and use a
	     hierarchical structure.  In this case, we don't need to
	     do any extra work.  This can be detected by looking for a
	     GNAT-encoded name.  */
	  if (strstr (entry->name (), "__") == nullptr)
	    {
	      entry->canonical = entry->name ();

	      /* If the entry does not have a parent, then there's
		 nothing extra to do here -- the entry itself is
		 sufficient.

		 However, if it does have a parent, we have to
		 synthesize an entry with the full name.  This is
		 unfortunate, but it's necessary due to how some of
		 the Ada name-lookup code currently works.  For
		 example, without this, ada_get_tsd_type will
		 fail.

		 Eventually it would be good to change the Ada lookup
		 code, and then remove these entries (and supporting
		 code in cooked_index_entry::full_name).  */
	      if (entry->get_parent () != nullptr)
		{
		  const char *fullname
		    = entry->full_name (&m_storage, FOR_ADA_LINKAGE_NAME);
		  cooked_index_entry *linkage = create (entry->die_offset,
							entry->tag,
							(entry->flags
							 | IS_LINKAGE
							 | IS_SYNTHESIZED),
							language_ada,
							fullname,
							nullptr,
							entry->per_cu);
		  linkage->canonical = fullname;
		  new_gnat_entries.push_back (linkage);
		}
	    }
	  else
	    handle_gnat_encoded_entry (entry, gnat_entries.get (),
				       new_gnat_entries);
	}
      else if (entry->lang == language_cplus || entry->lang == language_c)
	{
	  auto [it, inserted] = seen_names.insert (entry);

	  if (inserted)
	    {
	      /* No entry with that name was present, compute the canonical
		 name.  */
	      gdb::unique_xmalloc_ptr<char> canon_name
		= (entry->lang == language_cplus
		   ? cp_canonicalize_string (entry->name ())
		   : c_canonicalize_name (entry->name ()));
	      if (canon_name == nullptr)
		entry->canonical = entry->name ();
	      else
		entry->canonical = m_names.insert (std::move (canon_name));
	    }
	  else
	    {
	      /* An entry with that name was present, reuse its canonical
		 name.  */
	      entry->canonical = (*it)->canonical;
	    }
	}
      else
	entry->canonical = entry->name ();
    }

  /* Make sure any new Ada entries end up in the results.  This isn't
     done when creating these new entries to avoid invalidating the
     m_entries iterator used in the foreach above.  */
  m_entries.insert (m_entries.end (), new_gnat_entries.begin (),
		    new_gnat_entries.end ());

  m_entries.shrink_to_fit ();
  std::sort (m_entries.begin (), m_entries.end (),
	     [] (const cooked_index_entry *a, const cooked_index_entry *b)
	     {
	       return *a < *b;
	     });
}

/* See cooked-index-shard.h.  */

cooked_index_shard::range
cooked_index_shard::find (const std::string &name, bool completing) const
{
  struct comparator
  {
    cooked_index_entry::comparison_mode mode;

    bool operator() (const cooked_index_entry *entry,
		     const char *name) const noexcept
    {
      return cooked_index_entry::compare (entry->canonical, name, mode) < 0;
    }

    bool operator() (const char *name,
		     const cooked_index_entry *entry) const noexcept
    {
      return cooked_index_entry::compare (entry->canonical, name, mode) > 0;
    }
  };

  return std::make_from_tuple<range>
    (std::equal_range (m_entries.cbegin (), m_entries.cend (), name.c_str (),
		       comparator { (completing
				     ? cooked_index_entry::COMPLETE
				     : cooked_index_entry::MATCH) }));
}
