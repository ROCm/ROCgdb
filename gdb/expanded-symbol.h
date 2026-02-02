/* An implementation of "quick" symbol functions for already expanded
   symbol tables.

   Copyright (C) 2026-2026 Free Software Foundation, Inc.

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

#ifndef GDB_EXPANDED_SYMBOL_H
#define GDB_EXPANDED_SYMBOL_H

#include "quick-symbol.h"

/* An implementation of "quick" symbol functions that relies solely only on
   already present symtabs.  This is useful in cases where symbols are created
   eagerly by user-provided code (such as when using JIT reader API).  */

struct expanded_symbols_functions : public quick_symbol_functions
{
  bool has_symbols (objfile *objfile) override
  {
    return true;
  }

  bool has_unexpanded_symtabs (objfile *objfile) override
  {
    return false;
  }

  symtab *find_last_source_symtab (objfile *objfile) override;

  void forget_cached_source_info (objfile *objfile) override
  {
  }

  enum language lookup_global_symbol_language (objfile *objfile,
					       const char *name,
					       domain_search_flags domain,
					       bool *symbol_found_p) override;
  void print_stats (objfile *objfile, bool print_bcache) override
  {
  }

  void dump (objfile *objfile) override
  {
  }

  void expand_all_symtabs (objfile *objfile) override
  {
  }

  bool search (objfile *objfile,
	       search_symtabs_file_matcher file_matcher,
	       const lookup_name_info *lookup_name,
	       search_symtabs_symbol_matcher symbol_matcher,
	       search_symtabs_expansion_listener listener,
	       block_search_flags search_flags, domain_search_flags domain,
	       search_symtabs_lang_matcher lang_matcher) override;

  compunit_symtab *find_pc_sect_compunit_symtab
    (objfile *objfile, bound_minimal_symbol msymbol, CORE_ADDR pc,
     obj_section *section, int warn_if_readin) override
  {
    /* Simply returning NULL here is okay since the (only) caller
       find_compunit_symtab_for_pc_sect iterates over existing CUs
       anyway.  */
    return nullptr;
  }

  symbol *find_symbol_by_address (objfile *objfile, CORE_ADDR address)
    override;

  void map_symbol_filenames (objfile *objfile, symbol_filename_listener fun,
			     bool need_fullname) override
  {
  }
};


#endif /* GDB_EXPANDED_SYMBOL_H */
