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

#include "objfiles.h"
#include "symtab.h"
#include "source.h"
#include "expanded-symbol.h"

/* See expanded-symbol.h.  */

symtab *
expanded_symbols_functions::find_last_source_symtab (objfile *objfile)
{
  if (objfile->compunit_symtabs.empty ())
    return nullptr;
  else
    return objfile->compunit_symtabs.back ().primary_filetab ();
}

/* See expanded-symbol.h.  */

enum language
expanded_symbols_functions::lookup_global_symbol_language
     (objfile *objfile, const char *name, domain_search_flags domain,
      bool *symbol_found_p)
{
  *symbol_found_p = false;
  return language_unknown;
}

/* See expanded-symbol.h.  */

bool
expanded_symbols_functions::search
     (objfile *objfile,
      search_symtabs_file_matcher file_matcher,
      const lookup_name_info *lookup_name,
      search_symtabs_symbol_matcher symbol_matcher,
      search_symtabs_expansion_listener listener,
      block_search_flags search_flags,
      domain_search_flags domain,
      search_symtabs_lang_matcher lang_matcher)
{
  /* This invariant is documented in quick-functions.h.  */
  gdb_assert (lookup_name != nullptr || symbol_matcher == nullptr);

  for (compunit_symtab &cu : objfile->compunits ())
    {
      if (lang_matcher != nullptr && !lang_matcher (cu.language ()))
	continue;

      if (file_matcher != nullptr)
	{
	  bool matched = false;
	  for (auto st : cu.filetabs ())
	    {
	      if (file_matcher (st->filename (), false))
		{
		  matched = true;
		  break;
		}
	      if ((basenames_may_differ
		   || file_matcher (lbasename (st->filename ()), true))
		  && file_matcher (symtab_to_fullname (st), false))
		{
		  matched = true;
		  break;
		}
	    }
	  if (!matched)
	    continue;
	}

      /* Here we simply call the listener (if any) without bothering to
	 consult lookup_name and symbol_matcher (if any).  This should be
	 okay since i) all symtabs are already expanded and ii) listeners
	 iterate over matching symbols themselves.  */
      if (listener != nullptr && !listener (&cu))
	return false;
    }
  return true;
}

/* See expanded-symbol.h.  */

symbol *
expanded_symbols_functions::find_symbol_by_address (objfile *objfile,
						    CORE_ADDR address)
{
  for (compunit_symtab &symtab : objfile->compunits ())
    {
      symbol *sym = symtab.symbol_at_address (address);
      if (sym != nullptr)
	return sym;
    }

  return nullptr;
}
