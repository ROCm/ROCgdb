/* PDB lazy (cooked-index) symbol lookup.

   Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
   along with this program.  If not, see <http://www.gnu.org/licenses/>

   This file implements the PDB cooked index and the quick_symbol_functions
   that drive lazy module loading.

   The PDB cooked index is a name -> owning-module map that lets a symbol
   query (a quick function) expand only the module(s) that define the name.
   It is built on the DWARF cooked-index model (parallel shards).

   A fast name index already ships in the PDB as the GSI (global symbol
   index) hash, when present.  We do not consume that hash table directly:
   it is a multi-layered structure with a complex lookup algorithm.
   that are easy to read in parallel.

   To build the cooked index we use the GSI function references (S_PROCREF /
   S_LPROCREF) and globals from the global symbol stream, to which the GSI
   points (S_GPROC32 / S_LPROC32).  A referenced global is expanded when a
   quick function needs it.  Globals can also be eagerly built into the
   <pdb-globals> CUs using a maintenance set command.

   Note that the types are not handled by the cooked index.  Unlike GSI,
   the types stream cannot be split across multiple threads, as each TPI
   record is of variable length (GSI records are fixed length).  In
   addition, our design is to eagerly build a type-name to type-index map,
   which we use later to build the type when a quick function needs
   it.

   With no GSI (or under "maintenance set pdb-force-module-index") the worker
   scans every module symbol stream instead, indexing the full records it
   finds there — S_GPROC32 / S_LPROC32 (functions), S_GDATA32 / S_LDATA32
   (data) and S_UDT (module-local type aliases).

   Besides the name map, the build also fills a PC -> module address map
   (addrmap_fixed) from the DBI section contributions, backing the
   by-address queries.

   QF search:
     - by name: first walk the already-built compunits (<pdb-globals>,
       <pdb-types>, any expanded module); on a miss, consult the index and
       expand each owning module.  A type-domain query first builds
       that type.
     - by PC: the address map resolves PC to its module and expands it.  The
       map holds one range per module, taken from the module's DBI section
       contribution.  That contribution is only the module's primary code
       range; under /O2 the linker splits a module's functions across several
       ranges but the DBI records only the first, so a function placed past it
       is not in the map.  On a miss the search falls back to scanning the
       already-expanded modules' compunit global-block ranges: expanding a
       module extends its block to span all its function ranges, so the
       split-out function is found there (only for modules already expanded by
       a prior query).
     - by address (find_symbol_by_address) reuses the PC->module map, and
       language (lookup_global_symbol_language) reuses the name map; neither
       adds a new lookup.

    */

#include "pdb/pdb.h"

#include "objfiles.h"
#include "symtab.h"
#include "block.h"
#include "addrmap.h"
#include "language.h"
#include "filenames.h"
#include "quick-symbol.h"
#include "event-top.h"
#include "run-on-main-thread.h"
#include "command.h"
#include "cli/cli-cmds.h"
#include "exceptions.h"
#include "pdb/pdb-internal.h"
#include "gdbsupport/thread-pool.h"
#include "gdbsupport/parallel-for.h"
#include "gdbsupport/gdb_obstack.h"
#include "gdbsupport/pathstuff.h"
#include "gdbsupport/unordered_set.h"
#include "c-ctype.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pdb
{

/* Quick functions debug statements (shared verbosity with pdb.c).  */
#define pdb_qf_printf(fmt, ...) \
  debug_prefixed_printf_cond (pdb_qf_debug >= 1, "pdb qf", fmt, ##__VA_ARGS__)

/* "maintenance set pdb-force-module-index": build the cooked index by
   scanning modules even when the PDB includes GSI.  */
bool pdb_force_module_index = false;

/* "maintenance set pdb-lazy-globals": do not build <pdb-globals> eagerly at
   load; index globals and build each on the by-name lookup that asks for
   it.  For measuring eager-vs-lazy  load time.  */
bool pdb_lazy_globals = true;

/* "maintenance set pdb-synchronous": build the cooked index on the calling
   thread instead of a worker.  */
bool pdb_synchronous = false;

/* Whether RECTYPE is a procedure record.  */

static bool
pdb_is_proc_record (uint16_t rectype)
{
  return (rectype == S_GPROC32 || rectype == S_LPROC32
	  || rectype == S_GPROC32_ID || rectype == S_LPROC32_ID);
}

/* The name of the RECTYPE record whose body is [REC_DATA, REC_END), or
   nullptr when the index does not list that kind or the name does not end
   inside the record.  */

static const char *
pdb_indexed_record_name (uint16_t rectype, const gdb_byte *rec_data,
			 const gdb_byte *rec_end)
{
  size_t body = rec_end - rec_data;
  size_t name_offs;

  if (pdb_is_proc_record (rectype))
    name_offs = PDB_SYMBOL_FUNC_NAME_OFFS;
  else if (rectype == S_GDATA32 || rectype == S_LDATA32)
    name_offs = PDB_SYMBOL_VAR_NAME_OFFS;
  else if (rectype == S_UDT)
    name_offs = PDB_SYMBOL_UDT_NAME_OFFS;
  else if (rectype == S_CONSTANT)
    {
      if (body <= PDB_SYMBOL_CONST_VALUE_OFFS)
	return nullptr;

      uint64_t value;
      uint32_t leaf
	= pdb_cv_read_numeric (rec_data + PDB_SYMBOL_CONST_VALUE_OFFS,
			       body - PDB_SYMBOL_CONST_VALUE_OFFS, &value);
      if (leaf == 0)
	return nullptr;
      name_offs = PDB_SYMBOL_CONST_VALUE_OFFS + leaf;
    }
  else
    return nullptr;

  if (body <= name_offs)
    return nullptr;
  return pdb_extract_string (rec_data + name_offs, rec_end);
}

/* The index follows the DWARF cooked index (dwarf2/cooked-index-entry.h).
   Each shard keeps one list of entries sorted by unqualified name, and
   every entry points to the entry of the scope that encloses it.  A PDB
   records a qualified name as one string, so the scopes are cut out of it
   when the name is indexed: "fubar::inner::deep_func" gives

     fubar         scope
     inner         scope, parent fubar
     deep_func     parent inner

   A scope entry stands where DWARF has a DW_TAG_namespace or class entry.
   Only the names say what encloses what, so namespaces and classes are
   not told apart.  Each module gets its own scope entries, as each DWARF
   unit has its own namespace DIEs.

   A lookup binary-searches the last segment of the name it is given and
   checks each earlier segment against the entry's parents.  "break
   deep_func", "break inner::deep_func" and "break fubar::inner::deep_func"
   all find deep_func; completion also reaches the scope entries, so
   "fu<TAB>" finds fubar before any of its members.  */

/* Record type of a scope entry.  No S_* record kind is 0.  */
static constexpr uint16_t PDB_INDEX_SCOPE = 0;

/* One entry in the cooked index.  */

struct pdb_cooked_index_entry
{
  /* The qualified name.  Borrowed from the symbol records or copied onto
     the shard's obstack.  */
  const char *name;

  /* The entry of the enclosing scope, or nullptr.  */
  const pdb_cooked_index_entry *parent;

  uint16_t module_index;
  uint16_t rectype;

  /* Length of NAME's nested-name-specifier: 11 in "ns::Class::func", 0
     when NAME has none.  */
  uint16_t key_off;

  /* NAME without its nested-name-specifier: "func" for "ns::Class::func".
     The shard is sorted and searched on this.  */

  const char *unqualified_name () const
  { return name + key_off; }

  /* The GDB search domain(s) this record kind can satisfy.  */
  domain_search_flags search_domain () const
  {
    switch (rectype)
      {
      case S_GPROC32:
      case S_LPROC32:
      case S_GPROC32_ID:
      case S_LPROC32_ID:
      case S_PROCREF:
      case S_LPROCREF:
      case S_INLINESITE:
    	return SEARCH_FUNCTION_DOMAIN;
      case S_GDATA32:
      case S_LDATA32:
      case S_DATAREF:
	return SEARCH_VAR_DOMAIN;
      case S_UDT:
	return SEARCH_TYPE_DOMAIN | SEARCH_STRUCT_DOMAIN;
      case PDB_INDEX_SCOPE:
	return SEARCH_TYPE_DOMAIN;
      default:
	return SEARCH_VAR_DOMAIN;
      }
  }

  bool matches (domain_search_flags domain) const
  {
    return (search_domain () & domain) != 0;
  }
};

/* Comparison modes of pdb_index_compare.  */

enum class pdb_index_compare_mode
{
  MATCH,
  SORT,
  COMPLETE,
};

/* A copy of cooked_index_entry::compare, which the PDB reader cannot call
   when GDB is configured without DWARF support.  Compare STRA with STRB
   case-insensitively, treating '<' as the end of a name so that "func"
   finds "func<int>".  In MATCH and COMPLETE mode STRB is the name
   searched for; COMPLETE also accepts STRB ending first.  */

static int
pdb_index_compare (const char *stra, const char *strb,
		   pdb_index_compare_mode mode)
{
  auto munge = [] (char c) -> unsigned char
    {
      if (c == '<')
	return '\0';
      return c_tolower (c);
    };

  unsigned char a = munge (*stra);
  unsigned char b = munge (*strb);

  while (a != '\0' && b != '\0' && a == b)
    {
      a = munge (*++stra);
      b = munge (*++strb);
    }

  if (a == b)
    return 0;

  if (mode == pdb_index_compare_mode::COMPLETE && b == '\0')
    return 0;

  return a < b ? -1 : 1;
}

/* Typed view over the GSI hash-record bytes (gsi->hr) so parallel_for_each
   can split them by index.  Layout: microsoft-pdb gsi.h HRFile.  */

struct gsi_hash_record
{
  uint32_t off;
  uint32_t cref;
};
static_assert (sizeof (gsi_hash_record) == GSI_HASH_RECORD_SIZE,
	       "gsi_hash_record must match the on-disk record size");

/* A scope within one module, the key of pdb_index_shard::scopes.  */

struct pdb_scope_key
{
  uint16_t module_index;
  std::string_view name;

  bool operator== (const pdb_scope_key &other) const
  { return module_index == other.module_index && name == other.name; }
};

struct pdb_scope_key_hash
{
  size_t operator() (const pdb_scope_key &key) const
  { return std::hash<std::string_view> () (key.name) ^ key.module_index; }
};

/* One shard of the cooked index: built by a single worker from
   its slice of the CUs.  Entries are sorted by name once the worker
   finishes building its slice.  */

struct pdb_index_shard
{
  /* Sorted by unqualified name once the shard is built.  */
  std::vector<pdb_cooked_index_entry *> entries;

  /* The entries, and the names a module scan or a scope entry copies.
     One obstack per shard, so workers never share one.  */
  auto_obstack storage;

  /* Scope entries already made, so a module gets one per scope.  Needed
     only while the shard is built.  */
  std::unordered_map<pdb_scope_key, pdb_cooked_index_entry *,
		     pdb_scope_key_hash> scopes;

  /* Add NAME, of LEN bytes, with an entry for each scope enclosing it.
     NAME must outlive the index.  */

  void add (const char *name, size_t len, uint16_t module_index,
	    uint16_t rectype)
  {
    const pdb_cooked_index_entry *parent = nullptr;
    size_t off = 0;

    while (true)
      {
	size_t end = off + pdb_scope_component_len (name + off);
	if (name[end] == '\0' || end + 2 >= len)
	  break;

	parent = scope (name, end, off, module_index, parent);
	off = end + 2;
      }

    push (name, off, module_index, rectype, parent);
  }

  /* Sort the entries for searching.  */

  void sort ()
  {
    std::sort (entries.begin (), entries.end (),
	       [] (const pdb_cooked_index_entry *a,
		   const pdb_cooked_index_entry *b)
	       {
		 return pdb_index_compare (a->unqualified_name (),
					   b->unqualified_name (),
					   pdb_index_compare_mode::SORT) < 0;
	       });
    scopes = {};
  }

  /* Catch exceptions from calling F (), and add them to the list of caught
     exceptions.  These are passed forward and printed by the main thread,
     as cooked_index_worker_result::catch_error does.  */

  template <typename F>
  void catch_error (F &&f)
  {
    try
      {
	f ();
      }
    catch (gdb_exception &ex)
      {
	m_exceptions.push_back (std::move (ex));
      }
  }

  /* Print the caught exceptions not yet in SEEN_EXCEPTIONS.  Main thread
     only.  */

  void emit_exceptions (gdb::unordered_set<gdb_exception> &seen_exceptions)
  {
    gdb_assert (is_main_thread ());

    /* Only show a given exception a single time.  */
    for (auto &one_exc : m_exceptions)
      if (seen_exceptions.insert (one_exc).second)
	exception_print (gdb_stderr, one_exc);
  }

private:

  /* Exceptions caught while building this shard.  */
  std::vector<gdb_exception> m_exceptions;

  pdb_cooked_index_entry *push (const char *name, size_t key_off,
				uint16_t module_index, uint16_t rectype,
				const pdb_cooked_index_entry *parent)
  {
    pdb_cooked_index_entry *e = XOBNEW (&storage, pdb_cooked_index_entry);
    *e = { name, parent, module_index, rectype, (uint16_t) key_off };
    entries.push_back (e);
    return e;
  }

  /* The entry for the scope spelled by the first LEN bytes of NAME, whose
     own last component starts at KEY_OFF.  */

  const pdb_cooked_index_entry *
  scope (const char *name, size_t len, size_t key_off, uint16_t module_index,
	 const pdb_cooked_index_entry *parent)
  {
    auto it = scopes.find ({ module_index, std::string_view (name, len) });
    if (it != scopes.end ())
      return it->second;

    const char *copy = obstack_strndup (&storage, name, len);
    pdb_cooked_index_entry *e = push (copy, key_off, module_index,
				      PDB_INDEX_SCOPE, parent);
    scopes.emplace (pdb_scope_key { module_index,
				    std::string_view (copy, len) }, e);
    return e;
  }
};

/* Build state.  The worker moves INITIAL -> FINALIZED once every module
   is scanned and the entry vector is sorted.  */

enum class pdb_index_state
{
  INITIAL,
  FINALIZED,
};

/* The PDB cooked index: name -> owning module.  Built once at load, off
   the main thread: start() posts a single thread-pool task (do_build)
   that fans out into parallel per-shard workers — seeding from the GSI
   when present, else scanning every module's symbol stream; the build
   therefore touches neither gdb_stdlog nor the objfile obstack.  After
   FINALIZED the shards are immutable and read by the main thread; the
   state mutex provides the happens-before.  */

class pdb_cooked_index
{
public:
  explicit pdb_cooked_index (pdb_per_objfile *pdb)
    : m_pdb (pdb)
  {
  }

  ~pdb_cooked_index ()
  {
    /* Join the worker before its storage (m_shards / m_storage / the
       condition variable) is destroyed.  */
    if (m_posted)
      m_future.wait ();
  }

  DISABLE_COPY_AND_ASSIGN (pdb_cooked_index);

  /* Start the build: on this thread with pdb-synchronous, otherwise as a
     thread-pool task, which the pool runs inline when it has no worker
     threads (thread_pool::do_post_task), as the DWARF cooked-index worker
     relies on.  */
  void start ()
  {
    if (pdb_synchronous)
      {
	build_or_fail ();
	return;
      }

    m_posted = true;
    m_future = gdb::thread_pool::g_thread_pool->post_task ([this] ()
      {
	build_or_fail ();
      });
  }

  /* Block until the index is FINALIZED.  With ALLOW_QUIT, poll the quit
     flag so a long build stays interruptible from the main thread.  */
  void wait (bool allow_quit = true)
  {
    auto wstart = std::chrono::steady_clock::now ();
    bool blocked = false;
    {
      std::unique_lock<std::mutex> lock (m_mutex);
      blocked = m_state != pdb_index_state::FINALIZED;

      /* A query arrived before the background build finished: it must
	 wait here.  Announced before the wait, not after, so a long block
	 is visible while it happens.  Main thread only, so printing is
	 prompt-safe: this runs inside the command that issued the query.  */
      if (blocked && is_main_thread () && pdb_read_debug >= 1)
	{
	  lock.unlock ();
	  gdb_printf ("[pdb qf] query blocking: cooked-index build not"
		      " finalized yet, waiting...\n");
	  lock.lock ();
	}

      while (m_state != pdb_index_state::FINALIZED)
	{
	  if (allow_quit)
	    {
	      m_cond.wait_for (lock, std::chrono::milliseconds (15));
	      if (m_state == pdb_index_state::FINALIZED)
		break;
	      lock.unlock ();
	      QUIT;
	      lock.lock ();
	    }
	  else
	    m_cond.wait (lock);
	}
    }

    /* Report how long the query actually waited (only under "set debug
       pdb", main thread only where printing is safe).  */
    if (blocked && is_main_thread () && pdb_read_debug >= 1)
      {
	double ms = std::chrono::duration<double, std::milli> (
		      std::chrono::steady_clock::now () - wstart).count ();
	gdb_printf ("[pdb qf] query unblocked: waited %.2f ms for the"
		    " cooked-index build\n", ms);
      }

    spill_finalize_log ();
  }

  /* Print the finalize summary once, on the main thread, from the first
     wait() that observes FINALIZED.  The worker itself must never call this
     (ui_file is not thread-safe), and it is intentionally not printed
     asynchronously at finalize: doing so from the event loop clobbers the
     readline prompt, which is why GDB's own cooked-index worker also defers
     its output to a wait() in a command context.  m_shards / m_source /
     m_build_ms are immutable after FINALIZED.  */
  void spill_finalize_log ()
  {
    if (!is_main_thread ())
      return;

    if (!m_reported)
      {
	m_reported = true;
	if (m_failed.has_value ())
	  {
	    /* The build failed -- report it.  */
	    exception_print (gdb_stderr, *m_failed);
	    m_failed.reset ();
	  }
	else
	  {
	    gdb::unordered_set<gdb_exception> seen_exceptions;
	    for (auto &shard : m_shards)
	      shard->emit_exceptions (seen_exceptions);
	  }
      }

    if (m_log_printed || pdb_read_debug < 1)
      return;
    m_log_printed = true;
    gdb_printf ("[pdb qf] cooked index finalized: %zu entries from %s"
		" over %zu modules in %.2f ms"
		" (build %.2f [%zu shards], addrmap %.2f)\n",
		total_entries (), m_source, m_pdb->modules.size (),
		m_build_ms, m_seed_ms, m_shards.size (), m_addrmap_ms);
  }

  /* True if the build has finished (non-blocking peek).  */
  bool finalized ()
  {
    std::lock_guard<std::mutex> guard (m_mutex);
    return m_state == pdb_index_state::FINALIZED;
  }

  /* Invoke FN for every entry that LOOKUP names, across all shards,
     stopping early when FN returns true.  Waits for FINALIZED.  Follows
     the DWARF cooked-index search (cooked_index_functions::search): find
     the last segment of the name, check each earlier segment against the
     entry's parents, then match the full name, with SYMBOL_MATCHER when
     one is given.  A name may match in more than one shard, as it may span
     modules.  */

  template<typename F>
  void for_each_named (const lookup_name_info &lookup,
		       search_symtabs_symbol_matcher symbol_matcher, F fn)
  {
    wait ();

    lookup_name_info without_params = lookup.make_ignore_params ();
    bool completing = lookup.completion_mode ();
    symbol_name_match_type match_type = without_params.match_type ();
    const language_defn *lang = language_def (language_cplus);

    std::vector<std::string_view> segments
      = without_params.split_name (language_cplus);
    if (segments.empty ())
      return;

    std::vector<std::string> names (segments.begin (), segments.end ());
    std::vector<lookup_name_info> segment_lookups;
    segment_lookups.reserve (names.size ());
    for (const std::string &name : names)
      segment_lookups.emplace_back (name, match_type, completing, true);

    symbol_name_matcher_ftype *full_matcher
      = lang->get_symbol_name_matcher (without_params);

    struct comparator
    {
      pdb_index_compare_mode mode;

      bool operator() (const pdb_cooked_index_entry *e,
		       const char *name) const
      { return pdb_index_compare (e->unqualified_name (), name, mode) < 0; }

      bool operator() (const char *name,
		       const pdb_cooked_index_entry *e) const
      { return pdb_index_compare (e->unqualified_name (), name, mode) > 0; }
    };

    comparator cmp { completing ? pdb_index_compare_mode::COMPLETE
				: pdb_index_compare_mode::MATCH };

    for (const auto &shard : m_shards)
      {
	auto range = std::equal_range (shard->entries.begin (),
				       shard->entries.end (),
				       names.back ().c_str (), cmp);
	for (auto it = range.first; it != range.second; ++it)
	  {
	    const pdb_cooked_index_entry *e = *it;

	    bool found = true;
	    const pdb_cooked_index_entry *parent = e->parent;
	    for (size_t i = names.size () - 1; i > 0; --i)
	      {
		if (parent == nullptr)
		  {
		    found = false;
		    break;
		  }

		symbol_name_matcher_ftype *name_matcher
		  = lang->get_symbol_name_matcher (segment_lookups[i - 1]);
		if (!name_matcher (parent->unqualified_name (),
				   segment_lookups[i - 1], nullptr))
		  {
		    found = false;
		    break;
		  }

		parent = parent->parent;
	      }

	    if (!found)
	      continue;

	    /* Looking for "a::b" must not find "x::a::b".  */
	    if ((match_type == symbol_name_match_type::FULL
		 || match_type == symbol_name_match_type::EXPRESSION)
		&& parent != nullptr)
	      continue;

	    if (symbol_matcher == nullptr)
	      {
		if (!full_matcher (e->name, without_params, nullptr))
		  continue;
	      }
	    else if (!symbol_matcher (e->name))
	      continue;

	    if (fn (*e))
	      return;
	  }
      }
  }

  /* Module whose DBI section contribution covers PC, or nullptr.  An
     O(log n) address-map lookup, the analogue of DWARF's cooked-index
     address map (dwarf2_per_bfd::index_table).  Waits for FINALIZED.  */
  pdb_module_info *find_pc_module (CORE_ADDR pc)
  {
    wait ();
    if (m_pc_map == nullptr)
      return nullptr;

    /* The map holds unrelocated addresses, so undo the load bias here.  */
    CORE_ADDR unrelocated = pc - m_pdb->objfile->text_section_offset ();
    return static_cast<pdb_module_info *> (m_pc_map->find (unrelocated));
  }

  /* Counts for "maintenance print statistics".  Waits for FINALIZED.  */
  size_t num_entries ()
  {
    wait ();
    return total_entries ();
  }

  /* Print the index for "maintenance print objfiles", mirroring
     cooked_index::dump.  Waits for FINALIZED.  */
  void dump (gdbarch *arch)
  {
    wait ();

    gdb_printf ("PDB cooked index in use:\n");
    gdb_printf ("\n");
    gdb_printf ("  source:  %s\n", m_source);
    gdb_printf ("  shards:  %zu\n", m_shards.size ());
    gdb_printf ("  entries: %zu\n", total_entries ());

    /* Set only by compute_main_name; find_main_name's fallback records its
       answer in the program space, not here.  */
    const char *main_name = m_pdb->objfile->per_bfd->name_of_main;
    gdb_printf ("  main:    %s (%s)\n",
		main_name != nullptr ? main_name : "<not set>",
		language_str (m_pdb->objfile->per_bfd->language_of_main));

    if (pdb_read_debug >= 2)
      {
	gdb_printf ("\n");
	gdb_printf ("  entries:\n");
	gdb_printf ("\n");

	size_t i = 0;
	for (const auto &shard : m_shards)
	  for (const pdb_cooked_index_entry *e : shard->entries)
	    {
	      QUIT;
	      const char *modname = "?";
	      if (e->module_index < m_pdb->modules.size ()
		  && m_pdb->modules[e->module_index].module_name != nullptr)
		modname = lbasename (m_pdb->modules[e->module_index].module_name);

	      gdb_printf ("    [%zu] name:   %s\n", i++, e->name);
	      if (e->rectype == PDB_INDEX_SCOPE)
		gdb_printf ("        record: scope\n");
	      else
		gdb_printf ("        record: %s\n",
			    pdb_sym_rec_type_name (e->rectype).c_str ());
	      if (e->parent != nullptr)
		gdb_printf ("        parent: %s\n", e->parent->name);
	      domain_search_flags dom = e->search_domain ();
	      if (dom & SEARCH_VAR_DOMAIN)
		gdb_printf ("        domain: VARIABLES_DOMAIN\n");
	      else if (dom & SEARCH_FUNCTION_DOMAIN)
		gdb_printf ("        domain: FUNCTIONS_DOMAIN\n");
	      else if (dom & SEARCH_TYPE_DOMAIN)
		gdb_printf ("        domain: TYPES_DOMAIN\n");
	      else if (dom & SEARCH_STRUCT_DOMAIN)
		gdb_printf ("        domain: STRUCT_DOMAIN\n");
	      else
		gdb_printf ("        domain: 0x%x\n", (unsigned) dom);
	      gdb_printf ("        module: %u %s\n", e->module_index, modname);
	      gdb_printf ("\n");
	    }
      }

    gdb_printf ("\n");
    gdb_printf ("  address map:\n");
    gdb_printf ("\n");
    if (m_pc_map == nullptr)
      gdb_printf ("    ((addrmap *) 0)\n");
    else
      m_pc_map->foreach ([arch] (CORE_ADDR start_addr, const void *obj)
	{
	  QUIT;
	  const char *start = paddress (arch, start_addr);
	  if (obj != nullptr)
	    {
	      const pdb_module_info *mod
		= static_cast<const pdb_module_info *> (obj);
	      gdb_printf ("    [%s] %s\n", start,
			  mod->module_name != nullptr
			    ? lbasename (mod->module_name) : "?");
	    }
	  else
	    gdb_printf ("    [%s] ((pdb_module_info *) 0)\n", start);
	  return 0;
	});
    gdb_printf ("\n");
  }

private:

  /* Run do_build.  On failure keep the exception and publish an empty
     index, as cooked_index_worker::start does, so waiters are released and
     the failure can be reported.  */

  void build_or_fail ()
  {
    try
      {
	do_build ();
      }
    catch (const gdb_exception &exc)
      {
	m_failed = exc;
	m_shards.clear ();
	set_finalized ();
      }
  }

  /* Worker body: seed the index from the GSI or a module scan, build the
     PC->module map, then finalize.  */

  void do_build ()
  {
    using clock = std::chrono::steady_clock;
    auto start = clock::now ();

    auto seed_start = clock::now ();
    bool gsi = has_gsi ();

    if (gsi && !pdb_force_module_index)
      {
	m_source = "GSI";
	build_from_gsi ();
      }

    /* The GSI lists only procedures, so scan the modules for the other
       records, and for procedures too when the GSI gave none.  */
    build_from_module_scan (gsi, total_entries () == 0);

    auto addrmap_start = clock::now ();
    m_seed_ms = std::chrono::duration<double, std::milli> (
		  addrmap_start - seed_start).count ();

    build_pc_addrmap ();

    auto build_end = clock::now ();
    m_addrmap_ms = std::chrono::duration<double, std::milli> (
		     build_end - addrmap_start).count ();
    m_build_ms = std::chrono::duration<double, std::milli> (
		   build_end - start).count ();

    set_finalized ();
  }

  /* True when the PDB has a GSI.  */
  bool has_gsi () const
  {
    return m_pdb->gsi_stream != 0 && m_pdb->gsi_stream != 0xFFFF
	   && !m_pdb->sym_record_data.empty ();
  }

  /* Build the PC -> module address map from every code contribution of the
     DBI Section Contribution substream, as DWARF maps each unit's PC
     ranges, then fill gaps from each module header's one contribution.
     Backs find_pc_module.  Reads only load-immutable data, so it is safe
     on the worker; the map is allocated in m_storage.

     Addresses are stored unrelocated: the worker runs before the inferior
     starts, so the load bias is not known yet and would be baked in wrong.
     find_pc_module applies it at query time, as DWARF's index does.  */
  void build_pc_addrmap ()
  {
    addrmap_mutable mut;
    bool any = false;
    auto add = [&] (uint16_t section, uint32_t offset, uint32_t size,
		    pdb_module_info *mod)
      {
	if (!m_pdb->section_valid (section) || size == 0)
	  return;
	CORE_ADDR low = m_pdb->map_section_offset_unrelocated (section, offset);
	if (low == 0)
	  return;
	mut.set_empty (low, low + size - 1, mod);
	any = true;
      };

    for (const auto &c : m_pdb->code_contribs)
      add (c.section, c.offset, c.size, &m_pdb->modules[c.module_index]);
    for (pdb_module_info &mod : m_pdb->modules)
      add (mod.sc_section, mod.sc_offset, mod.sc_size, &mod);

    if (any)
      m_pc_map = new (&m_storage) addrmap_fixed (&m_storage, &mut);
  }

  /* Build the cooked index from the GSI hash records.  Each worker takes
     a slice of the records, resolves each S_PROCREF / S_LPROCREF into a
     {name, owning module} entry in its own shard, and sorts that shard.  */

  void build_from_gsi ()
  {
    auto gsi_buf = m_pdb->read_stream (m_pdb->gsi_stream);
    if (gsi_buf.empty ())
      return;

    auto gsi = pdb_parse_gsi_hash_header (gsi_buf.data (), gsi_buf.size ());
    if (!gsi)
      return;

    const gsi_hash_record *recs
      = reinterpret_cast<const gsi_hash_record *> (gsi->hr.data ());
    size_t num = gsi->hr.size () / GSI_HASH_RECORD_SIZE;

    size_t nthreads = std::max<size_t> (
			 gdb::thread_pool::g_thread_pool->thread_count (), 1);

    /* Sequential single shard when the pool has no other threads.  */
    if (nthreads < 2 || num < 2)
      {
	auto shard = std::make_unique<pdb_index_shard> ();

	for (size_t i = 0; i < num; i++)
	  shard->catch_error ([&] ()
	    {
	      parse_gsi_record (recs[i], shard.get ());
	    });

	collect_shard (std::move (shard));
	return;
      }

    gdb::parallel_for_each<1, const gsi_hash_record *, pdb_gsi_worker>
      (recs, recs + num, this);
  }

  /* Resolve one GSI hash record into SHARD.  NAME points into global symbol
     record data (on objfile's obstack).  */

  void parse_gsi_record (const gsi_hash_record &rec, pdb_index_shard *shard)
  {
    uint32_t offs = read_u32 (&rec);
    if (offs == 0)
      return;

    offs -= 1;

    size_t syms_size = m_pdb->sym_record_data.size ();
    if (offs > syms_size || syms_size - offs < PDB_RECORD_HDR_SIZE)
      return;

    const gdb_byte *syms = m_pdb->sym_record_data.data ();
    const gdb_byte *syms_end = syms + syms_size;
    const gdb_byte *sym_start = syms + offs;

    auto hdr = pdb_parse_sym_record_hdr (sym_start, syms_end);
    if (!hdr)
      return;

    if (hdr->type != S_PROCREF && hdr->type != S_LPROCREF)
      return;

    const gdb_byte *rec_data = sym_start + PDB_RECORD_DATA_OFFS;
    const gdb_byte *rec_end = sym_start + hdr->rec_size ();
    if (rec_end - rec_data <= PDB_SYMBOL_REF_NAME_OFFS)
      return;

    uint16_t imod = read_u16 (rec_data + PDB_SYMBOL_REF_MOD_INDEX_OFFS);
    const char *name = pdb_extract_string (rec_data + PDB_SYMBOL_REF_NAME_OFFS,
					   rec_end);
    if (name == nullptr)
      return;

    /* imod is 1-based; 0 means "no module".  */
    if (imod == 0 || imod > m_pdb->modules.size ())
      return;

    size_t name_len = strlen (name);
    if (name_len > UINT16_MAX)
      return;

    shard->add (name, name_len, (uint16_t) (imod - 1), hdr->type);
  }

  /* One parallel GSI worker.  It accumulates the records in the batches
     it is handed into its own shard, then hands that shard to the index
     on destruction — the DWARF parallel_indexing_worker shape, where
     each worker owns a shard and appends it to the parent when done.  */

  struct pdb_gsi_worker
  {
    explicit pdb_gsi_worker (pdb_cooked_index *idx)
      : m_idx (idx)
    {
    }

    DISABLE_COPY_AND_ASSIGN (pdb_gsi_worker);

    ~pdb_gsi_worker ()
    {
      m_idx->collect_shard (std::move (m_shard));
    }

    void operator() (::iterator_range<const gsi_hash_record *> batch)
    {
      for (const gsi_hash_record &rec : batch)
	m_shard->catch_error ([&] ()
	  {
	    m_idx->parse_gsi_record (rec, m_shard.get ());
	  });
    }

    pdb_cooked_index *m_idx;
    std::unique_ptr<pdb_index_shard> m_shard { new pdb_index_shard };
  };

  /* Sort SHARD and append it to m_shards under the shards mutex.  Called
     from a worker thread (on the worker's destruction) or the do_build
     thread (the sequential fallback).  */

  void collect_shard (std::unique_ptr<pdb_index_shard> shard)
  {
    if (shard == nullptr)
      return;
    shard->sort ();
    std::lock_guard<std::mutex> guard (m_shards_mutex);
    m_shards.push_back (std::move (shard));
  }

  /* Build the index by scanning module symbol streams, taking procedures
     only when PROCS.  Parallel across modules when the pool has >= 2
     threads and there are >= 2 modules; otherwise one shard scans them
     sequentially.  */

  void build_from_module_scan (bool gsi, bool procs)
  {
    if (procs)
      m_source = gsi ? "forced module scan" : "module scan";

    size_t nmod = m_pdb->modules.size ();
    size_t nthreads = std::max<size_t> (
			 gdb::thread_pool::g_thread_pool->thread_count (), 1);

    if (nthreads >= 2 && nmod >= 2)
      {
	std::vector<uint16_t> mod_indices (nmod);
	for (uint16_t i = 0; i < nmod; i++)
	  mod_indices[i] = i;

	gdb::parallel_for_each<1, const uint16_t *, pdb_scan_worker>
	  (mod_indices.data (), mod_indices.data () + mod_indices.size (),
     this, procs);
	return;
      }

    /* One shard - just scan the modules sequentially.  */
    auto shard = std::make_unique<pdb_index_shard> ();
    gdb::byte_vector scratch;
    for (uint16_t i = 0; i < nmod; i++)
      shard->catch_error ([&] ()
	{
	  scan_module (i, shard.get (), scratch, procs);
	});

    collect_shard (std::move (shard));
  }

  /* Total entries across all shards.  Call after FINALIZED.  */

  size_t total_entries () const
  {
    size_t n = 0;
    for (const auto &shard : m_shards)
      n += shard->entries.size ();
    return n;
  }

  /* Walk the symbol records in a module's symbol stream buffer
     [PDB_MODULE_SYMBOLS_OFFS, sym_byte_size) and call EMIT(name, rectype)
     for each indexed record outside every procedure, block and inline
     site, as DWARF indexes no function-local names.  Procedures are
     emitted only when PROCS.  NAME points into MODULE_STREAM and is valid
     only for the duration of the EMIT call.  */
  template<typename F>
  static void walk_module_syms (const gdb_byte *module_stream,
				uint32_t sym_byte_size, bool procs, F emit)
  {
    const gdb_byte *data = module_stream + PDB_MODULE_SYMBOLS_OFFS;
    const gdb_byte *syms_end = module_stream + sym_byte_size;
    int depth = 0;

    while (data + PDB_RECORD_HDR_SIZE <= syms_end)
      {
	auto hdr = pdb_parse_sym_record_hdr (data, syms_end);
	if (!hdr)
	  break;

	const gdb_byte *rec_end = data + hdr->rec_size ();
	if (depth == 0 && (procs || !pdb_is_proc_record (hdr->type)))
	  {
	    const char *name
	      = pdb_indexed_record_name (hdr->type,
					 data + PDB_RECORD_DATA_OFFS, rec_end);
	    if (name != nullptr && *name != '\0')
	      emit (name, hdr->type);
	  }

	/* The scopes pdb_parse_symbols opens and closes.  */
	switch (hdr->type)
	  {
	  case S_GPROC32:
	  case S_LPROC32:
	  case S_GPROC32_ID:
	  case S_LPROC32_ID:
	  case S_BLOCK32:
	  case S_THUNK32:
	  case S_INLINESITE:
	  case S_INLINESITE2:
	    depth++;
	    break;
	  case S_END:
	  case S_INLINESITE_END:
	  case S_PROC_ID_END:
	    if (depth > 0)
	      depth--;
	    break;
	  default:
	    break;
	  }

	data = rec_end;
      }
  }

  /* Index inlinee names without expanding the module or resolving types.  */

  void scan_module_inlinees (uint16_t mod_idx, pdb_index_shard *shard,
			    gdb::byte_vector &scratch)
  {
    const pdb_module_info &mod = m_pdb->modules[mod_idx];
    uint64_t offset = (uint64_t) mod.sym_byte_size + mod.c11_byte_size;
    uint64_t stream_size = m_pdb->streams[mod.stream_number].size;
    if (offset > stream_size || mod.c13_byte_size > stream_size - offset
	|| mod.c13_byte_size < CV_SIGNATURE_SIZE)
      return;

    uint64_t end = offset + mod.c13_byte_size;
    const gdb_byte *header = pdb_stream_bytes (m_pdb, mod.stream_number,
					      offset, CV_SIGNATURE_SIZE,
					      scratch);
    if (read_cv_signature (header) == CV_SIGNATURE_C13)
      offset += CV_SIGNATURE_SIZE;

    while (end - offset >= C13_SUBSECT_HEADER_SIZE)
      {
	header = pdb_stream_bytes (m_pdb, mod.stream_number, offset,
				   C13_SUBSECT_HEADER_SIZE, scratch);
	uint32_t kind = read_u32 (header);
	uint32_t size = read_u32 (header + 4);
	offset += C13_SUBSECT_HEADER_SIZE;
	if (size > end - offset)
	  return;

	if (kind == DEBUG_S_INLINEELINES && size >= 4)
	  {
	    const gdb_byte *data = pdb_stream_bytes (m_pdb, mod.stream_number,
						     offset, size, scratch);
	    const gdb_byte *data_end = data + size;
	    uint32_t signature = read_u32 (data);
	    data += 4;
	    if (signature != CV_INLINEE_SOURCE_LINE_SIGNATURE
		&& signature != CV_INLINEE_SOURCE_LINE_SIGNATURE_EX)
	      return;

	    while (data_end - data >= 12)
	      {
		uint32_t inlinee = read_u32 (data);
		data += 12;
		if (signature == CV_INLINEE_SOURCE_LINE_SIGNATURE_EX)
		  {
		    if (data_end - data < 4)
		      return;
		    uint32_t extra_files = read_u32 (data);
		    data += 4;
		    if (extra_files > (data_end - data) / 4)
		      return;
		    data += (size_t) extra_files * 4;
		  }

		std::string name;
		if (pdb_ipi_lookup_inlinee (m_pdb, inlinee, &name, nullptr)
		    && !name.empty () && name.size () <= UINT16_MAX)
		  shard->add (obstack_strdup (&shard->storage, name.c_str ()),
			      name.size (), mod_idx, S_INLINESITE);
	      }
	  }

	uint64_t next = (offset + size + 3) & ~uint64_t (3);
	if (next > end)
	  return;
	offset = next;
      }
  }

  /* Scan module MOD_IDX and add its indexed names to SHARD, procedures
     only when PROCS.  SCRATCH is a
     reusable byte buffer that pdb_stream_bytes fills with the module's
     symbol bytes, reused across the modules the worker owns.  Each name is
     copied onto SHARD's obstack since it points into SCRATCH, which the
     next module overwrites.  */
  void scan_module (uint16_t mod_idx, pdb_index_shard *shard,
		    gdb::byte_vector &scratch, bool procs)
  {
    pdb_module_info *mod = &m_pdb->modules[mod_idx];
    if (mod->stream_number == 0xFFFF
	|| mod->stream_number >= m_pdb->streams.size ()
	|| mod->sym_byte_size < PDB_MODULE_SYMBOLS_OFFS
	|| mod->sym_byte_size > m_pdb->streams[mod->stream_number].size)
      return;

    scan_module_inlinees (mod_idx, shard, scratch);

    const gdb_byte *stream = pdb_stream_bytes (m_pdb, mod->stream_number, 0,
					       mod->sym_byte_size, scratch);
    walk_module_syms (stream, mod->sym_byte_size, procs,
      [&] (const char *name, uint16_t rectype)
      {
	size_t name_len = strlen (name);
	if (name_len > UINT16_MAX)
	  return;

	shard->add (obstack_strdup (&shard->storage, name), name_len, mod_idx,
		    rectype);
      });
  }

  /* One parallel module-scan worker.  */

  struct pdb_scan_worker
  {
    pdb_scan_worker (pdb_cooked_index *idx, bool procs)
      : m_idx (idx), m_procs (procs)
    {
    }

    DISABLE_COPY_AND_ASSIGN (pdb_scan_worker);

    ~pdb_scan_worker ()
    {
      m_idx->collect_shard (std::move (m_shard));
    }

    void operator() (::iterator_range<const uint16_t *> batch)
    {
      for (uint16_t mod_idx : batch)
	m_shard->catch_error ([&] ()
	  {
	    m_idx->scan_module (mod_idx, m_shard.get (), m_scratch, m_procs);
	  });
    }

    pdb_cooked_index *m_idx;
    bool m_procs;
    std::unique_ptr<pdb_index_shard> m_shard { new pdb_index_shard };
    gdb::byte_vector m_scratch;
  };

  void set_finalized ()
  {
    std::lock_guard<std::mutex> guard (m_mutex);
    m_state = pdb_index_state::FINALIZED;
    m_cond.notify_all ();
  }

  pdb_per_objfile *m_pdb;

  /* Storage for the PC->module addrmap.  Written only by the worker
     (build_pc_addrmap), after the parallel build has joined.  */
  auto_obstack m_storage;

  /* The cooked index, split into per-worker shards.  Each shard owns its
     entries (sorted by name) and, for the module-scan path, the obstack
     that backs their copied names.  Built by the parallel workers and
     appended under m_shards_mutex; immutable and searched by every
     for_each_named after FINALIZED.  */
  std::vector<std::unique_ptr<pdb_index_shard>> m_shards;
  std::mutex m_shards_mutex;

  /* PC -> module map (see build_pc_addrmap), built by the worker on
     m_storage; nullptr when no module contributes code.  Read on the main
     thread after FINALIZED.  */
  addrmap_fixed *m_pc_map = nullptr;

  pdb_index_state m_state = pdb_index_state::INITIAL;
  std::mutex m_mutex;
  std::condition_variable m_cond;

  /* The worker's future, valid only when the build was posted to the
     pool (m_posted).  Joined in the destructor.  */
  bool m_posted = false;
  gdb::future<void> m_future;

  /* Set once the deferred build log has been spilled on the main thread
     (auto-printed on the first wait that sees FINALIZED).  */
  bool m_log_printed = false;

  /* The exception that ended the build, as cooked_index_worker::m_failed.
     Written by the worker before FINALIZED; reported once by the first
     main-thread wait.  */
  std::optional<gdb_exception> m_failed;

  /* Whether the first main-thread wait has reported the failure or the
     shards' exceptions.  */
  bool m_reported = false;

  /* Build wall time (ms) and source ("GSI" / "module scan" / "forced
     module scan"), set by the worker before FINALIZED, read on the main
     thread for the finalize print.  */
  double m_build_ms = 0;
  /* Per-phase build times (ms): the parallel per-shard build (seed + the
     per-shard sort folded in), and the PC->module addrmap.  */
  double m_seed_ms = 0;
  double m_addrmap_ms = 0;
  const char *m_source = "";
};

/* Build the tagged types LOOKUP names on demand, so a by-name type lookup
   finds them without resolving the whole TPI.  Completion builds every
   match; otherwise only the exact name is built.  Resolving a type is
   main-thread work.  A no-op once every tagged type has been built.  */

static void
pdb_lazy_type_by_name (pdb_per_objfile *pdb, const lookup_name_info &lookup)
{
  if (pdb->all_tagged_types_built)
    return;
  if (lookup.completion_mode ())
    pdb_build_tagged_types_matching (pdb, lookup);
  else
    pdb_build_tagged_type (pdb, lookup.name ());
}

/* Build every remaining tagged type at once — for a query that matches
   every symbol ("info types") or expand-all.  Runs the full sweep only
   once.  */

static void
pdb_build_all_types (pdb_per_objfile *pdb)
{
  if (pdb->all_tagged_types_built)
    return;
  pdb->all_tagged_types_built = true;
  pdb_register_tpi_typedefs (pdb);
}

/* If FILE_MATCHER is non-null, set MODULES_TO_SKIP for each module none of
   whose source files FILE_MATCHER accepts, as dw_search_file_matcher does
   for DWARF units.  */

static void
pdb_search_file_matcher (pdb_per_objfile *pdb,
			 std::vector<bool> &modules_to_skip,
			 search_symtabs_file_matcher file_matcher)
{
  if (file_matcher == nullptr)
    return;

  for (size_t m = 0; m < pdb->modules.size (); m++)
    {
      QUIT;

      pdb_module_info *mod = &pdb->modules[m];
      pdb_read_module_files (pdb, mod);

      bool matched = false;
      for (uint16_t j = 0; !matched && j < mod->num_files; j++)
	{
	  const char *file = pdb_module_file_name (pdb, mod, j);
	  if (file == nullptr)
	    continue;

	  std::string name = pdb_convert_path (file);
	  if (file_matcher (name.c_str (), false))
	    {
	      matched = true;
	      break;
	    }

	  /* Before we invoke realpath, which can get expensive when many
	     files are involved, do a quick comparison of the basenames.  */
	  if (!basenames_may_differ
	      && !file_matcher (lbasename (name.c_str ()), true))
	    continue;

	  matched = file_matcher (gdb_realpath (name.c_str ()).get (), false);
	}

      if (!matched)
	modules_to_skip[m] = true;
    }
}

struct pdb_cooked_index_functions : public quick_symbol_functions
{
  bool has_symbols (objfile *objfile) override
  {
    pdb_qf_printf ("LAZY has_symbols called");
    return true;
  }

  bool has_unexpanded_symtabs (objfile *objfile) override
  {
    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);
    if (pdb == nullptr)
      return false;

    for (const auto &module : pdb->modules)
      if (!module.expanded)
	return true;
    return false;
  }

  symtab *find_last_source_symtab (objfile *objfile) override
  {
    pdb_qf_printf ("find_last_source_symtab called");
    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);
    if (pdb == nullptr)
      return nullptr;

    /* Mirror DWARF (dwarf2_base_index_functions): expand the last unit
       and return its primary filetab, skipping modules with no code.  */
    for (auto it = pdb->modules.rbegin (); it != pdb->modules.rend (); ++it)
      {
	compunit_symtab *cu = pdb_build_module (pdb, &*it);
	if (cu != nullptr)
	  return cu->primary_filetab ();
      }
    return nullptr;
  }

  void forget_cached_source_info (objfile *objfile) override
  {
  }

  enum language lookup_global_symbol_language (objfile *objfile,
					       const char *name,
					       domain_search_flags domain,
					       bool *symbol_found_p) override
  {
    pdb_qf_printf ("lookup_global_symbol_language called for '%s'", name);
    *symbol_found_p = false;

    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);
    if (pdb == nullptr || pdb->cooked_index == nullptr)
      return language_unknown;

    /* The index holds functions (from the GSI); resolve the language
       from the owning module's S_COMPILE record.  Globals and types are
       matched by the core in <pdb-globals> / <pdb-types> directly.  */
    enum language lang = language_unknown;
    lookup_name_info lookup (name, symbol_name_match_type::FULL);
    pdb->cooked_index->for_each_named (lookup, nullptr,
      [&] (const pdb_cooked_index_entry &e)
      {
	if (!e.matches (domain))
	  return false;
	if (e.module_index >= pdb->modules.size ())
	  return false;

	*symbol_found_p = true;
	lang = pdb_module_language (pdb, &pdb->modules[e.module_index]);
	return true;
      });
    return lang;
  }

  void print_stats (objfile *objfile, bool print_bcache) override
  {
    /* Called once per objfile for each of the two stats blocks; the
       byte-cache pass has nothing of ours to report.  */
    if (print_bcache)
      return;

    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);
    if (pdb == nullptr)
      return;

    size_t expanded = 0;
    for (const auto &mod : pdb->modules)
      if (mod.expanded)
	expanded++;

    gdb_printf (_("  Number of read modules: %zu\n"), expanded);
    gdb_printf (_("  Number of unread modules: %zu\n"),
		pdb->modules.size () - expanded);
    if (pdb->cooked_index != nullptr)
      gdb_printf (_("  Number of cooked index entries: %zu\n"),
		  pdb->cooked_index->num_entries ());
  }

  void dump (objfile *objfile) override
  {
    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);
    if (pdb == nullptr || pdb->cooked_index == nullptr)
      return;
    pdb->cooked_index->dump (objfile->arch ());
  }

  void expand_all_symtabs (objfile *objfile) override
  {
    pdb_qf_printf ("expand_all_symtabs called");
    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);
    if (pdb == nullptr)
      return;
    pdb_build_all_types (pdb);
    if (pdb_lazy_globals)
      pdb_build_all_globals (pdb);
    if (pdb->cooked_index != nullptr)
      pdb->cooked_index->wait ();
    for (auto &module : pdb->modules)
      pdb_build_module (pdb, &module);
  }

  iteration_status
  search (objfile *objfile, search_symtabs_file_matcher file_matcher,
	  const lookup_name_info *lookup_name,
	  search_symtabs_symbol_matcher symbol_matcher,
	  compunit_symtab_iteration_callback compunit_callback,
	  block_search_flags search_flags, domain_search_flags domain,
	  search_symtabs_lang_matcher lang_matcher = nullptr) override
  {
    (void) search_flags;
    (void) lang_matcher;

    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);
    if (pdb == nullptr)
      return iteration_status::keep_going;

    /* Search the currently-instantiated compunits (the eager
       <pdb-types> / <pdb-globals> and any already-expanded modules),
       applying FILE_MATCHER.  */
    auto search_compunits = [&] () -> iteration_status
      {
	if (compunit_callback == nullptr)
	  return iteration_status::keep_going;

	for (compunit_symtab &cu : objfile->compunits ())
	  {
	    if (file_matcher != nullptr)
	      {
		bool match = false;
		for (const symtab *s : cu.filetabs ())
		  if (file_matcher (s->filename (), false)
		      || file_matcher (lbasename (s->filename ()), true))
		    {
		      match = true;
		      break;
		    }
		if (!match)
		  continue;
	      }

	    if (compunit_callback (&cu) == iteration_status::stop)
	      return iteration_status::stop;
	  }
	return iteration_status::keep_going;
      };

    std::vector<bool> modules_to_skip (pdb->modules.size ());
    pdb_search_file_matcher (pdb, modules_to_skip, file_matcher);

    /* "info types" / "info functions" / "info variables" pass
       lookup_name_info::match_any (), an empty name in completion mode, and
       carry their regexp in SYMBOL_MATCHER.  Both forms mean "every symbol".  */
    const bool completing
      = lookup_name != nullptr && lookup_name->completion_mode ();
    const bool match_all
      = lookup_name == nullptr || (completing && lookup_name->name ().empty ());

    if (match_all)
      {
	/* <pdb-types> and <pdb-globals> name no source file.  */
	if (file_matcher == nullptr)
	  {
	    if ((domain & (SEARCH_TYPE_DOMAIN | SEARCH_STRUCT_DOMAIN)) != 0)
	      pdb_build_all_types (pdb);
	    if (pdb_lazy_globals)
	      pdb_build_all_globals (pdb);
	  }
	for (size_t m = 0; m < pdb->modules.size (); m++)
	  if (!modules_to_skip[m])
	    pdb_build_module (pdb, &pdb->modules[m]);
	return search_compunits ();
      }

    /* Type-name lookup (ptype Tag, a cast, whatis Tag, ...).  The
       name->TI map (tagged_type_names) is built at load, but the tag
       is not resolved there.  Resolve this one tag now and add its
       symbol to <pdb-types>, so a session that never ptypes a struct
       never builds one.  Functions and globals do not come through
       here: their symbols carry their own types from the module or
       <pdb-globals>.  */
    if ((domain & (SEARCH_TYPE_DOMAIN | SEARCH_STRUCT_DOMAIN)) != 0)
      pdb_lazy_type_by_name (pdb, *lookup_name);

    /* An enumerator is reachable only through the enum that declares it,
       and it answers a variable-domain query.  */
    if ((domain & SEARCH_VAR_DOMAIN) != 0 && !pdb->all_tagged_types_built)
      pdb_build_enum_for_enumerator (pdb, *lookup_name);

    if (pdb_lazy_globals)
      pdb_build_global (pdb, lookup_name->name ());

    /* A tagged type or global already lives in an eager compunit, and
       a symbol in an already-expanded module needs no work.  Search
       those first; only on a miss consult the cooked index and expand
       the owning module(s).  */
    if (search_compunits () == iteration_status::stop)
      return iteration_status::stop;

    pdb_cooked_index *index = pdb->cooked_index.get ();
    if (index != nullptr)
      {
	std::string_view name = lookup_name->name ();
	index->for_each_named (*lookup_name, symbol_matcher,
	  [&] (const pdb_cooked_index_entry &e)
	  {
	    if (!e.matches (domain))
	      return false;
	    if (e.module_index >= pdb->modules.size ()
		|| modules_to_skip[e.module_index])
	      return false;

	    pdb_module_info *mod = &pdb->modules[e.module_index];
	    pdb_qf_printf ("'%s' [domain=%s] -> mod %u %s%s",
			   std::string (name).c_str (),
			   domain_name (domain).c_str (),
			   e.module_index,
			   mod->module_name != nullptr
			     ? lbasename (mod->module_name) : "?",
			   mod->expanded ? " [already expanded]"
					 : " [expanding]");
	    pdb_build_module (pdb, mod);
	    return false;
	  });
      }

    return search_compunits ();
  }

  compunit_symtab *find_pc_sect_compunit_symtab (objfile *objfile,
						 bound_minimal_symbol msymbol,
						 CORE_ADDR pc,
						 obj_section *section,
						 int warn_if_readin) override
  {
    (void) msymbol;
    (void) section;
    (void) warn_if_readin;
    pdb_qf_printf ("find_pc_sect called for %s",
		   paddress (objfile->arch (), pc));

    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);
    if (pdb == nullptr)
      return nullptr;

    /* Resolve PC to its module with the cooked index's O(log n) address
       map, then expand that module (a no-op if already expanded) — the
       DWARF cooked-index model (index_table->lookup + instantiate).  */
    if (pdb->cooked_index != nullptr)
      {
	pdb_module_info *mod = pdb->cooked_index->find_pc_module (pc);
	if (mod != nullptr)
	  {
	    pdb_qf_printf ("find_pc_sect: contribution match, %s",
			   mod->expanded ? "already expanded" : "expanding");
	    pdb_build_module (pdb, mod);

	    /* Common case: the resolved module's global block covers PC,
	       so return it directly without scanning the others.  */
	    if (mod->cu != nullptr)
	      {
		const blockvector *bv = mod->cu->blockvector ();
		const block *b = bv != nullptr ? bv->global_block () : nullptr;
		if (b != nullptr && b->start () <= pc && pc < b->end ())
		  return mod->cu;
	      }
	  }
      }

    /* Fallback: without a Section Contribution substream the address map
       has only one contribution per module and misses a function placed
       in a section of its own.  Consult the finalized block ranges of the
       expanded modules.  */
    for (auto &module : pdb->modules)
      {
	if (module.cu == nullptr)
	  continue;

	const blockvector *bv = module.cu->blockvector ();
	if (bv == nullptr)
	  continue;

	const block *b = bv->global_block ();
	if (b != nullptr && b->start () <= pc && pc < b->end ())
	  return module.cu;
      }

    pdb_qf_printf ("find_pc_sect: no match");
    return nullptr;
  }

  symbol *find_symbol_by_address (objfile *objfile, CORE_ADDR address) override
  {
    pdb_qf_printf ("find_symbol_by_address called for %s",
		   paddress (objfile->arch (), address));

    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);
    if (pdb == nullptr || pdb->cooked_index == nullptr)
      return nullptr;

    /* Fast index, no scan: the cooked-index address map resolves ADDRESS
       to its module; expand it and let the compunit find the exact-match
       symbol — the DWARF model (index_table->lookup + symbol_at_address).  */
    pdb_module_info *mod = pdb->cooked_index->find_pc_module (address);
    if (mod == nullptr)
      return nullptr;

    compunit_symtab *cu = pdb_build_module (pdb, mod);
    if (cu == nullptr)
      return nullptr;
    return cu->symbol_at_address (address);
  }

  void map_symbol_filenames (objfile *objfile, symbol_filename_listener fun,
			     bool need_fullname) override
  {
    pdb_qf_printf ("map_symbol_filenames called");
    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);
    if (pdb == nullptr)
      return;

    /* Get the names from DBI file-info substream.  */
    for (auto &mod : pdb->modules)
      {
	if (mod.expanded)
	  continue;

	pdb_read_module_files (pdb, &mod);
	if (mod.files == nullptr)
	  continue;
	for (uint16_t i = 0; i < mod.num_files; i++)
	  {
	    const char *name = mod.files[i];
	    if (name == nullptr)
	      continue;

	    fun (name, need_fullname ? name : nullptr);
	  }
      }
  }

  /* Resolving the main name here keeps find_main_name from falling back to
     a by-name lookup at an arbitrary later point, which would re-enter the
     reader mid-parse.  */

  void compute_main_name (objfile *objfile) override
  {
    pdb_per_objfile *pdb = get_pdb_per_objfile (objfile);

    if (pdb == nullptr || pdb->cooked_index == nullptr)
      return;

    if (objfile->per_bfd->name_of_main != nullptr)
      return;

    bool found = false;
    enum language lang = language_unknown;
    lookup_name_info lookup ("main", symbol_name_match_type::FULL);
    pdb->cooked_index->for_each_named (lookup, nullptr,
      [&] (const pdb_cooked_index_entry &e)
      {
	if (!e.matches (SEARCH_FUNCTION_DOMAIN))
	  return false;
	if (e.module_index >= pdb->modules.size ())
	  return false;

	found = true;
	lang = pdb_module_language (pdb, &pdb->modules[e.module_index]);
	return true;
      });

    if (found)
      set_objfile_main_name (objfile, "main", lang);
  }
};

/* See pdb-internal.h.  Build the eager globals CU, start the background
   cooked-index build, and install the lazy quick_symbol_functions.
   <pdb-types> is built lazily on the first type-domain query.  */

void
pdb_install_cooked_index (objfile *objfile, pdb_per_objfile *pdb)
{
  /* Globals stay eager: the core searches <pdb-globals> directly, and it
     also holds global data that variable lookups need.  Tagged types
     (<pdb-types>) are deferred to pdb_build_types_cu on first use, to
     keep resolving every tagged type off the load path.  Build globals
     before starting the worker so the main thread is the only writer to
     the objfile obstack.  */
  if (pdb_lazy_globals)
    {
      pdb_build_lazy_globals_index (pdb);
      pdb_register_global_namespaces (pdb);
    }
  else
    pdb_load_global_syms_cu (pdb);

  pdb_register_tpi_namespaces (pdb);

  pdb->cooked_index.reset (new pdb_cooked_index (pdb));
  pdb->cooked_index->start ();

  objfile->qf.emplace_front (new pdb_cooked_index_functions);
}

/* Constructor and destructor defined here, where pdb_cooked_index is a
   complete type, so the unique_ptr member's destructor can be
   instantiated (the header only forward-declares pdb_cooked_index).  */

pdb_per_objfile::pdb_per_objfile (::objfile *objfile)
  : objfile (objfile)
{
}

pdb_per_objfile::~pdb_per_objfile ()
{
  /* Join the index worker while the streams, modules and sections it
     reads still exist; members declared after cooked_index are destroyed
     before it otherwise.  */
  cooked_index.reset ();
}

} /* namespace pdb */

INIT_GDB_FILE (pdb_index)
{
  using namespace pdb;

  add_setshow_boolean_cmd ("pdb-force-module-index", class_maintenance,
			   &pdb_force_module_index, _("\
Set forced module scanning for the PDB cooked index."), _("\
Show forced module scanning for the PDB cooked index."), _("\
When on, the cooked index is built by scanning every module even if the\n\
PDB ships a GSI (which would otherwise build the index directly) -- for\n\
measuring the module-scan build path.  Off by default.  Takes effect on\n\
the next objfile load."),
			   nullptr, nullptr, &maintenance_set_cmdlist,
			   &maintenance_show_cmdlist);

  add_setshow_boolean_cmd ("pdb-lazy-globals", class_maintenance,
			   &pdb_lazy_globals, _("\
Set lazy globals for the PDB reader (prototype)."), _("\
Show lazy globals for the PDB reader (prototype)."), _("\
When on, <pdb-globals> is not built eagerly at load; globals are indexed\n\
and each is built on the by-name lookup that asks for it -- for\n\
measuring eager-vs-lazy load time.  Off by default.  Takes effect on the\n\
next objfile load."),
			   nullptr, nullptr, &maintenance_set_cmdlist,
			   &maintenance_show_cmdlist);

  add_setshow_boolean_cmd ("pdb-synchronous", class_maintenance,
			   &pdb_synchronous, _("\
Set synchronous building of the PDB cooked index."), _("\
Show synchronous building of the PDB cooked index."), _("\
When on, the cooked index is built on the thread that loads the objfile\n\
instead of a background worker, so a query can never observe a partial\n\
build.  Off by default.  Takes effect on the next objfile load."),
			   nullptr, nullptr, &maintenance_set_cmdlist,
			   &maintenance_show_cmdlist);
}
