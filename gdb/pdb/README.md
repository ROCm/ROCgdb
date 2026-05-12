# PDB Reader for GDB — Architecture & Data Flow

## Overview

PDB is a multi-stream file container where different streams provide different
debug information. Streams are composed of multiple blocks, which don't have to
be consecutive. The blocks are the actual physical parts of the file — the PDB
file itself consists of multiple fixed-size blocks (except for the header).

Following is the data from PDB we need to read on initialization.

### MSF header (SuperBlock)

The MSF SuperBlock is the first block in the PDB file and contains basic
information such as the block size, number of blocks, and most importantly,
the location of the stream directory, which is used to locate all other
streams in the file. The SuperBlock is 64 bytes long.

### Stream directory

The stream directory is located immediately after the SuperBlock and specifies
which block belongs to which stream. Each stream can span multiple physical
blocks that are not necessarily contiguous.

With the information from the stream directory, we are able to parse any stream.

### PDB Info stream (stream 1)

Basic information stream - its most significant part is the location of the
"/names" stream (the String Table) which contains the list of all the files
compiled into the PDB.

### Names stream (String Table)

Contains info on all the files used by all modules compiled into the PDB.
The names are read into the String Table, loaded eagerly because almost any
operation (break, info sources ...) references a module's files to get line
information, so preloading avoids repeated lookups.

### DBI stream (stream 3)

DBI stream contains the debug information (line numbers, symbols, etc.) for all
the modules (object files) linked into the program. Each module's debug info is
in a different stream and we read those streams on request. Eagerly we only load
the header which contains info on per module streams (debug info is per module).

### DBI File Info substream

Substream is just a piece of data located at a given offset in a stream.
The File Info substream contains info on all the files used by all the modules
compiled into the PDB - the Names Buffer. Names Buffer actually duplicates the
String Table but it also adds the information on files that go into each module.
This is suitable for Quick Functions that check if a file is in a module;
obtaining this info from the String Table would require expanding the parts
of the module stream, to get the sections that reference per module files
(indices into String Buffer).

### TPI stream (stream 2)

The TPI (Type Program Information) stream contains all non-builtin type
records used by the program — pointers, modifiers, arrays, procedures, member
functions, structs, classes, unions, enums, bitfields, argument lists, etc.

A **type index** is a 32-bit integer that uniquely identifies a type. Indices
below 0x1000 are reserved for simple/builtin types (encoded within the index).
Indices 0x1000 and above correspond to records in the TPI stream, assigned
sequentially: the first record is 0x1000, the second 0x1001, etc. Symbol records
and other type records reference types by their type index.

Each record in the stream has variable length consisting of a 2-byte `RecordLen`
, a 2-byte `RecordKind` (the "leaf type" identifier such as `LF_POINTER`,
`LF_MODIFIER`, `LF_ARRAY`, `LF_PROCEDURE`, `LF_ARGLIST`...), and a payload whose
layout depends on the leaf type. Fields within a record can reference other
types by their type index, forming a directed graph (e.g. an `LF_POINTER` record
contains the type index of the pointee type).

Reading the stream at load time only *indexes* it; no GDB type is created then.
Four structures come out of that, and each exists because CodeView leaves out
something DWARF gives a reader for free:

- **A table of records, addressed by type index.**  Records are variable length
  and refer to each other by number, so nothing can jump to type 0x1234 without
  having already walked everything before it.  Walking the stream once and
  remembering where each record starts turns every later reference into an
  array lookup.  DWARF needs no equivalent, because a reference to another DIE
  is already a byte offset.

- **A map from a type's name to its index.**  Used to find a type by it's name
  (CodeView doesn't store this).  It's similar to DWARF's `debug_names`.

- **A map from a nested type back to the class that declares it.**.
  Nested types hirearchy must be extracted from the records for later use.

- **A cache of the GDB types already built.**  Filled as types are requested,
  so an index is converted at most once.

A record becomes a GDB type only when something asks for it -- printing a
variable, running `ptype`, or setting a breakpoint on a method.



### IPI stream (stream 4)

The IPI (Id Program Information) stream layout is identical to the TPI but it
is used to described various symbols by their name.  E.g. which function this
is, which source file it came from, which compiler invocation produced it.

While linker can merge same  types from various object files into one TPI type,
the names would defeat that comparison, so everything that is a name rather than
a shape was moved out into a second stream.

What it holds that matters to a debugger:

- **A function's identity** -- its name plus the type index of its signature.
  Two forms exist: one for a free function, which also records the namespace it
  sits in, and one for a member function, which records its class instead.
- **Strings**, such as those namespace names, sometimes stored as a list of
  smaller pieces to be joined rather than as one string.
- **Build information** -- compiler executable, working directory, source file,
  output PDB, and the full command line.
- **Where a type was declared** -- the source file and line of a class or
  struct definition.

The reader needs the IPI because of inlining.  When a call is inlined its code
belongs to the caller, so no procedure record exists to name the callee; the
inline site stores an id number instead, and only the IPI can turn that number
into `ns::Klass::method` together with the signature needed to describe its
parameters.  DWARF requires no such lookup: an inlined instance points directly
at the abstract description of the function it came from, which already carries
the name.

An inlined function is also the one kind of function that can be looked up by
name before its caller has ever been read, so these names are additionally fed
into the symbol index described under *Lazy Loading*.

### Module Streams

Module streams contain the debug information for individual modules (object
files). Various debug sections are specified using identifiers — e.g. symbols
or line information or file info. The line information is in C13 sections
(C11 sections are obsolete). C13 sections are split into subsections, most
importantly Checksums and Lines. The Checksums subsection references the
String Table to provide the source files that belong to the module, while the
Lines subsection maps addresses to source lines (analogous to `.debug_line` in
DWARF).

### Symbol Record Stream / GSI / PSGSI

The Symbol Record Stream (referenced by the DBI header) contains all global
symbol records — procedures and global data, the file-static data that the
publics never mention, the cross-reference records the GSI indexes, and the
public symbols themselves.

The PSGSI (Public Symbol Index) stream is PDB's equivalent of the ELF symbol
table (`.symtab`/`.dynsym`) — it contains a hash table whose hash records point
into the Symbol Record Stream to locate the public symbols stored there.
The stream also carries an address-sorted map of the same publics.  The reader
ignores that map and iterates the hash records instead, because building
minimal symbols needs to visit every public exactly once and does not care
about their order.

#### How the minimal symbol table is built

Minimal symbols are the address-to-name table GDB falls back on when it has no
full symbol — `info symbol`, `x/` and the names annotating disassembly all come
from here.  Building it needs both streams above, and takes three passes.

First the reader walks the symbol record stream and notes, for every global and
file-static *data* object, its plain source-level name and the section it lives
in, keyed by address.

Then it walks the publics hash table.  Each public supplies an address, a
decorated name, and a flag saying whether it is code or data, which decides
whether the minimal symbol is a text or a data one.  Two things happen per
entry:

- The decorated name is filed under that address.  It is the only place a
  linkage name exists in the file, and other parts of the reader attach it to a
  full symbol later.
- For a data address, the plain name from the first pass is preferred over the
  decorated one — with no MSVC demangler, the decorated form is what the user
  would otherwise be shown.

Finally, every data object seen in the first pass that no public covered is
recorded on its own, as file-local data, or file-local bss when its section
says so.  This last pass is what makes file statics visible at all: publics
describe only what the linker exported, so a `static` object has none, and
without this its address resolves to whichever exported symbol happens to
precede it.  `&int8_search_buf` in `gdb.base/find.c` printed as
`<??_R0?AVpairNode@@@8+32>`, an unrelated MSVC RTTI descriptor, and the same
error reached `info symbol`, `x/` and every disassembly annotation near a
static.

If the publics index stream is absent, no minimal symbols are built.

The GSI (Global Symbol Index) stream is a hash table for O(1) name to symbol
lookup similar to DWARF's .debug_names. It indexes cross-reference records
(S_PROCREF, S_LPROCREF, S_DATAREF) that point into module streams — each
reference carries module index and offset, telling the reader which module
contains the full symbol definition. We use this table to build the cooked index
and provide quick functions for symbol lookup on GDB's request.

## PDB Discovery

PDB files are searched at different locations - the PDB name recorded in the
RSDS record of the Debug Directory section in the executable is used as the
base name. We look next to the EXE, then in each entry of GDB's
`debug-file-directory` (`set debug-file-directory`), then try the full RSDS
embedded path. We also search for the PDB by simply replacing the EXE
extension with `.pdb`.

Windows can specify the location of the PDB files in Windows registry or in the
environment variables.

TODO: For system DLLs, Windows normally uses so called Debug Symbol server from
where the PDB files can be downloaded.

### Path Conversion (MSYS2)

PDBs produced under MSYS2 can have Linux style paths which are converted into
Windows style paths before storing them to symtab linetables, so that GDB
can load them. This either requires prepending the MSYS2 root
(e.g. /home/PATH -> C:/msys2/PATH) or converting drive information
(e.g. /c/PATH -> C:/PATH).

The MSYS2 root must be specified using MSYS2_ROOT env. var, otherwise we look
into common msys2/mingw64 directories.


## Maintenance Commands

All commands accept optional `path=<pdb-path>` and `modi=N` arguments to
select a specific PDB / module. If omitted, the default (main program) PDB and
all modules are used.

| Command                            | What it shows                                             |
|------------------------------------|-----------------------------------------------------------|
| `maintenance info pdb-loaded-files`| Paths of all currently loaded PDB files                   |
| `maintenance info pdb-modules`     | Modules with stream numbers and file counts               |
| `maintenance info pdb-files`       | Source files per module (DBI File Info substream)         |
| `maintenance info pdb-files-c13`   | Source files per module (C13 checksums), with checksum    |
|                                    | type (MD5/SHA-1/SHA-256) and hash values                  |
| `maintenance info pdb-lines`       | C13 line info: section:offset ranges and line mappings    |
| `maintenance info pdb-symbols`     | Raw CodeView symbol records from module streams           |
| `maintenance info pdb-sym-records` | Records from the global Symbol Record Stream              |
| `maintenance info pdb-gsi`         | GSI (Global Symbol Index) hash table dump                 |
| `maintenance info pdb-psi`         | PSGSI (Public Symbol Index) hash table dump, address map  |
| `maintenance info pdb-locations`   | Variable location batons (ranges, register/offset, gaps); |
|                                    | requires `modi=N`, optional `symbol=NAME` filter          |
| `maintenance info pdb-types`       | TPI type records from the PDB                             |

## GDB Integration and Initialization Order

`coff_symfile_read()` calls `pdb_initialize_objfile()` in preference to the
DWARF reader; DWARF runs only if no PDB is found.  When PDB support is not
compiled in, a stub returns false.

Initialization then loads the PDB in this order:

1. Validate the MSF header and read the stream directory (which maps every
   stream to its blocks).
2. Read the PDB Info stream, validate its GUID/age against the EXE's RSDS
   record, and load the `/names` string table.  A mismatch aborts the load:
   symbol offsets, line tables and type indices are all keyed to the exact
   build that produced the EXE, so a PDB from another build cannot be used
   at all.  Aborting also discards the `pdb_per_objfile` that steps 1-2 had
   already attached to the objfile, so the refused file is not left behind
   for `maint info pdb-loaded-files` and the other maintenance commands to
   report as loaded.
3. Parse the DBI stream: module headers and the stream indices for the symbol,
   GSI and PSGSI streams.
4. Index the TPI (types) and IPI (ids) streams.  Records are indexed but not
   resolved; a type is built the first time a symbol references it.  The IPI
   supplies the names of inlined callees.
5. Read PE section addresses from BFD, used to relocate section:offset pairs
   into real addresses.
6. Build minimal symbols from the public symbol stream.
7. Install the lazy cooked index (`pdb_install_cooked_index()`): global data
   and constants are loaded into `<pdb-globals>`, and a background worker
   indexes names and addresses.  A module becomes a `compunit_symtab` only
   when a query needs it.  With `--readnow`, `pdb_expand_all_modules()`
   expands every module at load instead.

## Namespace inference

C++ scopes have to be reconstructed, because a namespace is never recorded as
anything; it survives only as a prefix inside a name like `ns_a::ns_b::foo`,
while GDB needs real namespace symbols for lookup to work.  Class scopes are
better off: a class does list the types declared inside it, and a nested type
is marked as nested, so those are read rather than guessed.

For each symbol the reader asks its type first, and falls back to the name only
when the type cannot answer:

- If the type is a tag carrying the symbol's own name, that tag already says
  what encloses it -- a function body, a class, or nothing but namespaces --
  and when it is a class, which one.  Nothing is guessed.
- If the type is a member function, the record names the owning class, and the
  class answers in its place.
- Otherwise -- an alias, a variable, a constant -- the type describes the thing
  rather than its scope, so only the name is left.  Each `::`-separated prefix
  is looked up by name: one that matches a known type is a class and is
  skipped, anything else becomes a namespace.

Splitting a name stops at the first character that cannot appear in an
identifier, such as `<` or `(`.  Everything after it is a template argument
list, a parameter list or a block inside a function, whose `::` separate
something other than namespaces.  Names that start part-way inside a template
argument list are discarded for the same reason -- they are fragments, not
whole C++ names.

Incomplete declarations count as type names in that lookup, so a class that is
only declared is never mistaken for a namespace, and such a declaration never
displaces the real definition of the same name.  `maintenance info pdb-nesting`
reports how large the class-nesting map is and what building it cost.

### What a prefix can be

An anonymous namespace has no name in the source, so the compiler invents one.
It writes `?A0x<ID>` in a symbol name and `` `anonymous-namespace' `` in a type
name, where `<ID>` is a hex number identifying one translation unit.  Both
stand for the same namespace.

Take `?A0x<ID>::Lowering::emit::__l2::<lambda_1>`, a lambda inside a method
of a class declared in an anonymous namespace.  Its prefixes are four
different kinds of scope, and only the first is a namespace:

- `?A0x<ID>` — the anonymous namespace.  Registered.
- `?A0x<ID>::Lowering` — a class.  Looking the name up finds nothing, because
  the type records spell the same class `` `anonymous-namespace'::Lowering ``,
  so it is asked a second time with the id rewritten to the quoted form.
- `?A0x<ID>::Lowering::emit` — a method.  Registering it would leave one
  name meaning both a function and a namespace.
- `?A0x<ID>::Lowering::emit::__l2` — `__l` followed by a number is the
  compiler's marker for a block inside a function; C++ cannot name it.

The name is only how all of that looks written down.  The tag itself records
that a function body encloses it, so the first rule above settles the whole
name at once and none of these components is ever examined.

### Method physname

CodeView splits the method name across two TPI records:
`LF_ONEMETHOD` carries the leaf name (`"make"`), and the enclosing
`LF_CLASS` carries the qualified tag name (`"ns_a::ns_b::Box"`).  The
linkage name MSVC emits for the function symbol is the
concatenation of these two. E.g. `"ns_a::ns_b::Box::make"`.

So when building a method's `fn_field`, the LF_ONEMETHOD /
LF_METHOD parsers glue `enclosing_name + "::" + leaf` together and
store the result as `physname` (for every non-friend method).

## Build & Configuration

- `gdb/configure.ac` / `gdb/configure` — PDB support is built when any Windows
  target is configured or when `--enable-targets=all` is used.  Can be forced
  with `--enable-gdb-pdb-support=yes` or disabled with `=no`.  Defines
  `PDB_FORMAT_AVAILABLE`.
- `gdb/Makefile.in` — adds `pdb` subdir; defines `PDB_SRCS` / `PDB_OBS`.
- `gdb/config.in` — `PDB_FORMAT_AVAILABLE` macro.

## Limitations / TODO

Record-level detail for everything below is in `LLM-WIKI.md`.

### Not implemented

- **No MSVC demangler.**  A `?`-name keeps its mangled spelling, so no
  source-form signature matches:

      break A::foo::test            ->  7 locations
      break A::foo::test(int)       ->  Function ... not defined
      break '?test@foo@A@@QEAAXH@Z' ->  resolves

  Not CodeView's fault -- clang DWARF on this target fails the same way, while
  the mingw build passes -- but it is the largest single source of failures
  here, and also breaks operator overloads, `typeid`, and calls through
  member-function pointers.  `libdemangle-msvc/` is an empty scaffold.
- No calling convention or stack unwinding support.
- No symbol server, so a PDB that is not already on disk is never fetched.
- No macro information; a macro-defined name such as `errno` never resolves.
- Thread-local variables are skipped.
- Source line columns are dropped, leaving line numbers only.
- Declaration lines are not recorded, so every symbol reports line 0 and
  `info functions` / `info variables` / `info types` print no `NN:` prefix.
  A procedure's first line-table entry is its first *executable* line, which
  also stops `break FILE:LINE` from advancing past a line holding no code.
- Template parameters are not decoded; an instantiation keeps whatever
  spelling GDB's own C++ parser produces.
- Separated code -- a function the optimizer split into chunks that are not
  contiguous -- is not mapped.
- Debug info left inside an unlinked object file is not read; only PDBs are.
  Reading it would need a different entry point, a type index with no stream
  header, section-relative addressing, the object's own file and string
  tables, and the procedure records that the linker normally rewrites.

### Needs an MSVC C++ ABI

- Virtual bases appear in `ptype` but cannot be read at run time.  GDB's path
  expects the Itanium layout, where the offset to a virtual base sits in the
  vtable; MSVC keeps it in a separate table reached through its own pointer.
- Calling through a member-function pointer does not work.
- Returning a class by value *is* handled, but not by GDB's usual rule.  MSVC
  returns anything that is not plain-old-data through memory the caller
  supplies, even when it would fit in a register, while mingw's GCC does the
  opposite -- so the reader marks those types itself rather than changing a
  convention shared with every other Windows target.

### Where CodeView carries less than DWARF

- Typedefs are flattened: there is no typedef record, so anything declared
  through one prints as the underlying type.  The name survives only as a
  separate binding.
- Two types with the same display name share one entry in the by-name index,
  so `info types` shows one where DWARF shows one per file, and a by-name
  lookup finds either.  Types declared inside a function or an anonymous
  namespace collide this way.  Variables and fields are unaffected, since they
  resolve through their own type rather than by name.
- `Enum::Enumerator` does not resolve for a scoped enum, which is recorded
  identically to a plain one.  The enumerators are reachable in the scope
  holding the enum instead.
- When one statement contains several inlined calls, only the first gets a
  source location.  The rest leave the caller printed bare in a backtrace and
  change how `step` and `next` behave there.  Compiling with clang's
  `-gcolumn-info` avoids it, but that is a property of the program being
  debugged, not something the debugger can supply.

### Variables in optimized code

- Only part of the location model is implemented, so a variable the compiler
  split across registers and stack slots can read `<optimized out>` while it
  is in fact live.  In the compiled fixtures the exposure is small: they carry
  no general location programs at all, and the sub-field records they do have
  are confined to C runtime code.
- A location describing one spilled member of an object is dropped rather than
  applied, because using it for the whole object would read the neighbouring
  members' bytes.  DWARF assembles the equivalent into a partially-available
  value; matching that needs piece-wise values in the location model.
- A location the producer marked as uncertain is presented as definite.
- Only the AMD64 general-purpose registers are mapped; a variable living
  anywhere else reads as unavailable.

### Lookups that fail

Each of these refuses the query rather than answering it wrongly.

- A class-qualified expression does not parse inside a method: stopped in
  `D::f`, `print i` works but `print D::i` is a syntax error
  (`gdb.cp/impl-this.exp`).
- A typedef member reached through its class does not resolve --
  `ptype A::value_type` (`gdb.cp/derivation.exp`).
- A type declared inside a function or an anonymous namespace is not reachable
  by qualified name -- `ptype main::Foo` (`gdb.cp/subtypes.exp`).
- A name introduced by a `using` directive does not resolve
  (`gdb.cp/nsusing.exp`).
- Choosing a method overload by explicit instantiation can fail
  (`gdb.cp/meth-typedefs.exp`).
- Calling a function the compiler inlined is refused rather than redirected to
  an out-of-line copy.
- `break FUNC:LABEL` does not find labels, although the records are read.

### Fallback

A PE naming a PDB that cannot be used falls through to the DWARF reader, as
does a build configured without PDB support.

## Memory Allocations

Most of the allocations use the objfile obstack.

Non obstack allocations:

**Heap (`new`):**
- `pdb_per_objfile` — registered via `registry<objfile>::key`, auto-deleted
  when objfile is destroyed.
- `buildsym_compunit` —  builder, deleted after modules are
  `pdb_build_module()` / `pdb_expand_all_modules()`.

**Memory-mapped:**
- The whole PDB file is mapped read-only for the objfile's lifetime
  (`pdb_map_file`).  `pdb_stream_bytes()` returns a pointer into the mapping
  when the requested range lies inside one MSF block, and gathers into a
  caller-supplied scratch buffer only when it crosses a block boundary.

**Scoped (`unique_ptr<gdb_byte[]>`):**
- `pdb.c` - reading stream directory and stream block map.  Freed automatically.
- `pdb.c` `pdb_read_stream()` - reading of the actual streams bytes.
  Released into `pdb->stream_data[]` (`pdb` on obstack) or freed automatically.
- `pdb-path.c` — temporary buffers for PE executable access.


## Source Files and What They Do

### `pdb.h` — Public API and Data Structures

The public types, constants and prototypes.  One context object per objfile
holds everything read from the file -- stream layout, module list, section
addresses, the type and symbol indexes -- and dies with the objfile.  The
smaller structures hang off it: one per module, one for the type stream and
its cache, and one per variable whose location changes as the program runs.

### `pdb.c` — MSF Parsing, Stream Assembly, Module Expansion

The orchestrator: file I/O, block assembly, stream parsing, and turning a
module into something GDB can use.  It is the entry point the COFF reader
calls, and it drives the other readers in the order listed above.

Serving stream bytes is its job too.  A range inside a single block is handed
out as a pointer into the mapped file; only a range crossing a block boundary
is copied.

Expanding a module produces a compilation unit with its line table and block
tree.  Two details are easy to get wrong and are handled here:

- **Where a breakpoint on a function lands.**  Procedures are scanned for the
  end of their prologue before lines are read, so the line table can mark the
  entry past the parameter stores.  When the record does not say where the
  prologue ends, it is inferred from where the parameters become live, or from
  the frame description for a procedure that sets nothing up.
- **Line ordering across files.**  Rows are collected and sorted by address
  before being handed over, so a run of lines from one file is closed where
  another file's code begins -- which happens whenever a header is inlined.

What GDB gets from all this is a compilation unit per module tagged as
`CodeView`, the usual block tree including ranges for functions the optimizer
scattered, and the hooks GDB calls to look symbols up.

By default expansion is lazy (see `pdb-index.c`); `--readnow` expands every
module at load instead.

### `pdb-index.c` — Lazy Loading

A module's `compunit_symtab` is built only when a query needs it, and a tagged
type only when one is asked for.  A name index says which module owns which
name.

The index holds function and global-data names with their owning module, plus
a PC → module map from the DBI section contributions.  Names come from the GSI
(`S_PROCREF` / `S_LPROCREF` and the global symbol stream they point at), or,
with no GSI, from scanning every module symbol stream.  Types are not indexed
here: a type-name → type-index map is built at load, the type itself on first
use.

Inlined functions need a source of names of their own.  A function that was
only ever inlined has no procedure record anywhere, so neither the GSI nor a
module scan would find it until its caller had been expanded -- and the caller
is exactly what a user has not found yet.  Every module also lists the
functions inlined into it, so those names are added to the index as well, and
join the no-GSI fallback scan.  Turning an entry in that list into a name is
what the IPI is for: a member function is joined to its class name, a free
function to its namespace string, which may itself have to be assembled from
separate pieces.  This work happens on the index threads, so it reads raw
records and builds no types.  The result is an ordinary function entry, not a
minimal symbol.

Global data and constants are built at load into `<pdb-globals>`, so variable
lookups never expand a module.  A name owned by several modules expands them
all.  Function-local statics and local types are in neither the GSI nor the
index; they are reached by expanding the module covering the PC.

A lookup searches what is already built and consults the index only on a miss.
The index is built on a background thread; queries wait for it.

Three lookup shapes need more than an exact match on the stored name:

- **Qualified C++ names.**  The GSI records a method as the single string
  `A::func`, where DWARF gives `func` and a parent link.  The index cuts the
  scopes out of the name when it is indexed, as `pdb_scope_component_len`
  steps through them: `fubar::inner::deep_func` gives the entries `fubar`,
  `inner` (parent `fubar`) and `deep_func` (parent `inner`), one scope entry
  per module and scope.  Each shard is one list sorted by unqualified name.
  A lookup follows DWARF's `cooked_index_functions::search`: find the last
  segment, check each earlier one against the parents, then match the full
  name.  So `break deep_func`, `break inner::deep_func` and
  `break fubar::inner::deep_func` all find it, and completion reaches the
  scope entries (`fu<TAB>` finds `fubar`).  The name comparison is a copy of
  `cooked_index_entry::compare`, since GDB can be built without DWARF.
- **Enumerators.**  An enumerator has no record of its own; it exists only as
  an `LF_ENUMERATE` inside its enum's `LF_FIELDLIST`.  After the tag-name
  scan, the field list of each selected non-forward `LF_ENUM` is walked into
  `enumerator_names` (bare enumerator → every declaring enum's tag).  A
  variable-domain lookup forms each candidate's qualified name
  (`scope::Enumerator`) and builds the enums whose names the C++ symbol
  matcher accepts.  DWARF's cooked index carries `DW_TAG_enumerator` entries
  for the same reason.
- **Completion.**  A completion prefix cannot use the exact-match type path,
  so a type-domain query in completion mode builds every tagged type whose
  name the C++ symbol matcher accepts for the lookup.

| Command | Effect |
|---|---|
| `--readnow` | Skip the index; expand every module at load |
| `maintenance expand-symtabs` | Build every module and type now |
| `maintenance set pdb-force-module-index` | Scan modules even with a GSI present |
| `maintenance set pdb-lazy-globals` | Index globals instead of building `<pdb-globals>` at load |
| `maintenance set pdb-synchronous` | Build the index on the loading thread |

The three `maintenance set` options are off by default and take effect on the
next objfile load.

### `pdb-read-types.c` — TPI Stream Parsing & Type Resolution

Reads the TPI stream header, builds an indexed array of `pdb_tpi_type` records
(pointers directly into cached stream data — no copy), and resolves type
indices to GDB `struct type` on demand with caching.

`pdb_read_tpi_stream()` is the entry point: it indexes every record without
resolving it.  `pdb_tpi_resolve_type()` is the on-demand resolver — it builds
and caches a `struct type` the first time a symbol references an index,
dispatching per leaf kind (pointer, array, struct/class, enum, ...).

The file is ordered in six sections: raw record access and decoding (no GDB
types), TPI/IPI stream loading and load-time indexes, IPI function
identities, GDB type construction (makers and the resolver), struct/union/enum
members (field-list handlers, virtual-slot recovery and the deferred-member
queue), and type symbols.


### `pdb-read-symbols.c` — CodeView Symbol Parsing & Location Handling

Parses CodeView symbol records from module streams and the global symbol record
stream. Creates GDB symbols with resolved types and locations.

`pdb_parse_symbols()` is the per-module entry point; `pdb_load_global_syms()`
adds globals and constants and `pdb_build_minsyms()` builds minimal symbols
from public symbols.  At runtime `pdb_loclist_read_variable()` serves a
variable's value through GDB's `symbol_computed_ops`.

The `pdb_sym` wrapper structs are stack-allocated during parsing — they
exist only long enough to extract fields from the raw record and call
`create_gdb_sym()`, which creates the obstack-allocated GDB symbol.


### `pdb-path.c` — PDB File Discovery

Searches for the PDB file using multiple strategies in priority order:

1. RSDS basename in the EXE directory
2. `<exe-name>.pdb` next to the executable
3. Each entry in `debug-file-directory`: `<dir>/<pdb-basename>`
4. Full RSDS embedded path
5. `_NT_ALT_SYMBOL_PATH` / `_NT_SYMBOL_PATH` (semicolon-separated;
   `SRV*`/`SYMSRV*`/`CACHE*` prefixes recognized but fetch is stubbed)
6. Windows registry `Software\Microsoft\VisualStudio\MSPDB\SymbolSearchPath`
   (HKCU then HKLM)

Also contains `pdb_read_rsds_info()` which reads the RSDS debug directory
entry from the PE executable to extract GUID, age, and PDB path.

### `pdb-cmd.c` — Diagnostic / Introspection Commands

Registers all `maintenance info pdb-*` commands (listed under
[Maintenance Commands](#maintenance-commands)). Each accepts optional
`path=<pdb>` and `modi=N` arguments to select a specific PDB or module.
