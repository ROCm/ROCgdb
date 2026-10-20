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

Each record in the stream has variable length consisting of a 2-byte
`RecordLen`, a 2-byte `RecordKind` (the "leaf type" identifier such as `LF_POINTER`,
`LF_MODIFIER`, `LF_ARRAY`, `LF_PROCEDURE`, `LF_ARGLIST`...), and a payload whose
layout depends on the leaf type. Fields within a record can reference other
types by their type index, forming a directed graph (e.g. an `LF_POINTER` record
contains the type index of the pointee type).

Reading the stream at load time only *indexes* it; no GDB type is created then.
Two structures come out of that, and each exists because CodeView leaves out
something DWARF gives a reader for free:

- **A table of records, addressed by type index.**  Records are variable length
  and refer to each other by number, so nothing can jump to type 0x1234 without
  having already walked everything before it.  Walking the stream once and
  remembering where each record starts turns every later reference into an
  array lookup.  DWARF needs no equivalent, because a reference to another DIE
  is already a byte offset.

- **A cache of the GDB types already built.**  Filled as types are requested,
  so an index is converted at most once.

A record becomes a GDB type only when something asks for it -- printing a
variable, running `ptype`, or setting a breakpoint on a method.

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
from here.  Building it needs both streams above, and takes two passes.

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

If the publics index stream is absent, no minimal symbols are built.

The GSI (Global Symbol Index) stream is a hash table for O(1) name to symbol
lookup similar to DWARF's .debug_names. It indexes cross-reference records
(S_PROCREF, S_LPROCREF, S_DATAREF) that point into module streams — each
reference carries module index and offset, telling the reader which module
contains the full symbol definition.

## PDB Discovery

PDB files are searched at different locations - the PDB name recorded in the
RSDS record of the Debug Directory section in the executable is used as the
base name. We look next to the EXE, then try the full RSDS embedded path. We
also search for the PDB by simply replacing the EXE extension with `.pdb`.

### Path Conversion (MSYS2)

PDBs produced under MSYS2 can have Linux style paths which are converted into
Windows style paths before storing them to symtab linetables, so that GDB
can load them. This either requires prepending the MSYS2 root
(e.g. /home/PATH -> C:/msys2/PATH) or converting drive information
(e.g. /c/PATH -> C:/PATH).

The MSYS2 root must be specified using MSYS2_ROOT env. var, otherwise we look
into common msys2/mingw64 directories.

## Maintenance Commands

`pdb-loaded-files` takes no arguments.

| Command                            | What it shows                                             |
|------------------------------------|-----------------------------------------------------------|
| `maintenance info pdb-loaded-files`| Paths of all currently loaded PDB files                   |

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
4. Index the TPI (types) stream.  Records are indexed but not resolved; a
   type is built the first time a symbol references it.
5. Read PE section addresses from BFD, used to relocate section:offset pairs
   into real addresses.
6. Build minimal symbols from the public symbol stream.
7. Expand every module (`pdb_expand_all_modules()`) into a compunit.

## Build & Configuration

- `gdb/configure.ac` / `gdb/configure` — PDB support is built when any Windows
  target is configured or when `--enable-targets=all` is used.  Can be forced
  with `--enable-gdb-pdb-support=yes` or disabled with `=no`.  Defines
  `PDB_FORMAT_AVAILABLE`.
- `gdb/Makefile.in` — adds `pdb` subdir; defines `PDB_SRCS` / `PDB_OBS`.
- `gdb/config.in` — `PDB_FORMAT_AVAILABLE` macro.

## Supported

- The PDB that the executable's RSDS debug record names, found next to the
  executable under that name, as `<executable>.pdb`, or at the recorded
  path.  A PDB whose GUID and age differ from the executable's is refused,
  and GDB reads the executable's DWARF, if it has any, instead.
- Minimal symbols from the public symbols (`S_PUB32`).  A data object is
  named by the source name of its global record.
- Line tables (`DEBUG_S_LINES`, `DEBUG_S_FILECHKSMS`), for breakpoints by
  line, stepping and `list`.  MSYS2 paths are converted to Windows paths.
- Functions (`S_GPROC32`, `S_LPROC32`, `S_THUNK32`) with their address
  ranges, and function types built from built-in types (`LF_PROCEDURE`,
  `LF_ARGLIST`).  A function recorded by id (`S_GPROC32_ID`,
  `S_LPROC32_ID`) has no type.
- `maintenance info pdb-loaded-files`.

## Memory Allocations

Most of the allocations use the objfile obstack.

Non obstack allocations:

- `pdb_per_objfile` — registered via `registry<objfile>::key`, deleted
  with the objfile.
- `buildsym_compunit` — a local of the function building a compunit, such
  as `pdb_build_module()`.
- Streams are read into `gdb::byte_vector`s.  The ones kept for the
  objfile's lifetime live in `pdb_per_objfile` (`info_data`, `names_data`,
  `dbi_data`, `sym_record_data`); the others, such as a module's stream, are
  freed when the reading function returns.

**Memory-mapped:**
- The whole PDB file is mapped read-only for the objfile's lifetime
  (`pdb_map_file`).  `pdb_stream_bytes()` returns a pointer into the mapping
  when the requested range lies inside one MSF block, and gathers into a
  caller-supplied scratch buffer only when it crosses a block boundary.
  On Windows the file is opened with `FILE_SHARE_DELETE`, so a linker can
  still replace the PDB while GDB has it mapped, as it can a mapped ELF
  file; with a plain `fopen` lld failed to write it ("Stream Error").

## Source Files and What They Do

### `pdb.h` and `pdb-internal.h` — Interfaces and Data Structures

`pdb.h` declares the entry point the COFF reader calls.  `pdb-internal.h`
holds the types, constants and prototypes shared by the reader's files.  One
context object per objfile holds everything read from the file -- stream
layout, module list, section addresses, the type and symbol indexes -- and
dies with the objfile.  The smaller structures hang off it: one per module
and one for the type stream and its cache.

### `pdb.c` — MSF Parsing, Stream Assembly, Module Expansion

The orchestrator: file I/O, block assembly, stream parsing, and turning a
module into something GDB can use.  It is the entry point the COFF reader
calls, and it drives the other readers in the order listed above.

Serving stream bytes is its job too.  A range inside a single block is handed
out as a pointer into the mapped file; only a range crossing a block boundary
is copied.

Expanding a module produces a compilation unit with its line table and block
tree.  These details are easy to get wrong and are handled here:
- **Line ordering across files.**  Rows are collected and sorted by address
  before being handed over, so a run of lines from one file is closed where
  another file's code begins -- which happens whenever a header is inlined.
- **Primary file.**  The compilation unit is named after the module.
- **Modules with no code.**  A module whose DBI record has no section
  contribution (it declares only data, constants or types) is still read
  when it has a symbol stream.  Its compilation unit gets an empty address
  range, so it never claims a PC, and is dropped if it holds procedures,
  which cannot be placed without a range, or no file or global symbols.
  A module without a symbol stream (an object compiled without debug info)
  gets no compilation unit, as DWARF gives such
  an object no CU; a PC in its code has no symbol table.  Nor does a module
  that has code but declares nothing and has no lines: lld's `Import:DLL`
  modules describe their import thunks only with `S_THUNK32`, and a
  compilation unit over a thunk would hide its minimal symbol from
  `info functions`.

What GDB gets from all this is a compilation unit per module tagged as
`CodeView`, the usual block tree and the hooks GDB calls to look symbols up.
Every module is expanded when the PDB is loaded.

### `pdb-read-types.c` — TPI Stream Parsing & Type Resolution

Reads the TPI stream header, builds an indexed array of `pdb_tpi_type` records
(pointers directly into cached stream data — no copy), and resolves type
indices to GDB `struct type` on demand with caching.

`pdb_read_tpi_stream()` is the entry point: it indexes every record without
resolving it.  `pdb_tpi_resolve_type()` is the on-demand resolver — it builds
and caches a `struct type` the first time a symbol references an index,
dispatching per leaf kind.

### `pdb-read-symbols.c` — CodeView Symbol Parsing

Parses CodeView symbol records from module streams and the global symbol record
stream. Creates GDB symbols with resolved types and locations.

`pdb_parse_symbols()` is the per-module entry point and `pdb_build_minsyms()`
builds minimal symbols from public symbols.

The `pdb_sym` wrapper structs are stack-allocated during parsing — they
exist only long enough to extract fields from the raw record and call
`create_gdb_sym()`, which creates the obstack-allocated GDB symbol.

### `pdb-path.c` — PDB File Discovery

Searches for the PDB file using multiple strategies in priority order:

1. RSDS basename in the EXE directory
2. `<exe-name>.pdb` next to the executable
3. Full RSDS embedded path

Also contains `pdb_read_rsds_info()` which reads the RSDS debug directory
entry from the PE executable to extract GUID, age, and PDB path, and
`pdb_convert_path()`, which converts MSYS2 source paths
(see [Path Conversion](#path-conversion-msys2)).

### `pdb-cmd.c` — Diagnostic / Introspection Commands

Registers all `maintenance info pdb-*` commands (listed under
[Maintenance Commands](#maintenance-commands)). Each accepts optional
`path=<pdb>` and `modi=N` arguments to select a specific PDB or module.
