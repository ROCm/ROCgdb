# PDB Reader for GDB — Architecture & Data Flow

## Overview

PDB is a multi-stream file container where different streams provide different
debug information. Streams are composed of multiple blocks, which don't have to
be consecutive. The blocks are the actual physical parts of the file — the PDB
file itself consists of multiple fixed-size blocks (except for the header).

The reader reads the following from the PDB at load.

### MSF header (SuperBlock)

The MSF SuperBlock is the first block in the PDB file and contains basic
information such as the block size, number of blocks, and most importantly,
the location of the stream directory, which is used to locate all other
streams in the file. The SuperBlock is 64 bytes long.

### Stream directory

The stream directory, whose blocks the SuperBlock's block map lists, specifies
which block belongs to which stream. Each stream can span multiple physical
blocks that are not necessarily contiguous.

With the information from the stream directory, we are able to parse any stream.

### PDB Info stream (stream 1)

Basic information stream - it holds the GUID and age that identify the build
that wrote the PDB, and the named stream map, which locates the "/names"
stream (the String Table) that names all the files compiled into the PDB.

### Names stream (String Table)

Contains info on all the files used by all modules compiled into the PDB.
The names are read into the String Table, loaded eagerly because almost any
operation (break, info sources ...) references a module's files to get line
information, so preloading avoids repeated lookups.

### DBI stream (stream 3)

DBI stream contains the debug information (line numbers, symbols, etc.) for all
the modules (object files) linked into the program. Each module's debug info is
in a different stream. The reader reads the header, which holds the DBI age and
the stream numbers of the global streams, the module records, which say which
stream holds each module's debug info, and the Section Contribution and File
Info substreams; then it reads every module's stream.

### DBI Section Contribution substream

One entry per contribution: the bytes an object file put in a section of the
image. A module's record names only its first contribution, while a module
compiled with a section per function (`/Gy`) has one per function, so the
reader takes a module's code from its code contributions here.

### DBI File Info substream

Substream is just a piece of data located at a given offset in a stream.
The File Info substream contains info on all the files used by all the modules
compiled into the PDB - the Names Buffer. Names Buffer actually duplicates the
String Table but it also adds the information on files that go into each module.
Getting a module's files from the String Table instead would mean reading
the module's stream for the file checksums that refer to them.

### Module Streams

Module streams contain the debug information for individual modules (object
files). Various debug sections are specified using identifiers — e.g. symbols
or line information or file info. The line information is in C13 sections
(C11 sections are obsolete). C13 sections are split into subsections, most
importantly Checksums and Lines. The Checksums subsection references the
String Table to provide the source files that belong to the module, while the
Lines subsection maps addresses to source lines (analogous to `.debug_line` in
DWARF).

### Symbol Record Stream / PSGSI

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

## PDB Discovery

The PDB is looked for in three places, in this order: next to the EXE under
the base name the RSDS record of the executable's debug directory gives,
next to the EXE under the EXE's own name with `.pdb` for its extension, and
at the full path the RSDS record gives. That path is not opened when it is
a network or device path (`\\host\share\...`, `\\?\...`), as opening one
makes Windows send the user's credentials to that host, or when it is not
absolute on the host GDB runs on.

### Path Conversion (MSYS2)

PDBs produced under MSYS2 can have Linux style paths which are converted into
Windows style paths before storing them to symtab linetables, so that GDB
can load them. This either requires prepending the MSYS2 root
(e.g. /home/PATH -> C:/msys2/PATH) or converting drive information
(e.g. /c/PATH -> C:/PATH).

The MSYS2_ROOT environment variable names the MSYS2 root; without it, or for a
file not under it, the reader tries the usual `msys64` and `msys2` roots.

## Maintenance Commands

`pdb-loaded-files` takes no arguments.

| Command                            | What it shows                                             |
|------------------------------------|-----------------------------------------------------------|
| `maintenance info pdb-loaded-files`| Paths of the PDB files GDB read                           |

## GDB Integration and Initialization Order

`coff_reader::symfile_read()` reads DWARF first, then looks for a separate
debug file, and calls `pdb_initialize_objfile()` only for an executable that
has neither.  lld-link writes a PDB for every link with `-debug`, so a DWARF
build names one too; it holds only the C runtime's CodeView, which is why a
separate debug file comes first.  `pdb_initialize_objfile()` returns true
only when it loaded the PDB that the executable's RSDS record names.  An
RSDS record with an empty path, as lld-link `/build-id` writes, names none.
A missing, mismatched or malformed PDB returns false, and GDB reads no debug
information.  When PDB support is not compiled in, a stub returns false.

Initialization then loads the PDB in this order:

1. Map the PDB file read-only, validate the MSF magic and header, and read
   the stream directory (which maps every stream to its blocks).
2. Read the PDB Info stream, validate its GUID/age against the EXE's RSDS
   record, and load the `/names` string table.  A mismatch aborts the load:
   symbol offsets, line tables and type indices are all keyed to the exact
   build that produced the EXE, so a PDB from another build cannot be used
   at all.  Aborting also discards the `pdb_per_objfile` that steps 1-2 had
   already attached to the objfile, so the refused file is not left behind
   for `maint info pdb-loaded-files` to report as loaded.
3. Parse the DBI stream: its header, whose age, when not 0, must match the
   EXE's, the stream numbers of the symbol record and PSGSI streams, and
   the module records.
4. Read PE section addresses from BFD, used to relocate section:offset pairs
   into real addresses.
5. Build minimal symbols from the public symbol stream.
6. Expand every module (`pdb_expand_all_modules()`) into a compunit.

A malformed PDB that fails a check in steps 1-5 is dropped as a mismatched
one is, with the error shown as a warning.  A module whose stream is
malformed loses its own lines only, and is skipped with a warning.  Counts
and sizes read from the file are checked against the bytes that hold them in
64-bit arithmetic, and a stream cannot have more blocks than the file.

## Build & Configuration

- `gdb/configure.ac` / `gdb/configure` — PDB support is built when any Windows
  target is configured or when `--enable-targets=all` is used.  Can be forced
  with `--enable-gdb-pdb-support=yes` or disabled with `=no`.  Defines
  `PDB_FORMAT_AVAILABLE`.
- `gdb/Makefile.in` — adds `pdb` subdir; defines `PDB_SRCS` / `PDB_OBS`,
  which hold the whole reader and are built only with it.
- `gdb/config.in` — `PDB_FORMAT_AVAILABLE` macro.  Without it, `pdb.h`
  defines `pdb_initialize_objfile` as an inline function that loads no
  PDB, as `dwarf2/public.h` does for DWARF.

## Supported

- The PDB that the executable's RSDS debug record names, found next to the
  executable under that name, as `<executable>.pdb`, or at the recorded
  path.  A PDB whose GUID differs from the executable's, or whose age is
  lower, is refused, and nothing is read from it.  An executable with
  DWARF is read as DWARF, and the PDB it names is not read.
- Minimal symbols from the public symbols (`S_PUB32`).  A data object is
  named by the source name of its global record.
- Line tables (`DEBUG_S_LINES`, `DEBUG_S_FILECHKSMS`), for breakpoints by
  line, stepping and `list`.  MSYS2 paths are converted to Windows paths.
- `maintenance info pdb-loaded-files`.

## Memory Allocations

Most of the allocations use the objfile obstack.

Non obstack allocations:

- `pdb_per_objfile` — registered via `registry<objfile>::key`, deleted
  with the objfile.
- `buildsym_compunit` — a local of the function building a compunit, such
  as `pdb_build_module()`.
- Streams are copied out of the file mapping into `gdb::byte_vector`s by
  `read_stream()`.  The ones kept for the
  objfile's lifetime live in `pdb_per_objfile` (`info_data`, `names_data`,
  `dbi_data`, `sym_record_data`); the others, such as a module's stream, are
  freed when the reading function returns.

**Memory-mapped:**
- The whole PDB file is mapped read-only for the objfile's lifetime
  (`pdb_map_file`) as soon as it is opened; the MSF magic, the SuperBlock,
  the stream directory and every stream are read from the mapping.
  `pdb_stream_bytes()` returns a pointer into the mapping
  when the requested range lies inside one MSF block, and gathers into a
  caller-supplied scratch buffer only when it crosses a block boundary.
  Only a regular file is mapped, with the size its handle reports, and the
  file's handles are closed once it is mapped.  On Windows the file is
  opened with `FILE_SHARE_DELETE`, so lld can still replace the PDB while
  GDB has it mapped; with a plain `fopen` lld failed to write it
  ("Stream Error").  It is also shared for writing, as `fopen` shares it.
  On a POSIX host another process can truncate a mapped file, and GDB then
  faults (SIGBUS) on a read past the new end.

## Source Files and What They Do

### `pdb.h` and `pdb-internal.h` — Interfaces and Data Structures

`pdb.h` declares the entry point the COFF reader calls.  `pdb-internal.h`
holds the types, constants and prototypes shared by the reader's files.  One
context object per objfile holds everything read from the file -- stream
layout, module list and section addresses -- and dies with the objfile.
It also holds one record per module.

### `pdb.c` — MSF Parsing, Stream Assembly, Module Expansion

The orchestrator: file I/O, block assembly and stream parsing.  It is the
entry point the COFF reader calls, and it drives the other readers in the
order listed above.

Serving stream bytes is its job too.  A range inside a single block is handed
out as a pointer into the mapped file; only a range crossing a block boundary
is copied.

Expanding a module produces a compilation unit with its line table.  These
details are easy to get wrong and are handled here:
- **Line ordering across files.**  Rows are collected and sorted by address
  before being handed over, so a run of lines from one file is closed where
  another file's code begins -- which happens whenever a header is inlined.
  At one address an end marker goes first, since buildsym drops the rows at
  an end marker's address: Microsoft cl can list a procedure's line block
  before the block that ends where the procedure starts.
- **Primary file.**  The compilation unit is named after the module.
- **Modules with no code or no lines.**  A module whose DBI record has no
  section contribution, or that has no line data, gets no compilation unit.
  Nor does a module without a symbol stream (an object compiled without
  debug info), as DWARF gives such an object no CU.

What GDB gets from all this is a compilation unit per module tagged as
`CodeView`, with its line table.  Every module is expanded when the PDB is
loaded.

### `pdb-read-symbols.c` — CodeView Symbol Parsing

Parses the global symbol record stream and the public symbols.
`pdb_build_minsyms()` builds minimal symbols from public symbols, and
`pdb_module_language()` reads a module's language from its `S_COMPILE2` or
`S_COMPILE3` record.

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

Registers `maintenance info pdb-loaded-files` (see
[Maintenance Commands](#maintenance-commands)).
