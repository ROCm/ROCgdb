# PDB Reader for GDB — Architecture & Data Flow

## Overview

PDB is a multi-stream file container where different streams provide different
debug information. Streams are composed of multiple blocks, which don't have to
be consecutive. The blocks are the actual physical parts of the file — the PDB
file itself consists of multiple fixed-size blocks (except for the header).

Following is the data from PDB we need to read on initialization.

### MSF header (SuperBlock):

The MSF SuperBlock is the first block in the PDB file and contains basic
information such as the block size, number of blocks, and most importantly,
the location of the stream directory, which is used to locate all other
streams in the file. The SuperBlock is 64 bytes long.

###  Stream directory

The stream directory is located immediately after the SuperBlock and specifies
which block belongs to which stream. Each stream can span multiple physical 
blocks that are not necessarily contiguous.

With the information from the stream directory, we are able to parse any stream.

###  PDB Info stream (stream 1)

Basic information stream - it's most significant part is the location of the 
the "/names" stream (the String Table) which contains the list of all the files
compiled into the PDB.

###  Names stream (String Table):

Contains info on all the files used by all modules compiled into the PDB.
The names are read out into the String Table. The table is loaded eagerly
because just about any module will need to reference it when trying to display
it's files. The concept of lazy loading assumes we access the data only when
needed i.e. - only when a particular module is referenced. In PDB case,
accessing just about any module (break, info sources...) will quickly reference 
this table in order to get the line information, thus we just preload the table.

###  DBI stream (stream 3):

DBI stream contains the debug information (line numbers, symbols, etc.) for all
the modules (object files) linked into the program. Each module's debug info is
in a different stream and we read those streams on request. Eagerly we only load
the header which contains info on per module streams (debug info is per module).

###  DBI File Info substream

Substream is just a piece of data located at a given offset in a stream.
The File Info substream contains info on all the files used by all the modules
compiled into the PDB - the Names Buffer. Names Buffer actually duplicates the
String Table but it also adds the information on files that go into each module.
This is suitable for Quick Functions that check if a files is in a module;
obtaining this info from the String Table would require expanding the parts
of the module stream, to get the sections that reference per per module files 
(indices into String Buffer).

The duplication of the file names likely exists for compatibility.

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

The TPI stream is parsed eagerly at load time — type records are indexed so
they can be resolved on demand when a symbol references a type index. Resolved
types are cached so each type index is converted to a GDB `struct type` at most
once.

### IPI stream (stream 4). TODO

The IPI (Id Program Information) stream has the same physical layout as the TPI
stream but contains *id records* rather than type records. Id records reference
items like functions, strings, and build information by name rather than by
type structure. Currently the IPI stream is not parsed.

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
symbol records — both private globals (S_GPROC32, S_GDATA32, S_PROCREF, etc.)
and public symbols (S_PUB32).

The PSGSI (Public Symbol Index) stream is PDB's equivalent of the ELF symbol
table (`.symtab`/`.dynsym`) — it contains a hash table whose hash records point
into the Symbol Record Stream to locate the S_PUB32 records stored there. 
After the hash table there is an address to name map that is used to build 
the GDB minimal symbol table.

The GSI (Global Symbol Index) stream is a hash table for O(1) name to symbol
lookup similar to DWARF's .debug_names. It indexes cross-reference records 
(S_PROCREF, S_LPROCREF, S_DATAREF) that point into module streams — each 
reference carries module index and offset, telling the reader which module 
contains the full symbol definition. We use this table to build the cooked index 
and provide quick functions for symbol lookup on GDB's request.

## Finding PDB files.

PDB files are searched at different locations - for the main executable the user
can specify the  --pdb-path command-line override. Further, we search for PDB
by the PDB name recorded in the so called RSDS record of the Debug Directory
section in the actual executable. This PDB name is searched as is or as the
base name in the EXE directory. We also search for the PDB by simply replacing
the EXE name.

Windows can specify the location of the PDB files in Windows registry or in the
environment variables.

TODO: For system DLLs, Windows normally uses so called Debug Symbol server from
where the PDB files can be downloaded.

## Path Conversion (MSYS2)

PDBs produced under MSYS2 can have Linux style paths which are converted into
Windows style paths before storing them to symtab linetables, so that GDB
can load them. This either requires prepending the MSYS2 root
(e.g. /home/PATH -> C:/msys2/PATH) or converting drive information
(e.g. /c/PATH -> C:/PATH).

The MSYS2 root must be specified using MSYS2_ROOT env. var, otherwise we look
into common msys2/mingw64 directories.


## Info Commands

All commands accept optional `path=<pdb-path>` and `modi=N` arguments to
select a specific PDB / module. If omitted, the default (main program) PDB and
all modules are used.

`info pdb-loaded-files`  List paths of all currently loaded PDB files.
`info pdb-modules`       List modules (object files) in the PDB with 
                         stream numbers and file counts.
`info pdb-files`         List source files per module from the DBI File Info 
                         substream.
`info pdb-files-c13`     List source files per module from C13 Checksums 
                         subsections, showing checksum type (MD5/SHA-1/SHA-256) 
                         and hash values.
`info pdb-lines`         Dump C13 line info: section:offset ranges and line 
                         number to offset mappings.
`info pdb-symbols`       Dump raw CodeView symbol records from module streams.
`info pdb-sym-records`   Dump records from the global symbol record stream.
`info pdb-gsi`           Dump GSI (Global Symbol Index) hash table: header, 
                         hash records, bitmap, bucket data. 
`info pdb-psi`           Dump PSGSI (Public Symbol Index) hash table with 
                         embedded GSI hash and address map.
`info pdb-locations`     Dump resolved variable location batons (ranges, 
                         register/offset, gaps). Requires `modi=N`; optional 
                        `symbol=NAME` to filter.

## GDB Integration

### Entry: `coffread.c`

`coff_symfile_read()` calls `pdb_initialize_objfile(objfile)` (guarded by
`#ifdef PDB_FORMAT_AVAILABLE`) before the DWARF initialization path. This is
the single hook that connects the PDB reader to GDB's object-file machinery.

### Entry: `main.c`

Adds the `--pdb-path <path>` CLI option, which sets `pdb_file_override` —
used by the PDB locator as the highest-priority search path for the main
executable's PDB.


## Initialization Order

`pdb_initialize_objfile()` is called from `coff_symfile_read()`. It calls
`pdb_read_pdb_file()` followed by `pdb_expand_all_modules()`.

`pdb_read_pdb_file()` loads data in this sequence:

1. Validate MSF header (magic, block size, block count, directory location).
   `pdb_load_pdb_file()` → `pdb_find_pdb_file()` locates the PDB on disk;
   `pdb_read_msf_header()` reads the 64-byte SuperBlock.

2. Parse stream directory (maps streams → blocks).
   `pdb_read_stream_directory()` → `pdb_read_file_bytes()` assembles
   directory blocks; fills `stream_sizes[]` and `stream_blocks[][]`.

3. Read `/names` stream (global string table for filenames).
   `pdb_read_names_stream()` → `pdb_read_stream()` loads stream 1; walks
   the named-stream hash map to find `/names`; reads it into
   `pdb->string_table`.

4. Parse DBI stream (module headers, stream indices for GSI/PSGSI/SymRec).
   `pdb_read_dbi_stream()` → `pdb_read_stream()` caches stream 3; two-pass
   walk counts modules then fills `pdb->modules[]` with stream numbers,
   sym/C11/C13 sizes, section contributions, and module names.

5. Parse File Info substream (per-module file lists).
   `pdb_read_file_info_substream()` precomputes per-module file-name offset
   pointers into the Names Buffer.

6. Read and parse TPI stream (type records indexed for on-demand resolution).
   `pdb_read_tpi_stream()` → `pdb_parse_tpi_records()` walks raw record
   bytes, fills indexed `pdb_tpi_type[]` array (leaf, length, data pointer
   into cached stream — no copy); allocates type cache on obstack.

7. Read PE section addresses from BFD (for section:offset → PC mapping).
   `pdb_read_sections_from_bfd()` builds `pdb->section_addresses[]` used by
   `pdb_map_section_offset_to_pc()` throughout.

8. Cache Symbol Record Stream.
   `pdb_read_sym_record_stream()` → `pdb_read_stream()` stores result in
   `pdb->sym_record_data` for later GSI/PSGSI lookups and global symbol
   loading.

9. Build minimal symbols from PSGSI.
   `pdb_build_minsyms()` → `pdb_read_stream()` loads the PSGSI stream;
   `pdb_parse_gsi_hash()` parses the public symbol hash; walks hash records
   into Symbol Record Stream to read S_PUB32 records; creates GDB minimal
   symbols via `minimal_symbol_reader`.

10. Register PDB for `info` commands.
    `pdb_register_loaded_pdb()`.

11. Expand all modules eagerly (lazy loading not yet implemented).
    `pdb_expand_all_modules()` → `pdb_build_module()` for each module:
    - `pdb_read_module_stream()` — load the module's stream data.
    - `pdb_read_module_files()` — cache file list from File Info substream.
    - `pdb_walk_c13_line_blocks()` → callback `pdb_add_symtab_linetable()`
      records line entries; `pdb_convert_path()` converts MSYS2 paths.
    - `pdb_parse_symbols()` — dispatches to per-type handlers
      (`handle_func_sym`, `handle_var_sym`, `handle_local_sym`, etc.);
      each creates GDB `struct symbol` with resolved type via
      `pdb_tpi_resolve_type()`.
    - Finalizes `compunit_symtab` with linetable and block tree.
    Then `pdb_load_global_syms()` loads S_GDATA32/S_LDATA32/S_CONSTANT
    from the Symbol Record Stream into a `<pdb-globals>` compilation unit.

## Build & Configuration

- `gdb/configure.ac` / `gdb/configure` — `--enable-gdb-pdb-support`; default
  enabled on mingw/cygwin, disabled elsewhere. Defines `PDB_FORMAT_AVAILABLE`.
- `gdb/Makefile.in` — adds `pdb` subdir; defines `PDB_SRCS` / `PDB_OBS`.
- `gdb/config.in` — `PDB_FORMAT_AVAILABLE` macro.

## Limitations / TODO

- No Windows x64 calling convention support.
- No struct/class/union/enum types. LF_STRUCTURE, LF_CLASS, LF_UNION,
  LF_ENUM are not yet resolved — returns void/unsupported placeholder.
  Variables of these types display as `<unsupported PDB type>`.
- No inline function support.
- Locals only accessible in current frame. `pdb_loclist_read_variable()`
  reads variables from live registers, so only the innermost frame (frame #0)
  is supported.  `up`/`down`/`frame N` need unwinding that is not yet supported.
- CodeView register mapping covers AMD64 GPRs only (RAX–R15, RSP, RBP).
  Other registers are not mapped — variables stored in those registers show as 
  unavailable.  Only x86-64 is supported.
- No IPI stream parsing.
- No language type detection.
- No MSVC name demangling. S_PUB32 records store mangled names, which appear in  
  `info pdb-psi` and minsyms. Module-level records store undecorated names so
  symbols display correct names.
- No PDB symbol server support (placeholder exists in `pdb-path.c`).
- No lazy loading — all modules expanded eagerly at load time.
- GSI table not yet used for lazy symbol lookup.
- Which API to expose to BFD ?
- Heap allocations - review for security - maybe limit the size.
- PDB file checksum checks.
- tests - switch from precompiled exe files to compilation.

## Memory Allocations

Most of the allocations use the objfile obstack.

Non obstack allocations:

**Heap (`new`):**
- `pdb_per_objfile` — registered via `registry<objfile>::key`, auto-deleted
  when objfile is destroyed.
- `buildsym_compunit` —  builder, deleted after modules are 
  `pdb_build_module()` / `pdb_expand_all_modules()`.

**Scoped (`unique_ptr<gdb_byte[]>`):**
- `pdb.c` - reading stream directory and stream block map.  Freed automatically.
- `pdb.c` `pdb_read_stream()` - reading of the actual streams bytes.
  Released into `pdb->stream_data[]` (`pdb` on objstack) or freed automaticaly.
- `pdb-path.c` — temporary buffers for PE executable access.


## Source Files and What They Do

### `pdb.h` — Public API and Data Structures

Defines all public structs, MSF/DBI/CodeView constants, stream indices
(PDB=1, TPI=2, DBI=3, IPI=4), cast/access macros, and the public function
prototypes.

**Key structs and what GDB objects they feed:**

| Struct                | Purpose                          | GDB object produced                      |
|-----------------------|----------------------------------|------------------------------------------|
| `pdb_per_objfile`     | Top-level PDB context: MSF       | `registry<objfile>::key`;                |
|                       | geometry, streams, DBI, modules,  | lifetime = objfile                      |
|                       | sections, TPI, GSI, sym cache    |                                          |
| `pdb_module_info`     | Per-module: stream number,       | Drives `compunit_symtab` creation        |
|                       | sym/C11/C13 sizes, section       |                                          |
|                       | contribution, file lists         |                                          |
| `pdb_tpi_context`     | Parsed TPI: record array + cache | `struct type *` cache (per type index)   |
| `pdb_tpi_type`        | Single raw TPI record: leaf,     | Input to type resolution                 |
|                       | length, data ptr into stream     |                                          |
| `pdb_loclist_baton`   | Per-symbol location baton:       | Registered with `symbol_computed_ops`    |
|                       | linked list of `pdb_loc_entry`   | on `struct symbol`                       |
| `pdb_loc_entry`       | One DEFRANGE range: start/end    | Consumed by                              |
|                       | PC, register, offset, gaps       | `pdb_loclist_read_variable()`            |
| `pdb_loc_gap`         | Gap within a location entry      | Part of `pdb_loc_entry` chain            |
|                       | (start/end addresses)            |                                          |
| `pdb_rsds_info`       | RSDS record from PE debug dir:   | Used by `pdb_find_pdb_file()`            |
|                       | GUID, age, PDB path              |                                          |
| `pdb_file_info`       | Per-file checksum from C13:      | Source file resolution                   |
|                       | filename, checksum type & data   |                                          |
| `pdb_line_block_info` | Callback data for C13 line walk  | `linetable_entry` via `record_line()`    |
| `pdb_gsi_hdr`         | Parsed GSI hash header           | Drives minimal symbol creation           |
| `CV_FileBlock`,       | On-disk C13 file block and       | Parsed during C13 line walking           |
| `CV_FileChecksum`     | checksum structures              |                                          |
| `CV_LineSection`,     | On-disk C13 line section header  | Parsed during C13 line walking           |
| `CV_Line`             | and line entry                   |                                          |

### `pdb.c` — MSF Parsing, Stream Assembly, Module Expansion

This is the orchestrator. It handles all file I/O, MSF block assembly, stream
parsing, and module-to-symtab expansion.

**Important functions:**

| Function                         | What it does                     | GDB objects created                 |
|----------------------------------|----------------------------------|------------------------------------ |
| `pdb_initialize_objfile()`       | Entry point from COFF reader.    | `quick_symbol_functions`            |
|                                  | Loads PDB, expands modules,      | on the objfile                      |
|                                  | registers quick symbol funcs.    |                                     |
| `pdb_read_pdb_file()`            | Main parsing orchestrator —      | `pdb_per_objfile` (populated)       |
|                                  | calls all sub-readers in order.  |                                     |
| `pdb_read_msf_header()`          | Reads and validates the 64-byte  | —                                   |
|                                  | MSF SuperBlock.                  |                                     |
| `pdb_read_stream_directory()`    | Assembles stream directory from  | —                                   |
|                                  | block-map blocks; fills          |                                     |
|                                  | `stream_sizes[]`, `blocks[][]`.  |                                     |
| `pdb_read_stream()`              | Reads an MSF stream by index,    | `gdb_byte_ptr` (owned buffer)       |
|                                  | assembling blocks into a buffer. |                                     |
|                                  | Core I/O for all other readers.  |                                     |
| `pdb_read_names_stream()`        | Reads PDB Info stream (stream 1),| `pdb->string_table`                 |
|                                  | walks hash map to find `/names`. |                                     |
| `pdb_read_dbi_stream()`          | Parses DBI header and module     | `pdb_module_info[]` array           |
|                                  | info records. Fills              |                                     |
|                                  | `pdb->modules[]`.                |                                     |
| `pdb_read_file_info_substream()` | Parses File Info substream from  | `module->file_name_offsets`,        |
|                                  | DBI: precomputes per-module      | `pdb->file_info_names_buffer`       |
|                                  | file-name offset pointers.       |                                     |
| `pdb_read_sections_from_bfd()`   | Reads PE section VMAs from BFD   | `pdb->section_addresses[]`          |
|                                  | for (section, offset) → PC map.  |                                     |
| `pdb_build_module()`             | Expands one module: reads stream,| `compunit_symtab`, `linetable`,     |
|                                  | parses C13 lines, parses symbols,| `struct block` (global, static,     |
|                                  | creates `compunit_symtab`.       | function, lexical), `blockranges`   |
| `pdb_expand_all_modules()`       | Iterates all modules, calls      | All module `compunit_symtab`s       |
|                                  | `pdb_build_module()` for each,   |                                     |
|                                  | then loads global symbols.       |                                     |
| `pdb_walk_c13_line_blocks()`     | Walks `DEBUG_S_LINES`, resolves  | (callback populates linetable)      |
|                                  | filenames via checksums + string |                                     |
|                                  | table, invokes callback.         |                                     |
| `pdb_add_symtab_linetable()`     | Callback: records line entries.  | `linetable_entry`                   |
|                                  | Converts MSYS2 paths.            | via `record_line()`                 |
| `pdb_find_pdb_file()`            | Locates PDB on disk using RSDS,  | —                                   |
|                                  | CLI override, env vars, registry.|                                     |
| `pdb_read_module_files_c13()`    | Resolves file names from C13     | Per-module file list                |
|                                  | checksums subsection.            |                                     |
| `pdb_map_section_offset_to_pc()` | Converts (section, offset) to    |                                     |
|                                  | relocated `CORE_ADDR`.           |                                     |
| `pdb_convert_path()`             | Converts Unix-style paths to     | —                                   |
|                                  | Windows paths (MSYS2 builds).    |                                     |

**GDB integration objects created by `pdb.c`:**
- `compunit_symtab` — one per module, with linetables
- `struct block` — global block, static block (with `blockranges` for
  scattered functions), function blocks, lexical blocks
- `quick_symbol_functions` (`pdb_readnow_functions`) — registered on the
  objfile for symbol lookup, `find_pc_sect_compunit_symtab`, etc.

### `pdb-read-types.c` — TPI Stream Parsing & Type Resolution

Reads the TPI stream header, builds an indexed array of `pdb_tpi_type` records
(pointers directly into cached stream data — no copy), and resolves type
indices to GDB `struct type` on demand with caching.

**Important functions:**
| Function                          | What it does                    | GDB objects created                 |
|-----------------------------------|---------------------------------|------------------------------------ |
| `pdb_read_tpi_stream()`           | Reads TPI header, validates     | `pdb_tpi_type[]` array,             |
|                                   | version, calls                  | `type_cache[]`                      |
|                                   | `pdb_parse_tpi_records()`,      |                                     |
|                                   | allocates type cache.           |                                     |
| `pdb_parse_tpi_records()`         | Walks raw record bytes, fills   | `pdb_tpi_type[]` (on obstack)       |
|                                   | entries (leaf, length, data     |                                     |
|                                   | pointer, data_len).             |                                     |
| `pdb_tpi_resolve_type()`          | Public entry point: resolves a  | `struct type *`                     |
|                                   | type index to `struct type`.    |                                     |
| `pdb_tpi_resolve_type_internal()` | Internal recursive resolver.    | `struct type *` (cached)            |
|                                   | Dispatches to simple decoder or |                                     |
|                                   | leaf-specific makers. Cached.   |                                     |
| `pdb_tpi_resolve_simple_type()`   | Decodes simple indices          | `struct type *`                     |
|                                   | (< 0x1000): kind → builtin,    | (builtin or pointer-to-builtin)      |
|                                   | mode → optional pointer wrap.   |                                     |
| `pdb_tpi_make_modifier()`         | LF_MODIFIER: wraps base type    | `struct type *`                     |
|                                   | with const/volatile.            | via `make_cv_type()`                |
| `pdb_tpi_make_pointer()`          | LF_POINTER: creates pointer,    | `struct type *`                     |
|                                   | lvalue-ref, or rvalue-ref.      | via `lookup_pointer_type()`         |
| `pdb_tpi_make_array()`            | LF_ARRAY: creates array type    | `struct type *`                     |
|                                   | with element type, index type,  | via `create_array_type()`           |
|                                   | and total size.                 |                                     |
| `pdb_tpi_make_procedure()`        | LF_PROCEDURE: function type     | `struct type *`                     |
|                                   | with return type and arglist.   | via `lookup_function_type()`        |
| `pdb_tpi_make_mfunction()`        | LF_MFUNCTION: member function   | `struct type *`                     |
|                                   | (same as procedure for now).    | via `lookup_function_type()`        |
| `pdb_tpi_make_struct()`           | LF_STRUCTURE / LF_CLASS:        | `struct type *` (TYPE_CODE_STRUCT), |
|                                   | struct/class with fields, base  | `struct field[]`,                   |
|                                   | classes, nested types, methods. | `cplus_struct_type`                 |
| `pdb_tpi_make_union()`            | LF_UNION: creates union type.   | `struct type *` (TYPE_CODE_UNION)   |
| `pdb_tpi_make_enum()`             | LF_ENUM: creates enum type with | `struct type *` (TYPE_CODE_ENUM),   |
|                                   | underlying type and enumerators.| enumerator `struct field[]`         |
| `pdb_tpi_parse_fieldlist()`       | Parses LF_FIELDLIST sub-records | `struct field[]` (members, bases,   |
|                                   | (LF_MEMBER, LF_ENUMERATE,       | enumerators), `struct decl_field[]` |
|                                   | LF_BCLASS, LF_STMEMBER,         | (nested types),                     |
|                                   | LF_NESTTYPE, LF_ONEMETHOD,      | `fn_fieldlist[]` + `fn_field[]`     |
|                                   | LF_METHOD, LF_INDEX cont.).     | (member functions)                  |
| `pdb_tpi_resolve_arglist()`       | Resolves LF_ARGLIST: populates  | `struct field[]` on function type   |
|                                   | function type's param fields.   |                                     |
| `pdb_tpi_get_func_param_count()`  | Returns param count from        | —                                   |
|                                   | LF_PROCEDURE or LF_MFUNCTION.   |                                     |
| `pdb_cv_read_numeric()`           | Decodes CodeView variable-len   | —                                   |
|                                   | numeric leaves (LF_CHAR through |                                     |
|                                   | LF_UQUADWORD).                  |                                     |

**Type caching:** A `struct type **` array (0x10000 entries, 512 KB on
64-bit) covers indices 0x0000–type_idx_end. Both simple and compound types
share this cache. Each index is resolved at most once; self-referencing
types (e.g. `struct Node { Node *next; }`) work because the type is cached
before its field list is parsed.

### `pdb-read-symbols.c` — CodeView Symbol Parsing & Location Handling

Parses CodeView symbol records from module streams and the global symbol record
stream. Creates GDB symbols with resolved types and locations.

**Important functions:**

| Function                       | What it does                    | GDB objects created              |
|--------------------------------|---------------------------------|----------------------------------|
| `pdb_parse_symbols()`          | Main per-module symbol parser.  | `struct symbol`, `struct block`  |
|                                | Walks records, dispatches to    | (function, lexical), scope tree  |
|                                | handlers, manages scope stack.  |                                  |
| `pdb_load_global_syms()`       | Reads S_GDATA32/S_LDATA32/      | `struct symbol`                  |
|                                | S_CONSTANT from SymRecordStream,| (global variables, constants)    |
|                                | deduplicates, creates symbols.  |                                  |
| `pdb_build_minsyms()`          | Reads PSGSI hash → SymRecord →  | `minimal_symbol` via             |
|                                | S_PUB32; creates minsyms.       | `minimal_symbol_reader`          |
| `pdb_read_sym_record_stream()` | Caches Symbol Record Stream for | `pdb->sym_record_data`           |
|                                | later GSI/PSGSI lookups.        |                                  |
| `pdb_parse_sym_record_stream()`| Parses/dumps the SymRecord      | (used by `info pdb-sym-records`) |
|                                | stream for info commands.       |                                  |
| `pdb_parse_gsi_hash()`         | Parses GSI hash header for dump | `pdb_gsi_hdr`                    |
|                                | commands and minsym building.   |                                  |
| `pdb_init_gsi_table()`         | Parses GSI stream into a hash   | GSI lookup table                 |
|                                | table for name-based lookup.    |                                  |
| `pdb_loclist_read_variable()`  | Runtime: walks `pdb_loc_entry`  | `struct value *`                 |
|                                | chain for current PC, reads     |                                  |
|                                | register+offset. Registered     |                                  |
|                                | with `symbol_computed_ops`.     |                                  |
| `pdb_init_loclist()`           | Registers PDB location ops at   | `pdb_loclist_index`              |
|                                | GDB startup.                    |                                  |
| `cv_reg_to_gdb_regnum()`       | Maps CodeView register IDs to   | —                                |
|                                | GDB regnums via DWARF mapping.  |                                  |

The `pdb_sym` wrapper structs are stack-allocated during parsing — they
exist only long enough to extract fields from the raw record and call
`create_gdb_sym()`, which creates the obstack-allocated GDB symbol.

**Per-symbol-type handlers** (called by `pdb_parse_symbols()`):

| Handler                  | Symbol type              | GDB result                        |
|--------------------------|--------------------------|-----------------------------------|
| `handle_func_sym`        | S_GPROC32/S_LPROC32      | `struct symbol` (LOC_BLOCK)       |
|                          |                          |   + pushes function scope         |
| `handle_var_sym`         | S_GDATA32/S_LDATA32     | `struct symbol` (LOC_STATIC)       |
| `handle_local_sym`       | S_LOCAL                  | `struct symbol` (LOC_COMPUTED)    |
|                          |                          |   with empty baton                |
| `handle_regrel_sym`      | S_REGREL32               | `struct symbol` (LOC_COMPUTED)    |
|                          |                          |   with full-scope baton           |
| `handle_register_sym`    | S_REGISTER               | `struct symbol` (LOC_REGISTER)    |
| `handle_constant_sym`    | S_CONSTANT               | `struct symbol` (LOC_CONST)       |
| `handle_udt_sym`         | S_UDT                    | `struct symbol` (LOC_TYPEDEF)     |
| `handle_block_sym`       | S_BLOCK32                | Pushes lexical scope →            |
|                          |                          |   `struct block`                  |
| `handle_scope_end_sym`   | S_END/S_INLINESITE_END   | Pops scope → finalizes block      |
|                          |                          |   via `finish_block()`            |
| `handle_defrange_*`      | S_DEFRANGE_*             | Appends `pdb_loc_entry` to        |
|                          |                          |   most recent S_LOCAL's baton     |
| `handle_frameproc_sym`   | S_FRAMEPROC              | Extracts frame pointer register   |
|                          |                          |   for current function scope      |

**Location model:** Unlike DWARF which uses `DW_OP_*` expressions, PDB uses
`S_DEFRANGE_*` records that directly specify register + offset + PC range +
gaps. All location data is stored in `pdb_loclist_baton` → `pdb_loc_entry`
chains, served through GDB's `symbol_computed_ops` interface. Variables are
only reliably readable in the current frame (frame #0) because no PDB unwinder
exists yet.

### `pdb-path.c` — PDB File Discovery

Searches for the PDB file using multiple strategies in priority order:

1. `--pdb-path` CLI override (main executable only)
2. RSDS basename in the EXE directory
3. `<exe-name>.pdb` next to the executable
4. Full RSDS embedded path
5. `_NT_ALT_SYMBOL_PATH` / `_NT_SYMBOL_PATH` (semicolon-separated;
   `SRV*`/`SYMSRV*`/`CACHE*` prefixes recognized but fetch is stubbed)
6. Windows registry `Software\Microsoft\VisualStudio\MSPDB\SymbolSearchPath`
   (HKCU then HKLM)

Also contains `pdb_read_rsds_info()` which reads the RSDS debug directory
entry from the PE executable to extract GUID, age, and PDB path.

### `pdb-cmd.c` — Diagnostic / Introspection Commands

Registers all `info pdb-*` commands. Each command accepts optional `path=<pdb>`
and `modi=N` arguments to select a specific PDB or module.

| Command                  | What it shows                                            |
|--------------------------|----------------------------------------------------------|
| `info pdb-loaded-files`  | Paths of all loaded PDB files                            |
| `info pdb-modules`       | Modules with stream numbers, sizes, file counts          |
| `info pdb-files`         | Source files per module (from DBI File Info substream)   |
| `info pdb-files-c13`     | Source files per module (from C13 checksums)             |
|                          |   with hash type/value                                   |
| `info pdb-lines`         | C13 line info: section:offset ranges and line mappings   |
| `info pdb-symbols`       | Raw CodeView symbol records from module streams          |
| `info pdb-sym-records`   | Records from the global Symbol Record Stream             |
| `info pdb-gsi`           | GSI hash table dump                                      |
| `info pdb-psi`           | PSGSI hash table dump with address map                   |
| `info pdb-locations`     | Variable location batons (ranges, register/offset, gaps) |

### `pdb-cv-regs-amd64.h` — AMD64 Register Mapping

Maps CodeView register IDs (from Microsoft's `cvconst.h`) to DWARF register
numbers, which are then converted to GDB register numbers via
`gdbarch_dwarf2_reg_to_regnum()`. Currently covers GPRs (RAX–R15, RSP, RBP).
XMM and other registers are not yet mapped.
