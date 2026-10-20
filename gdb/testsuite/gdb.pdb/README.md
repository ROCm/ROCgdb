# gdb.pdb — the PDB reader test suite

Tests for GDB's Microsoft PDB debug-format reader (`gdb/pdb/`). Source-compiled
tests use the selected DejaGnu board. YAML tests generate their PDBs at test
time using `llvm-pdbutil`, `llvm-readobj` and `python3`. Generated files stay in
the individual test's output directory, not the source tree.

No executable/PDB pairs or batch/shell rebuild utilities are stored here.

Two kinds of input:

- **Compiled** — matching executable/debug-info pairs, built by the selected
  compiler during the test.
- **YAML** — PDBs *assembled* from hand-written CodeView records by
  `llvm-pdbutil yaml2pdb`, not compiled from source. These test the reader
  against authored records, including unusual and malformed inputs independent
  of compiler emission choices. Each test builds a companion PE from
  `pdb-yaml-stub.c` in its output directory. The YAML sources are committed;
  generated PDBs and executables are not. Synthetic symbols do not describe
  the stub's machine code, so these tests provide load-only coverage.

## Running

```bash
cd build/gdb/testsuite
HIP_SDK='C:/AMD/ROCm/7.2' \
  runtest --tool gdb --target_board=hip-sdk-pdb 'gdb.pdb/*.exp'
```

`HIP_SDK` names the HIP SDK installation. Do not pass `GDB=` without
`GDB_DATA_DIRECTORY`: GDB then runs without its data directory.

Tests that need `--readnow` coverage add it themselves.

### Per-test compiler options

The selected board supplies the compiler and debug format. Each `.exp`
chooses the optimization level required by its source and assertions
(`optimize=-O0` or `optimize=-O2`) in the options it passes to
`pdb_prepare_for_testing`, which builds with `prepare_for_testing` and checks
that GDB loaded the generated PDB. Use distinct output names if a single test
deliberately builds more than one variant.

Compilation or failure to load the generated PDB returns nonzero and must stop
the caller. Merely having PDB support in GDB is not enough: a `unix` board run
using GCC/DWARF does not exercise these compiled PDB tests. The `hip-sdk-pdb`
board's Tcl compile hook translates library-search and optimization arguments for
native `clang-cl`; it never chooses a debug format or optimization level itself.

### Board TODO

The board makes these choices for the shared `gdb.base`/`gdb.cp` tests:

- A C++ compile that names no standard is built as C++17, GCC's default
  (clang-cl defaults to C++14).
- `-std=c++98/03/11` is compiled as C++14, because the MSVC STL headers
  need it.  A source that relies on C++03 itself, such as `casts03.cc`
  naming a function `decltype`, no longer compiles.
- `-Wl,-e,SYMBOL` is dropped: the CRT startup must stay the entry point,
  since it pulls in the C runtime.
- `-nostdlib` adds `-GS-` only when a source is compiled; on a link-only
  command clang-cl warns the option unused, and the warning fails the build
  (`gdb.base/list-dot-nodebug.exp`).

Not done yet:

- Compiling to assembly (`gdb_compile ... assembly`) passes DejaGnu's `-S`,
  which clang-cl ignores (`gdb.base/tailcall-msym.exp`).
- `nopie` is ignored, so every image loads at a random address.  Tests that
  read an address before `run` and expect it afterwards fail
  (`gdb.base/maint-info-inline-frames-and-blocks.exp`).  lld's
  `-dynamicbase:no` is the candidate; it has not been tested here.
- A test that declares its own `printf` (`gdb.cp/virtfunc.exp`) links only
  with `legacy_stdio_definitions.lib`, which the board does not add.
- Objects compiled with `debug` and then linked without it lose their
  CodeView: lld writes a PDB only for `-debug`, and the board adds only
  `-debug:symtab` (`gdb.cp/infcall-nodebug-c-d1.exp`, `c++-d1`,
  `gdb.base/break-on-linker-gcd-function.exp`).  DWARF sections survive such
  a link.  The link should get `-debug` when an input object carries
  CodeView.
- Windows will not replace an image GDB holds, so `gdb_compile` exits GDB
  before relinking.  A test that recompiles and keeps using the session
  then talks to a closed spawn (`gdb.base/cached-source-file.exp`).
- An object compiled to an extension-less name (`-c -o .../NAME`) is
  written by clang-cl as `NAME.obj`, so the test's `file NAME` finds nothing
  (`gdb.cp/expand-psymtabs-cxx.exp`, `gdb.base/readline-ask.exp`).  The
  board could rename the output to the requested name.

Testsuite library and tests (not the board):

- Many tests expect `Breakpoint N, ...` right after `Continuing.`; a Windows
  process has runtime worker threads, so GDB prints `[New Thread ...]` and
  `Thread 1 hit Breakpoint N, ...` (`gdb.cp/ovldbreak.exp`,
  `gdb.base/infcall-failure.exp`, `func-ptrs.exp`).  `gdb_continue_to_end`
  already allows `[Thread ... exited]` lines for `*-*-mingw*`; individual
  patterns do not.
- Tests that need GCC's `__GNUC__` attributes (`gdb.opt/inline-bt.exp`,
  `inline-break.exp`) or GCC location views (`gdb.opt/empty-inline*.exp`)
  run under clang-cl and fail; they should report UNSUPPORTED.

---

# 1. Tests

The tests include source-compiled sessions, YAML-authored records and
PDB discovery/identity checks. `break FILE:LINE`
without `run` checks linespec resolution, not execution.

## 1.1 Compiled-fixture tests

Every compiled test uses `pdb_prepare_for_testing` with the selected board.
All of them use `-O0`.

| Test | Loads | Named source | Inferior | Covers |
|---|---|---|---|---|
| [`pdb-simple-info.exp`](pdb-simple-info.exp) | built at test time | `pdb-test.cpp` | yes | End-to-end session: breakpoints, `info sources/functions/line`, `backtrace`, `list`, stepping, `finish`, function types, public symbols |
| [`pdb-psgsi-offset.exp`](pdb-psgsi-offset.exp) | built at test time, then patched | `pdb-yaml-stub.c` | no | A public-symbol hash record whose symbol offset wraps past 2^32 when the record header size is added is rejected, and GDB keeps running |

## 1.2 YAML-fixture tests

Each test calls `pdb_prepare_yaml` to assemble its committed
`pdb-yaml-<name>.in.yaml` into `pdb-yaml-stub.pdb` in its own output directory.
The helper reuses `pdb_llvm_pdbutil` for tool discovery and applies
the section-contribution patcher.
Each test builds its companion PE from [`pdb-yaml-stub.c`](pdb-yaml-stub.c)
and loads it with the freshly generated `pdb-yaml-stub.pdb`. No synthetic
fixture starts an inferior; its symbols do not describe the stub's code.

| Test | Fixture | Covers |
|---|---|---|
| [`pdb-yaml-modbounds.exp`](pdb-yaml-modbounds.exp) | `pdb-yaml-modbounds` | A PC belongs to the module whose code covers it: a module's section contribution above its procedure, and its procedure in the gap between another module's two procedures. Checked through `info line` |
| [`pdb-yaml-modbounds-first.exp`](pdb-yaml-modbounds-first.exp) | `pdb-yaml-modbounds-first` | The same modules with the one-procedure module first, its section contribution covering the other module's procedures: it claims no PC outside its procedure. Checked through `info line` |
| [`pdb-yaml-nochecksums.exp`](pdb-yaml-nochecksums.exp) | `pdb-yaml-modbounds` | Line blocks of a module whose `DEBUG_S_FILECHKSMS` is hidden after generation: the rows keep their lines under an unknown file, and loading prints nothing |
| [`pdb-yaml-header.exp`](pdb-yaml-header.exp) | `pdb-yaml-onefunc` | A TPI header patched with an inverted range, more records than the record bytes hold, or a base other than 0x1000: the header is rejected before allocation and GDB keeps running |
| [`pdb-yaml-proccycle.exp`](pdb-yaml-proccycle.exp) | `pdb-yaml-proccycle` | A procedure type that names itself as its return type: the return type is unsupported and GDB keeps running |
| [`pdb-procedure-owner.exp`](pdb-procedure-owner.exp) | `pdb-yaml-onefunc` | A procedure type returning a builtin belongs to the objfile, and a copy kept by Python survives `symbol-file` |

---

## 1.3 Which PDB gets loaded

These tests arrange generated PE/PDB inputs in their output directories to
check file selection and identity validation. `pdb-guid-mismatch` also builds
`pdb-test.cpp` for its foreign-PDB control; the matching and wrong-age inputs
come from YAML.

Both read the outcome back from `maint info pdb-loaded-files` instead of
matching the load-time message, and both exit GDB before rearranging files: the
reader keeps the PDB mapped for the objfile's lifetime, so the file is busy
while it is loaded.

| Test | Covers |
|---|---|
| [`pdb-discovery.exp`](pdb-discovery.exp) | The two outcomes a user can observe: found beside the executable, and found nowhere — which must leave none loaded rather than fall back to some other file |
| [`pdb-guid-mismatch.exp`](pdb-guid-mismatch.exp) | A PDB under the expected name but belonging to another program is refused; nothing is loaded, no type reaches the symbol tables, and the message names the identity wanted and the one found. The matching PDB is then accepted from that same path |

---

# 2. Supporting files

| File | Purpose |
|---|---|
| [`pdb-yaml-setsc.py`](pdb-yaml-setsc.py) | Patches module 0's DBI module-info section/offset/size in place for module expansion |
| [`pdb-yaml-stub.c`](pdb-yaml-stub.c) | Source of the companion PE, compiled by `pdb_yaml_stub` in the test output directory |
| [`lib-pdb.exp`](../lib/lib-pdb.exp) | Shared support probe and fixture utilities |
| [`hip-sdk-pdb.exp`](../boards/hip-sdk-pdb.exp) | Tcl compiler adapter using `HIP_SDK/bin/clang-cl.exe` in native mode |

Reader-side documentation lives in `gdb/pdb/README.md` (stream layout, init
order) and `gdb/pdb/LLM-WIKI.md` (CodeView record support matrix).

---

# 3. How the YAML fixtures work

## 3.1 Why a companion PE

A standalone `.pdb` cannot be loaded. GDB reaches a PDB through the PE's
CodeView **RSDS** record and validates GUID/age before using it. Each test
builds a stub PE and stamps its generated PDB with the identity read from
that executable. The stub's RSDS names `pdb-yaml-stub.pdb`, so the generated
PDB uses that basename next to the executable.

The PDB supplies section-relative offsets; GDB combines them with section
addresses from the PE. Address-bearing fixtures use segment 1, the stub's
`.text`; the synthetic functions do not have matching executable bodies.

## 3.2 Regenerating

Running a YAML test regenerates its PDB in that test's output directory.
`pdb_yaml_stub` compiles the companion PE if it is absent there. The helper
reads its GUID/age with `llvm-readobj`, substitutes the YAML placeholders,
runs `yaml2pdb`, and applies the module-contribution patch.
The third argument of `pdb_prepare_yaml` is the number of leading modules
given a contribution (default 1; they split the first 4096 bytes of `.text`):
2 for `pdb-yaml-modbounds`.
No separate rebuild script is required. Commit fixture sources, not generated
executables or PDBs.

## 3.3 `yaml2pdb` quirks

- **Check diagnostics as well as status.** The generator treats an `error:`
  diagnostic as failure even if `yaml2pdb` exits successfully.
- **`!Lines` blocks require an explicit `Columns: []`.**
- **A module may carry both a line table and symbol records.** `yaml2pdb`
  preserves `Modi: Records` alongside `!Lines` / `!InlineeLines` /
  `!FileChecksums`.

## 3.4 Adding a YAML fixture

1. Write `pdb-yaml-<name>.in.yaml` using the `__GUID__` / `__AGE__`
   placeholders, or `__BAD_AGE__` for an age the companion PE does not have.
2. Write `pdb-yaml-<name>.exp` calling `pdb_prepare_yaml`. The helper detects
   module symbols from `Modi:`.
3. Run the test through the selected PDB board. Commit the source and test,
   not the generated `.pdb`.

## 3.5 Adding a compiled fixture

Add source files and an `.exp` using `standard_testfile` and
`pdb_prepare_for_testing`, as in the per-test options example above. Supply
all translation units as a source list. Choose the optimization level in
the test; the board supplies the compiler and default debug/link flags.
Each consumer builds its own output, even when sources are shared.

Use source markers with `gdb_get_line_number` for line-dependent assertions.
Changes to shared sources still require running their consumers: statement
ordering, variable lifetimes and inlining affect the expected stops and values.
