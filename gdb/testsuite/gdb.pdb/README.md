# gdb.pdb — the PDB reader test suite

Tests for GDB's Microsoft PDB debug-format reader (`gdb/pdb/`). Tests build
their programs during the test, with the selected DejaGnu board or with
Microsoft cl. Generated files stay in the individual test's output
directory, not the source tree.

No executable/PDB pairs or batch/shell rebuild utilities are stored here.

## Running

```bash
cd build/gdb/testsuite
HIP_SDK='C:/AMD/ROCm/7.2' \
  runtest --tool gdb --target_board=hip-sdk-pdb 'gdb.pdb/*.exp'
```

`HIP_SDK` names the HIP SDK installation. Do not pass `GDB=` without
`GDB_DATA_DIRECTORY`: GDB then runs without its data directory.

Tests that need `--readnow` coverage add it themselves.

### Selftests

The reader's format checks also run as GDB selftests, on any host and
without a compiler: `maint selftest pdb-` runs them, and
`gdb.gdb/unittest.exp` runs them with the rest of the selftests. They build
small PDBs, records and paths in memory and check each rule the reader
applies to the sizes, counts and names it reads from them. A `.exp` here
checks what a selftest cannot: that GDB keeps running when a PDB is
rejected, and what loading prints.

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

### Microsoft cl builds

Some tests also build their program with Microsoft cl, through
`pdb_cl_prepare_for_testing`: cl compiles with `/Z7` and the optimization the
test names, and LLVM's `lld-link` or Microsoft `link` links. cl and
link run in the environment that Visual Studio's `vcvars64.bat` sets up.
`PDB_VCVARS` names that script; without it `vswhere` finds the newest
installation with the C++ tools. `lld-link` is looked for in `LLVM_BIN`,
`HIP_SDK` and `HIP_PATH`, in the LLVM and ROCm installations, then on
`PATH`. With no cl the cl leg is unsupported, as is an `lld-link` leg with
no `lld-link`; a cl build that fails is untested. These legs catch a cl
release that records something the other tests do not cover, and what link
writes differently from lld-link.

### Board choices and known gaps

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

The tests include source-compiled sessions and PDB discovery/identity
checks.

## 1.1 Compiled-fixture tests

Every compiled test uses `pdb_prepare_for_testing` with the selected board.

| Test | Loads | Named source | Inferior | Covers |
|---|---|---|---|---|
| [`pdb-load.exp`](pdb-load.exp) | built at test time | `pdb-test.cpp` | no | A program the selected board compiled gets its PDB loaded |

## 1.2 Which PDB gets loaded

Where the reader looks for a PDB, and whether it accepts the one it finds,
are checked by the selftests `pdb-find-pdb-file`, `pdb-rsds-info` and
`pdb-validate-guid`, on PE images and PDBs they build in memory. The tests
below load what a compiler produced.

| Test | Covers |
|---|---|
| [`pdb-cl-load.exp`](pdb-cl-load.exp) | A program Microsoft cl compiled from `pdb-test.cpp` gets its PDB loaded, whether lld-link or Microsoft link wrote it |
| [`pdb-separate-debug.exp`](pdb-separate-debug.exp) | An executable whose DWARF is in a separate debug file, and whose RSDS record names a PDB, is read from the separate debug file: the PDB lld-link writes for a DWARF build holds only the C runtime's CodeView |

---

# 2. Supporting files

| File | Purpose |
|---|---|
| [`lib-pdb.exp`](../lib/lib-pdb.exp) | Shared support probe, the compiled-test helper and the Microsoft cl builds |
| [`hip-sdk-pdb.exp`](../boards/hip-sdk-pdb.exp) | Tcl compiler adapter using `HIP_SDK/bin/clang-cl.exe` in native mode |

Reader-side documentation lives in `gdb/pdb/README.md` (stream layout, init
order, supported records).

---

# 3. Adding a compiled fixture

Add source files and an `.exp` using `standard_testfile` and
`pdb_prepare_for_testing`, as "Per-test compiler options" above says. Supply
all translation units as a source list. Choose the optimization level in
the test; the board supplies the compiler and default debug/link flags.
Each consumer builds its own output, even when sources are shared.

Use source markers with `gdb_get_line_number` for line-dependent assertions.
Changes to shared sources still require running their consumers: statement
ordering, variable lifetimes and inlining affect the expected stops and values.
