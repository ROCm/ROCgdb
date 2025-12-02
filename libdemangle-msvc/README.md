# libdemangle-msvc

Microsoft Visual C++ demangling library for ROCgdb.

## Purpose

This library provides MSVC symbol demangling functionality, currently using 
the existing LLVM's MSVC demangler code. It can be used by BFD, GDB and 
binutils to demangle Microsoft Visual C++ symbols.

## Licensing

This library is licensed under Apache 2.0 with LLVM Exception.

## API

### Core Demangling Functions

The library exposes a C API compatible with libiberty's demangling interface:

```c
/* Demangle MSVC symbol name
   Returns the demangled string (caller must free).
   Takes the same options as libiberty's cplus_demangle() 
   for compatibility. */
char *msvc_demangle (const char *mangled, int options);
```

### MSVC-Specific Utility Functions

```c
/* Extract class name from MSVC mangled physical name */
char *llvm_msvc_class_name_from_physname (const char *physname);

/* Extract method name from MSVC mangled physical name */
char *llvm_msvc_method_name_from_physname (const char *physname);
```

### BFD Integration Function

```c
/* BFD unified demangling function
   Automatically detects symbol type and uses appropriate demangler.
   
   Parameters:
     - abfd: BFD object context
     - mangled: The mangled symbol name
     - options: Demangling options (DMGL_* flags)
     - msvc_demangle_fn: Callback function for MSVC demangling
   
   Returns: Demangled string (caller must free), or NULL if demangling failed
*/
char *bfd_demangle_new (bfd *abfd, const char *mangled, int options,
                        char *(*msvc_demangle_fn)(const char *, int));
```

**Example usage:**
```c
char *demangled = bfd_demangle_new(abfd, symbol_name, DMGL_PARAMS, msvc_demangle);
if (demangled) {
    printf("%s\n", demangled);
    free(demangled);
}
```

## Integration

The library is built as a separate static library, libdemangle_msvc, 
which is then linked into GDB and binutils.

msvc_demangle is invoked from within bfd_demangle_new by checking the 
mangled name prefix and determining whether the symbol is in MSVC format.
msvc_demangle is provided to bfd_demangle_new as a callback in order to 
avoid linking libbfd with libdemangle_msvc unless it is necessary. 
Only the tools that require libdemangle_msvc will pass this callback.

bfd_is_msvc_symbol is an internal function that determines whether 
a mangled name is in MSVC format. A symbol is considered MSVC-style if:

- starts with ?
- starts with a combination of . and/or $ characters followed by ?
  (follows the existing bfd_demangle behavior, which strips these characters)
- matches $ANYTHING$?, which represents MSVC compiler-generated prefixes.

The decision of whether a symbol is MSVC-style is kept inside BFD in case 
future logic needs to examine the binary further (for example, stripping 
additional prefixes or suffixes, or checking the target triplet).

### Architecture Overview

```
┌─────────────────┐
│  Binutils Tool  │  (nm, addr2line, objdump, etc.)
│  or GDB         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ bfd_demangle_new│  Detects symbol type (MSVC vs Itanium)
└────────┬────────┘
         │
    ┌────┴──────┐
    ▼           ▼
┌─────────┐  ┌────────────────┐
│libiberty│  │libdemangle_msvc│
│(Itanium)│  │(MSVC/LLVM)     │
└─────────┘  └────────────────┘
```

### Symbol Format Detection

(`bfd_is_msvc_symbol()`) identifies MSVC symbols by:
- Symbol starts with `?` (standard MSVC mangling).
- Symbol starts with combination of `.` and `$` ending with `?` 
  (with prefix stripping).
- Symbol matches `$ANYTHING$?` (compiler-generated prefixes).

The detection logic is kept in BFD to allow future enhancements like:
- Examining the binary's target triplet.
- Additional prefix/suffix stripping based on binary format.
- Platform-specific symbol recognition.

### Callback Architecture

To avoid mandatory linking of `libdemangle-msvc` into `libbfd`, 
`msvc_demangle()` is passed as a **callback** to `bfd_demangle_new()`:
- Tools that don't use MSVC demangling don't link `libdemangle-msvc`
- Tools that support MSVC demangling pass the `msvc_demangle` function pointer
- `bfd_demangle_new()` calls the appropriate demangler based on symbol format.

### Binutils Tool Integration

Tools should use `bfd_demangle_new` instead of `bfd_demangle`

## Standalone Build

```bash
cd libdemangle-msvc
make clean
make
# Creates libdemangle-msvc.a
```

### ROCgdb build:

```bash
make all-libdemangle-msvc
```

## Running the Test Suite

```bash
cd libdemangle-msvc/testsuite
make check
```
