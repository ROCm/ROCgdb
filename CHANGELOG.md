# Changelog for ROCgdb

Full documentation for ROCgdb is available at
[rocm.docs.amd.com/rocgdb](https://docs.amd.com/bundle/rocgdb)

## ROCgdb-16

### Added

- When opening a core dump, GDB can use the `sysroot` or `solib-search-path`
  settings to locate files containing GPU code objects. This allows opening
  GPU code objects on systems different from the one where the core dump was
  generated.

- Support for precise ALU exception reporting for supported architectures. Precise ALU exceptions reporting is controlled with the following commands:
  - set amdgpu precise-alu-exceptions
  - show amdgpu precise-alu-exceptions

## (Unreleased) ROCgdb-14
### Added
- Introduce the `coremerge` utility to merge a host core dump and a GPU-only
  AMDGPU core dump into a unified AMDGPU corefile.
- Support for generating and opening core files for heterogeneous processes.

## (Unreleased) ROCgdb-13

### Added
- Support for watchpoints on scratch memory addresses.
- Add support for gfx1100, gfx1101, and gfx1102.
- Added support for gfx940, gfx941 and gfx942.

### Optimized
- Improved performances when handling the end of a process with a large
  number of threads.
### Known Issues
- On certain configurations, ROCgdb can show the following warning message:

    warning: Probes-based dynamic linker interface failed.
    Reverting to original interface.

  This does not affect ROCgdb's functionalities.
- ROCgdb cannot debug a program on an AMDGPU device past a `s_sendmsg
  sendmsg(MSG_DEALLOC_VGPRS)` instruction.  If an exception is reported
  after this instruction has been executed (including asynchronous
  exceptions), the wave is killed and the exceptions are only reported by
  the ROCm runtime.
