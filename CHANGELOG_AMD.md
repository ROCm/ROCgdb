# Changelog for ROCgdb

Full documentation for ROCgdb is available at
[docs.amd.com](https://docs.amd.com/bundle/rocgdb)

## (Unreleased) ROCgdb-13.2

### Added
- Support for watchpoints on scratch memory addresses.
- Add support for gfx1100, gfx1101, and gfx1102.

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
