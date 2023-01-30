# Changelog for ROCgdb

Full documentation for ROCgdb is available at
[docs.amd.com](https://docs.amd.com/bundle/rocgdb)

## (Unreleased) ROCgdb-13
### Added
- Add support for gfx1100, gfx1101, and gfx1102.
### Optimized
- Improved performances when handling the end of a process with a large
  number of threads.
### Known Issues
- On certain configurations, ROCgdb can show the following warning message:

    warning: Probes-based dynamic linker interface failed.
    Reverting to original interface.

  This does not affect ROCgdb's functionalities.
- It is not possible to debug programs that use cooperative groups or CU
  masking for gfx1100, gfx1101, and gfx11102.  A restriction will be reported
  when attaching to a process that has already created cooperative group queues
  or CU masked queues.  Any attempt to create a cooperative queue or CU masked
  queue when the debugger is attached will fail.
