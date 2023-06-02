# Changelog for ROCgdb

Full documentation for ROCgdb is available at
[docs.amd.com](https://docs.amd.com/bundle/rocgdb)

## (Unreleased) ROCgdb-13.2

### Added
- Support for watchppoints on scratch memory addresses.

### Optimized
- Improved performances when handling the end of a process with a large
  number of threads.
### Known Issues
- On certain configurations, ROCgdb can show the following warning message:

    warning: Probes-based dynamic linker interface failed.
    Reverting to original interface.

  This does not affect ROCgdb's functionalities.
