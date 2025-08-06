# Changelog for ROCgdb

Full documentation for ROCgdb is available at
[rocm.docs.amd.com/rocgdb](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/index.html)

## ROCgdb-X for ROCm-next

### Added

- GDB now determines the name of AMD GPU threads based on the name of
  their kernel function.
- Support for the HIP language and its built-in variables:
  - `threadIdx`
  - `blockIdx`
  - `blockDim`
  - `gridDim`
  - `warpSize`

## ROCgdb-16-2 for ROCm-7.0

### Added

- Support for the following architectures:
  - `gfx950`
  - `gfx1150`
  - `gfx1151`
- Support for FP4, FP6 and FP8 micro-scaling (MX) data types with the `gfx950`
  architecture.

### Removed

- Support for the gfx940 and gfx941 architectures.

## ROCgdb-15.2 for ROCm-6.4

### Added

- Support for debugging shaders compiled for the following generic targets:
  - `gfx9-generic`
  - `gfx9-4-generic`
  - `gfx10-1-generic`
  - `gfx10-3-generic`
  - `gfx11-generic`
  - `gfx12-generic`

## ROCgdb-15.2 (for ROCm-6.3)

### Added

- Support for gfx1200 and gfx1201 architectures.
- Support for precise ALU exception reporting for supported architectures.
  Precise ALU exceptions reporting is controlled with the following commands:
  - set amdgpu precise-alu-exceptions
  - show amdgpu precise-alu-exceptions

### Changed

- The `sysroot` or `solib-search-path` settings can now be used to locate files
  containing GPU code objects when opening a core dump.  This allows opening
  GPU code objects on systems different from the one where the core dump was
  generated.

### Resolved issues

- Fixed possible hangs when opening some AMDGPU core dumps in ROCgdb.
- Addressed cases where the `roccoremerge` utility improperly handled LOAD
  segment copy from the host core dump to the combined core dump.

## ROCgdb-14.2 (for ROCm-6.2)

### Added

- 'info agents' now prints the agent location as "DDD:BB:DD.F", where "DDDD" is
  the agent's PCI domain.
- Introduce the `coremerge` utility to merge a host core dump and a GPU-only
  AMDGPU core dump into a unified AMDGPU corefile.
- Support for generating and opening core files for heterogeneous processes.

## ROCgdb-13.2 (for ROCm-6.0)

### Added

- Add support for gfx1100, gfx1101, and gfx1102.
- Added support for gfx940, gfx941 and gfx942.

### Known Issues

- ROCgdb cannot debug a program on an AMDGPU device past a `s_sendmsg
  sendmsg(MSG_DEALLOC_VGPRS)` instruction.  If an exception is reported
  after this instruction has been executed (including asynchronous
  exceptions), the wave is killed and the exceptions are only reported by
  the ROCm runtime.

## ROCgdb-13.2 (for ROCm-5.7)

- Support for watchpoints on scratch memory addresses.

## ROCgdb-13.1 (for ROCm-5.6)

### Optimized
- Improved performances when handling the end of a process with a large
  number of threads.

### Known Issues
- On certain configurations, ROCgdb can show the following warning message:

    warning: Probes-based dynamic linker interface failed.
    Reverting to original interface.

  This does not affect ROCgdb's functionalities.
