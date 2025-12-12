/* Handle different target file systems for GDB, the GNU Debugger.
   Copyright (C) 2010-2026 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#ifndef GDB_FILESYSTEM_H
#define GDB_FILESYSTEM_H

#include "gdbsupport/pathstuff.h"

extern const char file_system_kind_auto[];
extern const char file_system_kind_unix[];
extern const char file_system_kind_dos_based[];

extern const char *target_file_system_kind;

/* Same as IS_DIR_SEPARATOR but with file system kind KIND's
   semantics, instead of host semantics.  */

#define IS_TARGET_DIR_SEPARATOR(kind, c)				\
  (((kind) == file_system_kind_dos_based) ? IS_DOS_DIR_SEPARATOR (c) \
   : IS_UNIX_DIR_SEPARATOR (c))

/* Same as IS_ABSOLUTE_PATH but with file system kind KIND's
   semantics, instead of host semantics.  */

#define IS_TARGET_ABSOLUTE_PATH(kind, p)				\
  (((kind) == file_system_kind_dos_based) ? IS_DOS_ABSOLUTE_PATH (p) \
   : IS_UNIX_ABSOLUTE_PATH (p))

/* Same as HAS_DRIVE_SPEC but with file system kind KIND's semantics,
   instead of host semantics.  */

#define HAS_TARGET_DRIVE_SPEC(kind, p)					\
  (((kind) == file_system_kind_dos_based) ? HAS_DOS_DRIVE_SPEC (p) \
   : 0)

/* Same as lbasename, but with file system kind KIND's semantics,
   instead of host semantics.  */
extern const char *target_lbasename (const char *kind, const char *name);

/* The effective setting of "set target-file-system-kind", with "auto"
   resolved to the real kind.  That is, you never see "auto" as a
   result from this function.  */
extern const char *effective_target_file_system_kind (void);

/* Return true if GDB should normalize backslashes to forward slashes.
   This is true if GDB is running on a system with a DOS-based
   filesystem (e.g., Windows), or, if cross debugging and the target
   itself has a DOS-based filesystem (e.g., remote debugging Windows
   GDBserver from Linux).  */
extern bool should_normalize_slashes ();

/* Normalizes backslashes to forward slashes in a path string if
   necessary, extending the lifetime of the normalized copy for the
   duration of the object's scope.

   The constructor takes a pointer-to-pointer and updates *PATH to
   point at the normalized string when normalization is needed, so
   callers need only:

     scoped_normalized_path path_storage (&path);

   and then use PATH directly without a separate reassignment.  */
struct scoped_normalized_path
{
  explicit scoped_normalized_path (const char **path)
  {
    gdb_assert (path != nullptr);
    gdb_assert (*path != nullptr);

    if (should_normalize_slashes ())
      {
	m_normalized = *path;
	normalize_slashes (&m_normalized[0]);
	*path = m_normalized.c_str ();
      }
  }

private:
  /* The normalized version of PATH if normalization was necessary,
     empty otherwise.  */
  std::string m_normalized;
};

/* Like getcwd but normalizes slashes if needed.  */
extern char *gdb_getcwd (char *buf, size_t size);

#endif /* GDB_FILESYSTEM_H */
