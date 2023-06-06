/* Helper to give local IO capabilities to a target.

   Copyright (C) 2023 Free Software Foundation, Inc.
   Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LOCAL_FILEIO_TARGET_H
#define LOCAL_FILEIO_TARGET_H

#include "gdbsupport/fileio.h"

/* Generic implementation of target_ops::fileio_open using the local
   filesystem.  */
extern int local_fileio_open (struct inferior *inf, const char *filename,
			      int flags, int mode, int warn_if_slow,
			      fileio_error *target_errno);

/* Generic implementation of target_ops::fileio_pwrite using the local
   filesystem.  */
extern int local_fileio_pwrite (int fd, const gdb_byte *write_buf, int len,
				ULONGEST offset, fileio_error *target_errno);

/* Generic implementation of target_ops::fileio_pread using the local
   filesystem.  */
extern int local_fileio_pread (int fd, gdb_byte *read_buf, int len,
			       ULONGEST offset, fileio_error *target_errno);

/* Generic implementation of target_ops::fileio_fstat using the local
   filesystem.  */
extern int local_fileio_fstat (int fd, struct stat *sb, fileio_error
			       *target_errno);

/* Generic implementation of target_ops::fileio_close using the local
   filesystem.  */
extern int local_fileio_close (int fd, fileio_error *target_errno);

/* Generic implementation of target_ops::fileio_unlink using the local
   filesystem.  */
extern int local_fileio_unlink (struct inferior *inf, const char *filename,
				fileio_error *target_errno);

/* Generic implementation of target_ops::fileio_readlink using the local
   filesystem.  */
extern gdb::optional<std::string> local_fileio_readlink
  (struct inferior *inf, const char *filename, fileio_error *target_errno);

/* Helper class implementing the fileio hooks using the local filesystem.  */

template <typename Target>
class local_fileio_target : public Target
{
  int fileio_open (struct inferior *inf, const char *filename,
		   int flags, int mode, int warn_if_slow,
		   fileio_error *target_errno) override
  {
    return local_fileio_open (inf, filename, flags, mode, warn_if_slow,
			      target_errno);
  }

  int fileio_pwrite (int fd, const gdb_byte *write_buf, int len,
		     ULONGEST offset, fileio_error *target_errno) override
  { return local_fileio_pwrite (fd, write_buf, len, offset, target_errno); }

  int fileio_pread (int fd, gdb_byte *read_buf, int len,
		    ULONGEST offset, fileio_error *target_errno) override
  { return local_fileio_pread (fd, read_buf, len, offset, target_errno); }

  int fileio_fstat (int fd, struct stat *sb, fileio_error *target_errno) override
  { return local_fileio_fstat (fd, sb, target_errno); }

  int fileio_close (int fd, fileio_error *target_errno) override
  { return local_fileio_close (fd, target_errno); }

  int fileio_unlink (struct inferior *inf,
		     const char *filename,
		     fileio_error *target_errno) override
  { return local_fileio_unlink (inf, filename, target_errno); }

  gdb::optional<std::string> fileio_readlink (struct inferior *inf,
					      const char *filename,
					      fileio_error *target_errno) override
  { return local_fileio_readlink (inf, filename, target_errno); }
};

#endif
