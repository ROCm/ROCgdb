/* Helper to give local IO capabilities to a target.

   Copyright (C) 2023-2024 Free Software Foundation, Inc.
   Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "defs.h"
#include "local-fileio-target.h"
#include "gdbsupport/fileio.h"
#include "gdbsupport/filestuff.h"

/* See local-fileio-target.h.  */

int
local_fileio_open (struct inferior *inf, const char *filename, int flags,
		   int mode, int warn_if_slow, fileio_error *target_errno)
{
  int nat_flags;
  mode_t nat_mode;
  int fd;

  if (fileio_to_host_openflags (flags, &nat_flags) == -1
      || fileio_to_host_mode (mode, &nat_mode) == -1)
    {
      *target_errno = FILEIO_EINVAL;
      return -1;
    }

  fd = gdb_open_cloexec (filename, nat_flags, nat_mode).release ();
  if (fd == -1)
    *target_errno = host_to_fileio_error (errno);

  return fd;
}

/* See local-fileio-target.h.  */

int
local_fileio_pwrite (int fd, const gdb_byte *write_buf, int len,
		     ULONGEST offset, fileio_error *target_errno)
{
  int ret;

#ifdef HAVE_PWRITE
  ret = pwrite (fd, write_buf, len, (long) offset);
#else
  ret = -1;
#endif
  /* If we have no pwrite or it failed for this file, use lseek/write.  */
  if (ret == -1)
    {
      ret = lseek (fd, (long) offset, SEEK_SET);
      if (ret != -1)
	ret = write (fd, write_buf, len);
    }

  if (ret == -1)
    *target_errno = host_to_fileio_error (errno);

  return ret;
}

/* See local-fileio-target.h.  */

int
local_fileio_pread (int fd, gdb_byte *read_buf, int len, ULONGEST offset,
		    fileio_error *target_errno)
{
  int ret;

#ifdef HAVE_PREAD
  ret = pread (fd, read_buf, len, (long) offset);
#else
  ret = -1;
#endif
  /* If we have no pread or it failed for this file, use lseek/read.  */
  if (ret == -1)
    {
      ret = lseek (fd, (long) offset, SEEK_SET);
      if (ret != -1)
	ret = read (fd, read_buf, len);
    }

  if (ret == -1)
    *target_errno = host_to_fileio_error (errno);

  return ret;
}

/* See local-fileio-target.h.  */

int
local_fileio_fstat (int fd, struct stat *sb, fileio_error *target_errno)
{
  int ret;

  ret = fstat (fd, sb);
  if (ret == -1)
    *target_errno = host_to_fileio_error (errno);

  return ret;
}

/* See local-fileio-target.h.  */

int
local_fileio_close (int fd, fileio_error *target_errno)
{
  int ret;

  ret = ::close (fd);
  if (ret == -1)
    *target_errno = host_to_fileio_error (errno);

  return ret;
}

/* See local-fileio-target.h.  */

int
local_fileio_unlink (struct inferior *inf, const char *filename,
		     fileio_error *target_errno)
{
  int ret;

  ret = unlink (filename);
  if (ret == -1)
    *target_errno = host_to_fileio_error (errno);

  return ret;
}

/* See local-fileio-target.h.  */

std::optional<std::string>
local_fileio_readlink (struct inferior *inf, const char *filename,
		       fileio_error *target_errno)
{
  /* We support readlink only on systems that also provide a compile-time
     maximum path length (PATH_MAX), at least for now.  */
#if defined (PATH_MAX)
  char buf[PATH_MAX];
  int len;

  len = readlink (filename, buf, sizeof buf);
  if (len < 0)
    {
      *target_errno = host_to_fileio_error (errno);
      return {};
    }

  return std::string (buf, len);
#else
  *target_errno = FILEIO_ENOSYS;
  return {};
#endif
}
