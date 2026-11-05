/* This testcase is part of GDB, the GNU debugger.

   Copyright 2026 Free Software Foundation, Inc.

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

/* Helper to reproduce getcwd failure by deleting cwd via child process.  */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>

int
main (int argc, char *argv[])
{
  const char *temp_dir = argv[1];
  const char *gdb_path = argv[2];

  /* Create and change to temp directory.  */
  if (mkdir (temp_dir, 0755) != 0)
    {
      perror ("mkdir failed");
      return 1;
    }

  if (chdir (temp_dir) != 0)
    {
      perror ("chdir failed");
      return 1;
    }

  /* child deletes directory, parent execs GDB.  */
  pid_t pid = fork ();
  if (pid < 0)
    {
      perror ("fork failed");
      return 1;
    }

  if (pid == 0)
    {
      /* cd to / and delete the temp directory.  */
      if (chdir ("/") != 0)
	exit (1);

      if (rmdir (temp_dir) != 0)
	exit (1);

      exit (0);
    }

  /* wait for child to delete directory.  */
  int status;
  waitpid (pid, &status, 0);

  if (!WIFEXITED (status) || WEXITSTATUS (status) != 0)
    {
      fprintf (stderr, "Failed to delete directory\n");
      return 1;
    }

  /* Exec GDB.  */
  char **gdb_args = malloc ((argc - 1) * sizeof (char *));
  gdb_args[0] = (char *) gdb_path;
  for (int i = 3; i < argc; i++)
    gdb_args[i - 2] = argv[i];
  gdb_args[argc - 2] = NULL;

  execv (gdb_path, gdb_args);
  perror ("execv failed");
  return 1;
}
