/* MI Command Set - MI Command Parser.

   Copyright (C) 2000-2023 Free Software Foundation, Inc.
   Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.

   Contributed by Cygnus Solutions (a Red Hat company).

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

#ifndef MI_MI_PARSE_H
#define MI_MI_PARSE_H

#include "gdbsupport/run-time-clock.h"
#include <chrono>
#include "mi-cmds.h"  /* For enum print_values.  */

/* MI parser */

/* Timestamps for current command and last asynchronous command.  */
struct mi_timestamp
{
  std::chrono::steady_clock::time_point wallclock;
  user_cpu_time_clock::time_point utime;
  system_cpu_time_clock::time_point stime;
};

enum mi_command_type
  {
    MI_COMMAND, CLI_COMMAND
  };

struct mi_parse
  {
    /* Attempts to parse CMD returning a ``struct mi_parse''.  If CMD is
       invalid, an exception is thrown.  For an MI_COMMAND COMMAND, ARGS
       and OP are initialized.  Un-initialized fields are zero.  *TOKEN is
       set to the token, even if an exception is thrown.  It is allocated
       with xmalloc; it must either be freed with xfree, or assigned to
       the TOKEN field of the resultant mi_parse object, to be freed by
       mi_parse_free.  */

    static std::unique_ptr<struct mi_parse> make (const char *cmd,
						  char **token);

    /* Create an mi_parse object given the command name and a vector
       of arguments.  Unlike with the other constructor, here the
       arguments are treated "as is" -- no escape processing is
       done.  */

    static std::unique_ptr<struct mi_parse> make
	 (gdb::unique_xmalloc_ptr<char> command,
	  std::vector<gdb::unique_xmalloc_ptr<char>> args);

    ~mi_parse ();

    DISABLE_COPY_AND_ASSIGN (mi_parse);

    /* Split the arguments into argc/argv and store the result.  */
    void parse_argv ();

    /* Return the full argument string, as used by commands which are
       implemented as CLI commands.  */
    const char *args ();

    enum mi_command_type op = MI_COMMAND;
    char *command = nullptr;
    char *token = nullptr;
    const struct mi_command *cmd = nullptr;
    struct mi_timestamp *cmd_start = nullptr;
    char **argv = nullptr;
    int argc = 0;
    int all = 0;
    int thread_group = -1; /* At present, the same as inferior number.  */
    int thread = -1;
    int lane = -1;
    int frame = -1;

    /* The language that should be used to evaluate the MI command.
       Ignored if set to language_unknown.  */
    enum language language = language_unknown;

  private:

    mi_parse () = default;

    /* Helper methods for parsing arguments.  Each takes the argument
       to be parsed.  It will either set a member of this object, or
       throw an exception on error.  In each case, *ENDP, if non-NULL,
       will be updated to just after the argument text.  */
    void set_thread_group (const char *arg, char **endp);
    void set_thread (const char *arg, char **endp);
    void set_lane (const char *arg, char **endp);
    void set_frame (const char *arg, char **endp);
    void set_language (const char *arg, const char **endp);

    std::string m_args;
  };

/* Parse a string argument into a print_values value.  */

enum print_values mi_parse_print_values (const char *name);

#endif /* MI_MI_PARSE_H */
