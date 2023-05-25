/* MI Command Set - MI parser.

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

#include "defs.h"
#include "mi-cmds.h"
#include "mi-parse.h"
#include "charset.h"

#include <ctype.h>
#include "cli/cli-utils.h"
#include "language.h"

static const char mi_no_values[] = "--no-values";
static const char mi_simple_values[] = "--simple-values";
static const char mi_all_values[] = "--all-values";

/* Like parse_escape, but leave the results as a host char, not a
   target char.  */

static int
mi_parse_escape (const char **string_ptr)
{
  int c = *(*string_ptr)++;

  switch (c)
    {
      case '\n':
	return -2;
      case 0:
	(*string_ptr)--;
	return 0;

      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
	{
	  int i = fromhex (c);
	  int count = 0;

	  while (++count < 3)
	    {
	      c = (**string_ptr);
	      if (isdigit (c) && c != '8' && c != '9')
		{
		  (*string_ptr)++;
		  i *= 8;
		  i += fromhex (c);
		}
	      else
		{
		  break;
		}
	    }
	  return i;
	}

    case 'a':
      c = '\a';
      break;
    case 'b':
      c = '\b';
      break;
    case 'f':
      c = '\f';
      break;
    case 'n':
      c = '\n';
      break;
    case 'r':
      c = '\r';
      break;
    case 't':
      c = '\t';
      break;
    case 'v':
      c = '\v';
      break;

    default:
      break;
    }

  return c;
}

void
mi_parse::parse_argv ()
{
  /* If arguments were already computed (or were supplied at
     construction), then there's no need to re-compute them.  */
  if (argv != nullptr)
    return;

  const char *chp = m_args.c_str ();
  int argc = 0;
  char **argv = XNEWVEC (char *, argc + 1);

  argv[argc] = NULL;
  while (1)
    {
      char *arg;

      /* Skip leading white space.  */
      chp = skip_spaces (chp);
      /* Three possibilities: EOF, quoted string, or other text. */
      switch (*chp)
	{
	case '\0':
	  this->argv = argv;
	  this->argc = argc;
	  return;
	case '"':
	  {
	    /* A quoted string.  */
	    int len;
	    const char *start = chp + 1;

	    /* Determine the buffer size.  */
	    chp = start;
	    len = 0;
	    while (*chp != '\0' && *chp != '"')
	      {
		if (*chp == '\\')
		  {
		    chp++;
		    if (mi_parse_escape (&chp) <= 0)
		      {
			/* Do not allow split lines or "\000".  */
			freeargv (argv);
			return;
		      }
		  }
		else
		  chp++;
		len++;
	      }
	    /* Insist on a closing quote.  */
	    if (*chp != '"')
	      {
		freeargv (argv);
		return;
	      }
	    /* Insist on trailing white space.  */
	    if (chp[1] != '\0' && !isspace (chp[1]))
	      {
		freeargv (argv);
		return;
	      }
	    /* Create the buffer and copy characters in.  */
	    arg = XNEWVEC (char, len + 1);
	    chp = start;
	    len = 0;
	    while (*chp != '\0' && *chp != '"')
	      {
		if (*chp == '\\')
		  {
		    chp++;
		    arg[len] = mi_parse_escape (&chp);
		  }
		else
		  arg[len] = *chp++;
		len++;
	      }
	    arg[len] = '\0';
	    chp++;		/* That closing quote.  */
	    break;
	  }
	default:
	  {
	    /* An unquoted string.  Accumulate all non-blank
	       characters into a buffer.  */
	    int len;
	    const char *start = chp;

	    while (*chp != '\0' && !isspace (*chp))
	      {
		chp++;
	      }
	    len = chp - start;
	    arg = XNEWVEC (char, len + 1);
	    strncpy (arg, start, len);
	    arg[len] = '\0';
	    break;
	  }
	}
      /* Append arg to argv.  */
      argv = XRESIZEVEC (char *, argv, argc + 2);
      argv[argc++] = arg;
      argv[argc] = NULL;
    }
}

mi_parse::~mi_parse ()
{
  xfree (command);
  xfree (token);
  freeargv (argv);
}

/* See mi-parse.h.  */

const char *
mi_parse::args ()
{
  /* If args were already computed, or if there is no pre-computed
     argv, just return the args.  */
  if (!m_args.empty () || argv == nullptr)
    return  m_args.c_str ();

  /* Compute args from argv.  */
  for (int i = 0; i < argc; ++i)
    {
      if (!m_args.empty ())
	m_args += " ";
      m_args += argv[i];
    }

  return m_args.c_str ();
}

/* See mi-parse.h.  */

void
mi_parse::set_thread_group (const char *arg, char **endp)
{
  if (thread_group != -1)
    error (_("Duplicate '--thread-group' option"));
  if (*arg != 'i')
    error (_("Invalid thread group id"));
  arg += 1;
  thread_group = strtol (arg, endp, 10);
}

/* See mi-parse.h.  */

void
mi_parse::set_thread (const char *arg, char **endp)
{
  if (thread != -1)
    error (_("Duplicate '--thread' option"));
  thread = strtol (arg, endp, 10);
}

/* See mi-parse.h.  */

void
mi_parse::set_lane (const char *arg, char **endp)
{
  if (lane != -1)
    error (_("Duplicate '--lane' option"));
  lane = strtol (arg, endp, 10);
}

/* See mi-parse.h.  */

void
mi_parse::set_frame (const char *arg, char **endp)
{
  if (frame != -1)
    error (_("Duplicate '--frame' option"));
  frame = strtol (arg, endp, 10);
}

/* See mi-parse.h.  */

void
mi_parse::set_language (const char *arg, const char **endp)
{
  std::string lang_name = extract_arg (&arg);

  language = language_enum (lang_name.c_str ());
  if (language == language_unknown)
    error (_("Invalid --language argument: %s"), lang_name.c_str ());

  if (endp != nullptr)
    *endp = arg;
}

std::unique_ptr<struct mi_parse>
mi_parse::make (const char *cmd, char **token)
{
  const char *chp;

  std::unique_ptr<struct mi_parse> parse (new struct mi_parse);

  /* Before starting, skip leading white space.  */
  cmd = skip_spaces (cmd);

  /* Find/skip any token and then extract it.  */
  for (chp = cmd; *chp >= '0' && *chp <= '9'; chp++)
    ;
  *token = (char *) xmalloc (chp - cmd + 1);
  memcpy (*token, cmd, (chp - cmd));
  (*token)[chp - cmd] = '\0';

  /* This wasn't a real MI command.  Return it as a CLI_COMMAND.  */
  if (*chp != '-')
    {
      chp = skip_spaces (chp);
      parse->command = xstrdup (chp);
      parse->op = CLI_COMMAND;

      return parse;
    }

  /* Extract the command.  */
  {
    const char *tmp = chp + 1;	/* discard ``-'' */

    for (; *chp && !isspace (*chp); chp++)
      ;
    parse->command = (char *) xmalloc (chp - tmp + 1);
    memcpy (parse->command, tmp, chp - tmp);
    parse->command[chp - tmp] = '\0';
  }

  /* Find the command in the MI table.  */
  parse->cmd = mi_cmd_lookup (parse->command);
  if (parse->cmd == NULL)
    throw_error (UNDEFINED_COMMAND_ERROR,
		 _("Undefined MI command: %s"), parse->command);

  /* Skip white space following the command.  */
  chp = skip_spaces (chp);

  /* Parse the --thread and --frame options, if present.  At present,
     some important commands, like '-break-*' are implemented by
     forwarding to the CLI layer directly.  We want to parse --thread
     and --frame here, so as not to leave those option in the string
     that will be passed to CLI.

     Same for the --language option.  */

  for (;;)
    {
      const char *option;
      size_t as = sizeof ("--all ") - 1;
      size_t tgs = sizeof ("--thread-group ") - 1;
      size_t ts = sizeof ("--thread ") - 1;
      size_t lane_s = sizeof ("--lane ") - 1;
      size_t fs = sizeof ("--frame ") - 1;
      size_t lang_s = sizeof ("--language ") - 1;

      if (strncmp (chp, "--all ", as) == 0)
	{
	  parse->all = 1;
	  chp += as;
	}
      /* See if --all is the last token in the input.  */
      if (strcmp (chp, "--all") == 0)
	{
	  parse->all = 1;
	  chp += strlen (chp);
	}
      if (strncmp (chp, "--thread-group ", tgs) == 0)
	{
	  char *endp;

	  option = "--thread-group";
	  chp += tgs;
	  parse->set_thread_group (chp, &endp);
	  chp = endp;
	}
      else if (strncmp (chp, "--thread ", ts) == 0)
	{
	  char *endp;

	  option = "--thread";
	  chp += ts;
	  parse->set_thread (chp, &endp);
	  chp = endp;
	}
      else if (strncmp (chp, "--lane ", lane_s) == 0)
	{
	  char *endp;

	  option = "--lane";
	  chp += lane_s;
	  parse->set_lane (chp, &endp);
	  chp = endp;
	}
      else if (strncmp (chp, "--frame ", fs) == 0)
	{
	  char *endp;

	  option = "--frame";
	  chp += fs;
	  parse->set_frame (chp, &endp);
	  chp = endp;
	}
      else if (strncmp (chp, "--language ", lang_s) == 0)
	{
	  option = "--language";
	  chp += lang_s;
	  parse->set_language (chp, &chp);
	}
      else
	break;

      if (*chp != '\0' && !isspace (*chp))
	error (_("Invalid value for the '%s' option"), option);
      chp = skip_spaces (chp);
    }

  /* Save the rest of the arguments for the command.  */
  parse->m_args = chp;

  /* Fully parsed, flag as an MI command.  */
  parse->op = MI_COMMAND;
  return parse;
}

/* See mi-parse.h.  */

std::unique_ptr<struct mi_parse>
mi_parse::make (gdb::unique_xmalloc_ptr<char> command,
		std::vector<gdb::unique_xmalloc_ptr<char>> args)
{
  std::unique_ptr<struct mi_parse> parse (new struct mi_parse);

  parse->command = command.release ();
  parse->token = xstrdup ("");

  if (parse->command[0] != '-')
    throw_error (UNDEFINED_COMMAND_ERROR,
		 _("MI command '%s' does not start with '-'"),
		 parse->command);

  /* Find the command in the MI table.  */
  parse->cmd = mi_cmd_lookup (parse->command + 1);
  if (parse->cmd == NULL)
    throw_error (UNDEFINED_COMMAND_ERROR,
		 _("Undefined MI command: %s"), parse->command);

  /* This over-allocates slightly, but it seems unimportant.  */
  parse->argv = XCNEWVEC (char *, args.size () + 1);

  for (size_t i = 0; i < args.size (); ++i)
    {
      const char *chp = args[i].get ();

      /* See if --all is the last token in the input.  */
      if (strcmp (chp, "--all") == 0)
	{
	  parse->all = 1;
	}
      else if (strcmp (chp, "--thread-group") == 0)
	{
	  ++i;
	  if (i == args.size ())
	    error ("No argument to '--thread-group'");
	  parse->set_thread_group (args[i].get (), nullptr);
	}
      else if (strcmp (chp, "--thread") == 0)
	{
	  ++i;
	  if (i == args.size ())
	    error ("No argument to '--thread'");
	  parse->set_thread (args[i].get (), nullptr);
	}
      else if (strcmp (chp, "--lane") == 0)
	{
	  ++i;
	  if (i == args.size ())
	    error ("No argument to '--lane'");
	  parse->set_lane (args[i].get (), nullptr);
	}
      else if (strcmp (chp, "--frame") == 0)
	{
	  ++i;
	  if (i == args.size ())
	    error ("No argument to '--frame'");
	  parse->set_frame (args[i].get (), nullptr);
	}
      else if (strcmp (chp, "--language") == 0)
	{
	  ++i;
	  if (i == args.size ())
	    error ("No argument to '--language'");
	  parse->set_language (args[i].get (), nullptr);
	}
      else
	parse->argv[parse->argc++] = args[i].release ();
    }

  /* Fully parsed, flag as an MI command.  */
  parse->op = MI_COMMAND;
  return parse;
}

enum print_values
mi_parse_print_values (const char *name)
{
   if (strcmp (name, "0") == 0
       || strcmp (name, mi_no_values) == 0)
     return PRINT_NO_VALUES;
   else if (strcmp (name, "1") == 0
	    || strcmp (name, mi_all_values) == 0)
     return PRINT_ALL_VALUES;
   else if (strcmp (name, "2") == 0
	    || strcmp (name, mi_simple_values) == 0)
     return PRINT_SIMPLE_VALUES;
   else
     error (_("Unknown value for PRINT_VALUES: must be: \
0 or \"%s\", 1 or \"%s\", 2 or \"%s\""),
	    mi_no_values, mi_all_values, mi_simple_values);
}
