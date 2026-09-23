/* Copyright (C) 2023-2026 Free Software Foundation, Inc.

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

#include "gdbsupport/remote-args.h"
#include "gdbsupport/common-inferior.h"
#include "gdbsupport/buildargv.h"
#include "gdbsupport/special-shell-chars.h"

/* This file contains the function used for splitting an argument string
   into separate arguments in preparation for sending the arguments over
   the remote protocol, as well as the function to merge the separate
   arguments back together into a single argument string.

   The logic within these functions is slightly more complex than it should
   be.  This is in order to maintain some level of backward compatibility.

   In the following text, examples of command line arguments will be given.
   To avoid confusion, arguments, and argument strings, will be delimited
   with '(' and ')', the parentheses are not part of the argument itself.
   This is clearer than using quotes, as some of the examples will include
   quotes within the arguments.

   Historically, the algorithm used to split the argument string into
   separate arguments removed a level of quoting from the arguments.  For
   example consider the following argument string: (abc* abc\*).  The
   historic algorithm would split this into (abc*) and (abc*).  Notice that
   the two arguments are identical.  On the remote end we are now destined
   for failure; either we apply an escape to both '*' characters, or we
   apply an escape to neither.  In either case, we get one of the arguments
   wrong.  The historic approach was just broken.

   However, the historic approach has been in place for many years.
   Clearly not all arguments were corrupted in the manner described above,
   so lots of things did work.  For example, the string: ("ab cd" "ef")
   will be split into (ab cd) and (ef).  And the string ('"') will become
   just (").

   What we can observe from all of these examples is that the historic
   approach at the remote end was to simply apply an escape to every
   special shell character, quotes, white space, as well as every other
   special character (e.g. (*)).  The problem with this approach is that
   sometimes special shell characters shouldn't be escaped.

   If we could start from scratch, then the simple approach would be to
   retain all escaping while splitting the argument string, and, where
   quotes are used, convert this into backslash escaping as needed.  Thus
   the argument string ("ab cd" "ef") would become (ab\ cd) and (ef).  And
   the argument string (abc* abc\*) would become (abc*) (abc\*).  On the
   remote end, joining these arguments is as simple as concatenation with a
   single space character between.

   However, if we change to this approach now, then consider ("ab cd").
   Previously this was sent as (ab cd), but now it would be (ab\ cd).  This
   breaks backward compatibility.  Remember, we need to think about newer
   GDB being used with older gdbserver, or newer GDB being used with
   targets that are not gdbserver at all.

   And so, this is where the complexity comes in.

   The strategy here is to split the arguments, removing all double and
   single quotes.  While removing quotes, special shell characters are
   escaped as needed.  But white space characters and quote characters
   are not escaped.  These characters must always be escaped, and so we can
   safely drop the escape in these cases.  This provides some degree of
   backward compatibility, but is still a breaking change from the old
   scheme.  */

/* Return true if C is a double or single quote character.  */

static bool
isquote (const char c)
{
  return c == '"' || c == '\'';
}

/* See remote-args.h.  */

std::vector<std::string>
gdb::remote_args::split (const std::string &args)
{
  std::vector<std::string> remote_args;

  const char *input = args.c_str ();
  bool squote = false, dquote = false;

  /* Create a shorter alias for SPECIAL_SHELL_CHARACTERS.  */
  const char *special = special_shell_characters;

  /* Characters that are special within double quotes.  */
  static const char dquote_special[] = "$`\\";

  input = skip_spaces (input);

  while (*input != '\0')
    {
      std::string arg;

      while (*input != '\0')
	{
	  if (isspace (*input) && !squote && !dquote)
	    break;
	  else if (*input == '\\' && !squote)
	    {
	      if (input[1] == '\0')
		arg += input[0];
	      else if (input[1] == '\n')
		++input;
	      else if (dquote && input[1] == '"')
		{
		  arg += input[1];
		  ++input;
		}
	      else if (dquote && strchr (dquote_special, input[1]) != nullptr)
		{
		  /* Within double quotes, these characters would usually
		     have special meaning, but they are escaped with a
		     backslash.  Preserve the escaping by adding both a
		     backslash and the special character to ARG.  */
		  arg += '\\';
		  arg += input[1];
		  ++input;
		}
	      else if (dquote)
		{
		  /* Within double quotes, none of the remaining characters
		     have any special meaning, the backslash before the
		     character is a literal backslash.

		     To retain a backslash with the quotes removed, we need
		     to escape the backslash with a second backslash.  */
		  arg += '\\';
		  arg += '\\';

		  /* If the character after the backslash has special
		     meaning outside of the double quotes, then we need to
		     escape it now, otherwise it will gain special meaning
		     as we remove the surrounding quotes.  However, as per
		     the comments at the head of this file; we don't
		     escape white space or quotes.  */
		  if (!isspace (input[1])
		      && !isquote (input[1])
		      && strchr (special, input[1]) != nullptr)
		    arg += '\\';

		  /* Now just add the character.  */
		  arg += input[1];
		  ++input;
		}
	      else if (isspace (input[1]) || isquote (input[1]))
		{
		  /* We remove the escaping from white space and quote
		     characters, just insert the character.  */
		  arg += input[1];
		  ++input;
		}
	      else
		{
		  /* For everything else, retain the escaping by adding back
		     a backslash and then the character.  */
		  arg += '\\';
		  arg += input[1];
		  ++input;
		}
	    }
	  else if (squote)
	    {
	      /* Inside a single quote argument there are no special
		 characters.  A single quote finishes the argument.  */
	      if (*input == '\'')
		squote = false;
	      /* Don't add escaping to white space or quotes.  We already
		 handled single quotes above, so the isquote call here will
		 only find double quotes.  */
	      else if (isspace (*input) || isquote (*input))
		arg += *input;
	      /* Any other special shell character needs a backslash added
		 to avoid the character gaining special meaning outside of
		 the single quotes.  */
	      else if (strchr (special, *input) != nullptr)
		{
		  arg += '\\';
		  arg += *input;
		}
	      /* Any other character just gets added to the output.  */
	      else
		arg += *input;
	    }
	  else if (dquote)
	    {
	      /* Inside a double quoted argument.  A double quote closes
		 the argument.  An escaped double quote will have been
		 handled in the backslash handling block above.  */
	      if (*input == '"')
		dquote = false;
	      /* Don't add escaping for white space or quotes.  We already
		 handled double quotes above, so the isquote call here will
		 only find single quotes.  */
	      else if (isspace (*input) || isquote (*input))
		arg += *input;
	      /* Any character that is not one of the few characters that
		 retain its special meaning without double quotes, but is
		 otherwise a special character, needs an escape character
		 added to avoid the character gaining special meaning
		 outside of the quotes.  */
	      else if (strchr (dquote_special, *input) == nullptr
		       && strchr (special, *input) != nullptr)
		{
		  arg += '\\';
		  arg += *input;
		}
	      /* Anything else just gets passed through to the output.  */
	      else
		arg += *input;
	    }
	  else
	    {
	      /* Found a character outside of a single or double quoted
		 argument, and not preceded by a backslash.  */
	      if (*input == '\'')
		squote = true;
	      else if (*input == '"')
		dquote = true;
	      else
		arg += *input;
	    }
	  ++input;
	}

      remote_args.push_back (std::move (arg));
      input = skip_spaces (input);
    }

  return remote_args;
}
