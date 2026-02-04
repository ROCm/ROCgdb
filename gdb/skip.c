/* Skipping uninteresting files and functions while stepping.

   Copyright (C) 2011-2026 Free Software Foundation, Inc.

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

#include "skip.h"
#include "event-top.h"
#include "value.h"
#include "valprint.h"
#include "ui-out.h"
#include "symtab.h"
#include "cli/cli-cmds.h"
#include "command.h"
#include "completer.h"
#include "stack.h"
#include "cli/cli-utils.h"
#include "arch-utils.h"
#include "linespec.h"
#include "objfiles.h"
#include "breakpoint.h"
#include "source.h"
#include "filenames.h"
#include "fnmatch.h"
#include "gdbsupport/gdb_regex.h"
#include <optional>
#include <list>
#include "cli/cli-style.h"
#include "gdbsupport/buildargv.h"
#include "safe-ctype.h"
#include "readline/tilde.h"
#include "gdbsupport/selftest.h"

#include <initializer_list>
#include <unordered_set>
#include <vector>

/* True if we want to print debug printouts related to file/function
   skipping. */
static bool debug_skip = false;

/* Print a "skip" debug statement.  */

#define skip_debug_printf(fmt, ...) \
  debug_prefixed_printf_cond (debug_skip, "skip", fmt, ##__VA_ARGS__)

class skiplist_entry
{
public:
  /* Create a skiplist_entry object and add it to the chain.  */
  static void add_entry (bool file_is_glob,
			 std::string &&file,
			 bool function_is_regexp,
			 std::string &&function);

  /* Return true if the skip entry has a file or glob-style file
     pattern that matches FUNCTION_SAL.  */
  bool skip_file_p (const symtab_and_line &function_sal) const;

  /* Return true if the skip entry has a function or function regexp
     that matches FUNCTION_NAME.  */
  bool skip_function_p (const char *function_name) const;

  /* Getters.  */
  int number () const { return m_number; };
  bool enabled () const { return m_enabled; };
  bool file_is_glob () const { return m_file_is_glob; }
  const std::string &file () const { return m_file; }
  const std::string &function () const { return m_function; }
  bool function_is_regexp () const { return m_function_is_regexp; }

  /* Setters.  */
  void enable () { m_enabled = true; };
  void disable () { m_enabled = false; };

  /* Print a gdb command that can be used to recreate this skip.  */
  void print_recreate (ui_file *stream) const;

private:
  /* Key that grants access to the constructor.  */
  struct private_key {};
public:
  /* Public so we can construct with container::emplace_back.  Since
     it requires a private class key, it can't be called from outside.
     Use the add_entry static factory method to construct instead.  */
  skiplist_entry (bool file_is_glob, std::string &&file,
		  bool function_is_regexp, std::string &&function,
		  private_key);

  /* Disable copy.  */
  DISABLE_COPY_AND_ASSIGN (skiplist_entry);

private:
  /* Return true if we're stopped at a file to be skipped.  */
  bool do_skip_file_p (const symtab_and_line &function_sal) const;

  /* Return true if we're stopped at a globbed file to be skipped.  */
  bool do_skip_gfile_p (const symtab_and_line &function_sal) const;

private: /* data */
  int m_number = -1;

  /* True if FILE is a glob-style pattern.
     Otherwise it is the plain file name (possibly with directories).  */
  bool m_file_is_glob;

  /* The name of the file or empty if no name.  */
  std::string m_file;

  /* True if FUNCTION is a regexp.
     Otherwise it is a plain function name (possibly with arguments,
     for C++).  */
  bool m_function_is_regexp;

  /* The name of the function or empty if no name.  */
  std::string m_function;

  /* If this is a function regexp, the compiled form.  */
  std::optional<compiled_regex> m_compiled_function_regexp;

  /* If this is a file glob, then we convert the glob to a regexp, and
     place the compiled form here.  */
  std::optional<compiled_regex> m_compiled_file_regexp;

  /* Enabled/disabled state.  */
  bool m_enabled = true;
};

static std::list<skiplist_entry> skiplist_entries;
static int highest_skiplist_entry_num = 0;

/* Structure to hold the results of parsing a glob bracket expression.  */

struct glob_bracket_expr
{
  /* Type used to express a character range, e.g 'a-z'.  In this case the
     first item in the pair would be 'a', and the second item 'z'.  */
  using char_range = std::pair<unsigned char, unsigned char>;

  /* Parse a bracket expression and return an instance of this class.  If
     the bracket expression is invalid then return an empty optional.  PTR
     must point to the '[' character that starts the bracket expression.  */
  static std::optional<glob_bracket_expr> parse (const char *ptr);

  /* Convert the parsed bracket expression into a regular expression, and
     return the regular expression as a string.  */
  std::string to_string () const;

  /* Return a pointer to the end of the bracket expression.  This will point
     to the final ']' that closes the expression.  This will never be NULL.  */
  const char *end () const;

private:
  /* When true the character bracket expression is negated, that is we want
     to match everything not mentioned in the expression.  When false we
     only want to match things mentioned in the expression.  */
  bool m_negated = false;

  /* Each string is something like '[:alpha:]' and represents a character
     class.  If validation is wanted then it should be done before items
     are added to this list, the contents of this list are not validated as
     they are used.  */
  std::vector<std::string> m_char_classes;

  /* Character ranges stored as pairs, first entry is the range start,
     second entry is the range end.  Both are inclusive.  */
  std::vector<char_range> m_char_ranges;

  /* Single characters within the bracket expression.  */
  std::unordered_set<char> m_chars;

  /* Points at the closing ']' for the bracket expression.  */
  const char *m_end = nullptr;
};

/* Return true if PTR is at the start of a character class descriptor,
   e.g. "[:alpha:]".  This doesn't validate the actual character class name,
   just that PTR is at the start of something that looks like a character
   class.  */

static bool
at_character_class (const char *ptr)
{
  if (*ptr != '[')
    return false;
  ++ptr;

  if (*ptr != ':')
    return false;
  ++ptr;

  /* Require at least one character in the class name.  */
  if (!c_isalpha (*ptr))
    return false;

  while (c_isalpha (*ptr))
    ++ptr;

  if (*ptr != ':')
    return false;
  ++ptr;

  return *ptr == ']';
}

/* Return true if PTR is pointing to something like 'A-B', a character
   range as could be found within a glob's bracket expression.  */

static bool
at_character_range (const char *ptr)
{
  gdb_assert (*ptr != '\0');

  if (*(ptr + 1) != '-')
    return false;

  if (*(ptr + 2) == '\0' || *(ptr + 2) == ']')
    return false;

  return true;
}

/* Helper for sanitize_range.  Pass RANGE to SHOULD_SPLIT, if it returns
   true then call DO_SPLIT to split RANGE, placing the resulting ranges in
   RESULTS.  If SHOULD_SPLIT returns false then just place RANGE in
   RESULTS.  */

template<typename Pred, typename Split>
static void
split_a_range (std::vector<glob_bracket_expr::char_range> &results,
	       const glob_bracket_expr::char_range &range,
	       Pred should_split, Split do_split)
{
  if (should_split (range))
    do_split (results, range);
  else
    results.emplace_back (range);
}

/* Helper for sanitize_range.  For each range in RESULTS, if SHOULD_SPLIT
   returns true, replace it with the sub-ranges produced by DO_SPLIT.
   Otherwise keep it unchanged.  */

template<typename Pred, typename Split>
static void
resplit_all_ranges (std::vector<glob_bracket_expr::char_range> &results,
		    Pred should_split, Split do_split)
{
  std::vector<glob_bracket_expr::char_range> temp;
  for (const glob_bracket_expr::char_range &p : results)
    split_a_range (temp, p, should_split, do_split);
  results = std::move (temp);
}

/* The character range START-END is taken from a filename glob.  When
   converting to a regular expression we cannot allow a directory
   separator to appear within the range.  So, if a directory separator
   does appear between START and END (inclusive), split the range into
   multiple ranges, excluding the directory separator.

   For Unix systems the only directory separator we check for is '/', but
   on DOS like file systems we also check for '\'.

   On case insensitive filesystems we also need to be careful about ranges
   that span into, or out of, the upper case character range.  For glibc
   at least, the REG_ICASE matching which we use (for case insensitive
   file systems) lowers any uppercase characters it finds.  So a range
   like [W-_] becomes [w-_], which is invalid as 'w' is after '_'.  To
   avoid this problem we split ranges at 'A' and 'Z', so the above range
   becomes [W-Z[-_], then when glibc lowers the letters this becomes
   [w-z[-_] which is still valid.

   Return a vector of all the ranges needed to cover START to END but
   exclude directory separators.  */

static std::vector<glob_bracket_expr::char_range>
sanitize_range (unsigned char start, unsigned char end)
{
  /* Ranges that start or end at a directory separator are invalid, and
     should have been filtered out before now.  */
  gdb_assert (!IS_DIR_SEPARATOR (start) && !IS_DIR_SEPARATOR (end));

  /* Backward ranges are invalid, and should have been filtered out before
     now.  */
  gdb_assert (start <= end);

  std::vector<glob_bracket_expr::char_range> results;

  /* Always split the range on '/'.  This creates at most two ranges.  */
  split_a_range (results, { start, end },
		 [] (const glob_bracket_expr::char_range &p)
		 { return p.first < '/' && p.second > '/'; },
		 [] (std::vector<glob_bracket_expr::char_range> &out,
		     const glob_bracket_expr::char_range &p)
		 {
		   out.emplace_back (p.first, '/' - 1);
		   out.emplace_back ('/' + 1, p.second);
		 });

#ifdef HAVE_DOS_BASED_FILE_SYSTEM
  /* Exclude '\\' from ranges.  The character after '\\' is ']', which
     needs to be kept as a single-character range since it has special
     meaning within bracket expressions, by keeping the ']' as a single
     character range the ']' can be moved within the bracket expression to
     a valid location. */
  resplit_all_ranges (results,
    [] (const glob_bracket_expr::char_range &p)
      { return p.first < '\\' && p.second > '\\'; },
    [] (std::vector<glob_bracket_expr::char_range> &out,
	const glob_bracket_expr::char_range &p)
      {
	out.emplace_back (p.first, '\\' - 1);
	gdb_assert ('\\' + 1 == ']');
	out.emplace_back (']', ']');
	if (p.second > ']')
	  out.emplace_back (']' + 1, p.second);
      });
#endif /* HAVE_DOS_BASED_FILE_SYSTEM */

#ifdef HAVE_CASE_INSENSITIVE_FILE_SYSTEM
  /* Split ranges at the 'A' and 'Z' boundaries to allow for case
     insensitive matching.  glibc's ICASE matching lowers upper case
     letters within ranges, so the range [W-_] becomes, with ICASE
     matching, [w-_].  Unfortunately, 'w' is after '_' so this range is
     now invalid.  By splitting the range into [W-Z[-_] glibc is now free
     to lower this to [w-z[-_] which is still valid.  */
  resplit_all_ranges (results,
    [] (const glob_bracket_expr::char_range &p)
      { return p.first < 'A' && p.second >= 'A'; },
    [] (std::vector<glob_bracket_expr::char_range> &out,
	const glob_bracket_expr::char_range &p)
      {
	out.emplace_back (p.first, 'A' - 1);
	out.emplace_back ('A', p.second);
      });
  resplit_all_ranges (results,
    [] (const glob_bracket_expr::char_range &p)
      { return p.first <= 'Z' && p.second > 'Z'; },
    [] (std::vector<glob_bracket_expr::char_range> &out,
	const glob_bracket_expr::char_range &p)
      {
	out.emplace_back (p.first, 'Z');
	out.emplace_back ('Z' + 1, p.second);
      });
#endif /* HAVE_CASE_INSENSITIVE_FILE_SYSTEM */

  return results;
}

/* CHAR_CLASS should be a character class name like "[:name:]".  Return
   true if CHAR_CLASS matches a directory separator, which is just '/'
   unless we're on a DOS like filesystem, in which case we check for '/'
   or '\'.  */

static bool
character_class_matches_dir_separator (const std::string &char_class)
{
  std::string pattern = string_printf ("[%s]", char_class.c_str ());
  compiled_regex re (pattern.c_str (), REG_NOSUB | REG_EXTENDED,
		     _("character class check"));

#ifdef HAVE_DOS_BASED_FILE_SYSTEM
  static constexpr const char *dir_sep_str = "/\\";
#else
  static constexpr const char *dir_sep_str = "/";
#endif	/* not HAVE_DOS_BASED_FILE_SYSTEM */

  return re.exec (dir_sep_str, 0, nullptr, 0) == 0;
}

/* See class declaration above.  */

std::optional<glob_bracket_expr>
glob_bracket_expr::parse (const char *ptr)
{
  gdb_assert (*ptr == '[');
  ++ptr;

  glob_bracket_expr result;

  /* POSIX defines '!' as the negation character within a bracket
     expression, and says that a '^' as the first character is undefined.
     Both fnmatch and bash treat a leading '^' as negation, which matches
     the regexp behaviour.  We match this.  */
  if (*ptr == '!' || *ptr == '^')
    {
      result.m_negated = true;
      result.m_chars.insert ('/');
      ptr++;
    }

  char end_bracket_char = '\0';

  for (; *ptr != '\0' && *ptr != end_bracket_char; ++ptr)
    {
      end_bracket_char = ']';

      /* Is this a character range?  */
      if (at_character_range (ptr))
	{
	  char range_start = *ptr;
	  char range_end = *(ptr + 2);

	  /* If the range starts with ']' then the opening ']' must be the
	     first character in the bracket expression (after any
	     negation).  Insert the ']' as a single character, and
	     increment the range start.  We know the range_end cannot also
	     be ']' as that would close the bracket expression, and not be
	     a valid range.  Having the ']' as a lone character make it
	     easier to place this in the correct place within the bracket
	     expression when we generate the regexp.  */
	  if (range_start == ']')
	    {
	      gdb_assert (range_end != ']');
	      result.m_chars.insert (range_start);
	      range_start++;
	    }

	  if (IS_DIR_SEPARATOR (range_start) || IS_DIR_SEPARATOR (range_end))
	    return {};

	  if (range_end < range_start)
	    {
	      /* Don't use RANGE_START here as it might have been modified
		 above.  We could use RANGE_END but don't just to be
		 consistent.  */
	      error (_("skip glob regexp: invalid character range %c-%c"),
		     *ptr, *(ptr + 2));
	    }

	  std::vector<glob_bracket_expr::char_range> ranges
	    = sanitize_range (range_start, range_end);
	  for (glob_bracket_expr::char_range p : ranges)
	    {
	      if (p.first == p.second)
		result.m_chars.insert (p.first);
	      else
		result.m_char_ranges.push_back (std::move (p));
	    }
	  ptr += 2;
	}
      else if (at_character_class (ptr))
	{
	  const char *end = strchr (ptr, ']');
	  gdb_assert (end != nullptr);
	  std::string cc (ptr, end - ptr + 1);
	  if (character_class_matches_dir_separator (cc))
	    {
	      /* We are converting a glob to a regexp and trying to
		 replicate the FNM_PATHNAME flag of fnmatch.  If we allow a
		 character class that matches '/' then this might cause
		 problems.

		 We could try to split the class into multiple ranges that
		 cover all the same characters, but not '/', but for now we
		 just give an error.  */
	      error (_("skip glob regexp: unsupported character class '%s'"),
		     cc.c_str ());
	    }
	  result.m_char_classes.push_back (std::move (cc));
	  ptr = end;
	}
      else
	result.m_chars.insert (*ptr);
    }

  if (*ptr != ']')
    return {};

  result.m_end = ptr;
  return result;
}

/* See class declaration above.  */

std::string
glob_bracket_expr::to_string () const
{
  std::string result ("[");

  if (m_negated)
    result += '^';

  if (m_chars.find (']') != m_chars.end ())
    result += ']';

  for (const glob_bracket_expr::char_range &p : m_char_ranges)
    {
      /* These should be converted to single characters by sanitize_range.  */
      gdb_assert (p.first != ']' && p.second != ']');

      result += p.first;
      result += '-';
      result += p.second;
    }

  for (const std::string &s : m_char_classes)
    result += s;

  for (const char c : m_chars)
    {
      if (strchr ("-^]", c) == nullptr)
	result += c;
    }

  if (m_chars.find ('^') != m_chars.end ())
    result += '^';
  if (m_chars.find ('-') != m_chars.end ())
    result += '-';
  result += ']';
  return result;
}

/* See class declaration above.  */

const char *
glob_bracket_expr::end () const
{
  /* The parse member function always sets to non-NULL before releasing an
     instance of this class into the world.  */
  gdb_assert (m_end != nullptr);
  return m_end;
}

/* When we convert a filename glob into a regular expression, these are
   the flags to use.  If filenames are case insensitive then we add the
   ICASE (ignore case) flag.  */

static constexpr int file_glob_regexp_flags = (REG_NOSUB
					       | REG_EXTENDED
#ifdef HAVE_CASE_INSENSITIVE_FILE_SYSTEM
					       | REG_ICASE
#endif /* HAVE_CASE_INSENSITIVE_FILE_SYSTEM */
					       );

/* GLOB is the user supplied glob pattern as supplied to the 'skip -gfile
   GLOB' command.  This function builds and returns a regular expression
   that matches GLOB.

   On DOS based filesystems, where backslash can serve as a directory
   separator, the returned regexp will accept either directory separator
   character.

   On case insensitive filesystems it is expected that the regexp will be
   compiled with the REG_ICASE flag.  Some character ranges within bracket
   expressions will be split in order to facilitate case insensitive
   matching, see sanitize_range for more details.  */

static std::string
glob_to_regexp (const std::string& glob)
{
  std::string result;

  /* If the GLOB is absolute then we add a beginning anchor, thus a glob
     '/tmp/file.c' will not match '/blah/tmp/file.c'.  If the glob is not
     absolute then we add a preceding slash, this means a glob 'a/file.c'
     will not match '/aa/file.c', but will match '/tmp/a/file.c'.  */
  if (!IS_ABSOLUTE_PATH (glob.c_str ()))
    result += "/";
  else
    result += "^";

  static const std::string dir_separator_regexp
#ifdef HAVE_DOS_BASED_FILE_SYSTEM
    ("(\\\\|/)");
#else
    ("/");
#endif /* not HAVE_DOS_BASED_FILE_SYSTEM */

  /* Many patterns want to match any character.  Any in this case doesn't
     include slash, so lets define a regexp to match anything except a
     slash.  */
  static const std::string any_char
#ifdef HAVE_DOS_BASED_FILE_SYSTEM
    ("[^\\/]");
#else
    ("[^/]");
#endif /* not HAVE_DOS_BASED_FILE_SYSTEM */

  /* True when we can accept '**', false otherwise.  The '**' pattern is
     modelled on the bash globstar feature, it matches zero or more
     directories.  */
  bool can_globstar = true;

  for (const char *ptr = glob.c_str ();
       *ptr != '\0';
       ++ptr)
    {
      /* In a position where '**' could appear, but this is not the first
	 '*', so we can no longer accept '**'.  This will reset after the
	 next slash.  */
      if (can_globstar && *ptr != '*')
	can_globstar = false;

      if (strchr (".+()|{}$^", *ptr) != nullptr)
	{
	  /* This is a character that has no special meaning within a glob,
	     but does within a regexp, this needs to be escaped.  */
	  result = result + '\\' + *ptr;
	}
      else if (*ptr == '*')
	{
	  /* Are we looking at '**' in a location where this is valid. After
	     the '**' must be either a slash, or the end of the string.  */
	  if (can_globstar && *(ptr + 1) == '*'
	      && (IS_DIR_SEPARATOR (*(ptr + 2)) || *(ptr + 2) == '\0'))
	    {
	      /* If the '**' is at the end of the string then we allow it
		 to match anything.  This is inline with bash.  */
	      if (*(ptr + 2) == '\0')
		{
		  result += ".*";
		  ptr += 1;
		}
	      /* The '**' is followed by a slash.  We accept zero or more
		 directories, which are any character sequence followed by
		 a slash.  */
	      else
		{
		  result += "(" + any_char + "*" + dir_separator_regexp + ")*";
		  ptr += 2;
		}
	    }
	  /* A single '*' or the start of '**' is a location where '**' is
	     not valid.  Match any number (zero or more) non-slash
	     characters.  */
	  else
	    result += any_char + '*';
	}
      else if (*ptr == '?')
	{
	  /* Match a single non-slash character.  */
	  result += any_char;
	}
      else if (IS_DIR_SEPARATOR (*ptr))
	{
	  /* After a slash we can see '**' again.  On builds where
	     backslash is also a possible directory separator, this
	     converts the backslash to a forward slash.  */
	  result += dir_separator_regexp;
	  can_globstar = true;
	}
      else if (*ptr == '\\')
	{
	  /* The code that this regexp logic replaced used to call fnmatch
	     with FNM_NOESCAPE flag.  This flag means that backslash has no
	     special meaning.  We maintain that here by escaping any
	     backslashes.  This will only trigger if the earlier
	     IS_DIR_SEPARATOR check doesn't match backslashes.  */
	  result += "\\\\";
	}
      else if (*ptr == '[')
	{
	  std::optional<glob_bracket_expr> g = glob_bracket_expr::parse (ptr);

	  if (g.has_value ())
	    {
	      /* Add a regexp version of this bracket expression.  */
	      result += g->to_string ();

	      /* Skip over the bracket expression in the glob.  */
	      ptr = g->end ();
	    }
	  else
	    {
	      /* Not a bracket expression.  Just treat this as a literal
		 character.  */
	      result += "\\[";
	    }
	}
      else
	{
	  /* All other characters are passed through.  */
	  result += *ptr;
	}
    }

  /* Anchor the regexp to the end of the filename.  */
  result += '$';

  return result;
}

skiplist_entry::skiplist_entry (bool file_is_glob,
				std::string &&file,
				bool function_is_regexp,
				std::string &&function,
				private_key)
  : m_file_is_glob (file_is_glob),
    m_file (std::move (file)),
    m_function_is_regexp (function_is_regexp),
    m_function (std::move (function))
{
  gdb_assert (!m_file.empty () || !m_function.empty ());

  if (m_file_is_glob)
    {
      gdb_assert (!m_file.empty ());
      m_compiled_file_regexp.emplace (glob_to_regexp (m_file).c_str (),
				      file_glob_regexp_flags,
				      _("skip glob regexp"));
    }

  if (m_function_is_regexp)
    {
      gdb_assert (!m_function.empty ());
      m_compiled_function_regexp.emplace (m_function.c_str (),
					  REG_NOSUB | REG_EXTENDED,
					  _("regexp"));
    }
}

void
skiplist_entry::add_entry (bool file_is_glob, std::string &&file,
			   bool function_is_regexp, std::string &&function)
{
  skiplist_entries.emplace_back (file_is_glob,
				 std::move (file),
				 function_is_regexp,
				 std::move (function),
				 private_key {});

  /* Incremented after push_back, in case push_back throws.  */
  skiplist_entries.back ().m_number = ++highest_skiplist_entry_num;
}

/* A helper function for print_recreate that prints a correctly-quoted
   string to STREAM.  */

static void
print_quoted (ui_file *stream, const std::string &str)
{
  gdb_putc ('"', stream);
  for (char c : str)
    {
      if (ISSPACE (c) || c == '\\' || c == '\'' || c == '"')
	gdb_putc ('\\', stream);
      gdb_putc (c, stream);
    }
  gdb_putc ('"', stream);
}

void
skiplist_entry::print_recreate (ui_file *stream) const
{
  if (!m_file_is_glob && !m_file.empty ()
      && !m_function_is_regexp && m_function.empty ())
    gdb_printf (stream, "skip file %s\n", m_file.c_str ());
  else if (!m_file_is_glob && m_file.empty ()
      && !m_function_is_regexp && !m_function.empty ())
    gdb_printf (stream, "skip function %s\n", m_function.c_str ());
  else
    {
      gdb_printf (stream, "skip ");
      if (!m_file.empty ())
	{
	  if (m_file_is_glob)
	    gdb_printf (stream, "-gfile ");
	  else
	    gdb_printf (stream, "-file ");
	  print_quoted (stream, m_file);
	}
      if (!m_function.empty ())
	{
	  if (m_function_is_regexp)
	    gdb_printf (stream, "-rfunction ");
	  else
	    gdb_printf (stream, "-function ");
	  print_quoted (stream, m_function);
	}
      gdb_printf (stream, "\n");
    }
}

static void
skip_file_command (const char *arg, int from_tty)
{
  struct symtab *symtab;
  const char *filename = NULL;

  /* If no argument was given, try to default to the last
     displayed codepoint.  */
  if (arg == NULL)
    {
      symtab = get_last_displayed_symtab ();
      if (symtab == NULL)
	error (_("No default file now."));

      /* It is not a typo, symtab_to_filename_for_display would be needlessly
	 ambiguous.  */
      filename = symtab_to_fullname (symtab);
    }
  else
    filename = arg;

  skiplist_entry::add_entry (false, std::string (filename),
			     false, std::string ());

  gdb_printf (_("File %ps will be skipped when stepping.\n"),
	      styled_string (file_name_style.style (), filename));
}

/* Create a skiplist entry for the given function NAME and add it to the
   list.  */

static void
skip_function (const char *name)
{
  skiplist_entry::add_entry (false, std::string (), false, std::string (name));

  gdb_printf (_("Function %ps will be skipped when stepping.\n"),
	      styled_string (function_name_style.style (), name));
}

static void
skip_function_command (const char *arg, int from_tty)
{
  /* Default to the current function if no argument is given.  */
  if (arg == NULL)
    {
      frame_info_ptr fi = get_selected_frame (_("No default function now."));
      struct symbol *sym = get_frame_function (fi);
      const char *name = NULL;

      if (sym != NULL)
	name = sym->print_name ();
      else
	error (_("No function found containing current program point %s."),
	       paddress (get_current_arch (), get_frame_pc (fi)));
      skip_function (name);
      return;
    }

  skip_function (arg);
}

/* Storage for the "skip" command option values.  */

struct skip_opts
{
  /* For the -file option.  */
  std::string file;

  /* For the -gfile option.  */
  std::string gfile;

  /* For the -function option.  */
  std::string function;

  /* For the -rfunction option.  */
  std::string rfunction;
};

/* The options for the "skip" command.  */

static const gdb::option::option_def skip_option_defs[] = {
  gdb::option::filename_option_def<skip_opts> {
    "file",
    [] (skip_opts *opts) { return &opts->file; },
    nullptr, /* show_cmd_cb */
    nullptr, /* set_doc */
  },

  gdb::option::filename_option_def<skip_opts> {
    "gfile",
    [] (skip_opts *opts) { return &opts->gfile; },
    nullptr, /* show_cmd_cb */
    nullptr, /* set_doc */
  },

  gdb::option::string_option_def<skip_opts> {
    "function",
    [] (skip_opts *opts) { return &opts->function; },
    nullptr, /* show_cmd_cb */
    nullptr, /* set_doc */
  },

  gdb::option::string_option_def<skip_opts> {
    "rfunction",
    [] (skip_opts *opts) { return &opts->rfunction; },
    nullptr, /* show_cmd_cb */
    nullptr, /* set_doc */
  },
};

/* Create the option_def_group for the "skip" command.  */

static inline gdb::option::option_def_group
make_skip_options_def_group (skip_opts *opts)
{
  return {{skip_option_defs}, opts};
}

/* Completion for the "skip" command.  */

static void
skip_command_completer (struct cmd_list_element *cmd,
			completion_tracker &tracker,
			const char *text, const char * /* word */)
{
  /* We only need to handle option completion here.  The sub-commands of
     'skip' are handled automatically by the command system.  */
  const auto group = make_skip_options_def_group (nullptr);
  gdb::option::complete_options
    (tracker, &text, gdb::option::PROCESS_OPTIONS_UNKNOWN_IS_OPERAND, group);
}

/* Process "skip ..." that does not match "skip file" or "skip function".  */

static void
skip_command (const char *arg, int from_tty)
{
  if (arg == nullptr)
    {
      skip_function_command (arg, from_tty);
      return;
    }

  /* Parse command line options.  */
  skip_opts opts;
  const auto group = make_skip_options_def_group (&opts);
  gdb::option::process_options
    (&arg, gdb::option::PROCESS_OPTIONS_UNKNOWN_IS_OPERAND, group);

  /* Handle invalid argument combinations.  */
  if (!opts.file.empty () && !opts.gfile.empty ())
    error (_("Cannot specify both -file and -gfile."));

  if (!opts.function.empty () && !opts.rfunction.empty ())
    error (_("Cannot specify both -function and -rfunction."));

  /* If there's anything left on the command line then this is an error if
     a valid command line flag was also given, or we assume that it's a
     function name if the user entered 'skip blah'.  */
  if (*arg != '\0')
    {
      if (!opts.file.empty () || !opts.gfile.empty ()
	  || !opts.function.empty () || !opts.rfunction.empty ())
	error (_("Junk after arguments: %s"), arg);
      else if (*arg == '-')
	{
	  const char *after_arg = skip_to_space (arg);
	  error (_("Invalid skip option: %.*s"), (int) (after_arg - arg), arg);
	}
      else
	{
	  /* Assume the user entered "skip FUNCTION-NAME".  FUNCTION-NAME
	     may be `foo (int)', and therefore we pass the complete
	     original ARG to skip_function_command as if the user typed
	     "skip function arg".  */
	  skip_function_command (arg, from_tty);
	  return;
	}
    }

  /* This shouldn't happen as "skip" by itself gets punted to
     skip_function_command, and if the user entered "skip blah" then this
     will have been handled above.  */
  gdb_assert (!opts.file.empty () || !opts.gfile.empty ()
	      || !opts.function.empty () || !opts.rfunction.empty ());

  /* Create the skip list entry.  */
  std::string entry_file;
  if (!opts.file.empty ())
    entry_file = opts.file;
  else if (!opts.gfile.empty ())
    entry_file = opts.gfile;

  std::string entry_function;
  if (!opts.function.empty ())
    entry_function = opts.function;
  else if (!opts.rfunction.empty ())
    entry_function = opts.rfunction;

  skiplist_entry::add_entry (!opts.gfile.empty (),
			     std::move (entry_file),
			     !opts.rfunction.empty (),
			     std::move (entry_function));

  /* I18N concerns drive some of the choices here (we can't piece together
     the output too much).  OTOH we want to keep this simple.  Therefore the
     only polish we add to the output is to append "(s)" to "File" or
     "Function" if they're a glob/regexp.  */
  {
    std::string &file_to_print
      = !opts.file.empty () ? opts.file : opts.gfile;
    std::string &function_to_print
      = !opts.function.empty () ? opts.function : opts.rfunction;

    const char *file_text = !opts.gfile.empty () ? _("File(s)") : _("File");
    const char *lower_file_text = !opts.gfile.empty () ? _("file(s)") : _("file");
    const char *function_text
      = !opts.rfunction.empty () ? _("Function(s)") : _("Function");

    if (function_to_print.empty ())
      {
	gdb_printf (_("%s %ps will be skipped when stepping.\n"),
		    file_text,
		    styled_string (file_name_style.style (),
				   file_to_print.c_str ()));
      }
    else if (file_to_print.empty ())
      {
	gdb_printf (_("%s %ps will be skipped when stepping.\n"),
		    function_text,
		    styled_string (function_name_style.style (),
				   function_to_print.c_str ()));
      }
    else
      {
	gdb_printf (_("%s %ps in %s %ps will be skipped"
		      " when stepping.\n"),
		    function_text,
		    styled_string (function_name_style.style (),
				   function_to_print.c_str ()),
		    lower_file_text,
		    styled_string (file_name_style.style (),
				   file_to_print.c_str ()));
      }
  }
}

static void
info_skip_command (const char *arg, int from_tty)
{
  int num_printable_entries = 0;
  struct value_print_options opts;

  get_user_print_options (&opts);

  /* Count the number of rows in the table and see if we need space for a
     64-bit address anywhere.  */
  for (const skiplist_entry &e : skiplist_entries)
    if (arg == NULL || number_is_in_list (arg, e.number ()))
      num_printable_entries++;

  if (num_printable_entries == 0)
    {
      if (arg == NULL)
	current_uiout->message (_("Not skipping any files or functions.\n"));
      else
	current_uiout->message (
	  _("No skiplist entries found with number %s.\n"), arg);

      return;
    }

  ui_out_emit_table table_emitter (current_uiout, 6, num_printable_entries,
				   "SkiplistTable");

  current_uiout->table_header (5, ui_left, "number", "Num");   /* 1 */
  current_uiout->table_header (3, ui_left, "enabled", "Enb");  /* 2 */
  current_uiout->table_header (4, ui_right, "regexp", "Glob"); /* 3 */
  current_uiout->table_header (20, ui_left, "file", "File");   /* 4 */
  current_uiout->table_header (2, ui_right, "regexp", "RE");   /* 5 */
  current_uiout->table_header (40, ui_noalign, "function", "Function"); /* 6 */
  current_uiout->table_body ();

  for (const skiplist_entry &e : skiplist_entries)
    {
      QUIT;
      if (arg != NULL && !number_is_in_list (arg, e.number ()))
	continue;

      ui_out_emit_tuple tuple_emitter (current_uiout, "blklst-entry");
      current_uiout->field_signed ("number", e.number ()); /* 1 */

      if (e.enabled ())
	current_uiout->field_string ("enabled", "y"); /* 2 */
      else
	current_uiout->field_string ("enabled", "n"); /* 2 */

      if (e.file_is_glob ())
	current_uiout->field_string ("regexp", "y"); /* 3 */
      else
	current_uiout->field_string ("regexp", "n"); /* 3 */

      current_uiout->field_string ("file",
				   e.file ().empty () ? "<none>"
				   : e.file ().c_str (),
				   e.file ().empty ()
				   ? metadata_style.style ()
				   : file_name_style.style ()); /* 4 */
      if (e.function_is_regexp ())
	current_uiout->field_string ("regexp", "y"); /* 5 */
      else
	current_uiout->field_string ("regexp", "n"); /* 5 */

      current_uiout->field_string ("function",
				   e.function ().empty () ? "<none>"
				   : e.function ().c_str (),
				   e.function ().empty ()
				   ? metadata_style.style ()
				   : function_name_style.style ()); /* 6 */

      current_uiout->text ("\n");
    }
}

static void
skip_enable_command (const char *arg, int from_tty)
{
  bool found = false;

  for (skiplist_entry &e : skiplist_entries)
    if (arg == NULL || number_is_in_list (arg, e.number ()))
      {
	e.enable ();
	found = true;
      }

  if (!found)
    error (_("No skiplist entries found with number %s."), arg);
}

static void
skip_disable_command (const char *arg, int from_tty)
{
  bool found = false;

  for (skiplist_entry &e : skiplist_entries)
    if (arg == NULL || number_is_in_list (arg, e.number ()))
      {
	e.disable ();
	found = true;
      }

  if (!found)
    error (_("No skiplist entries found with number %s."), arg);
}

static void
skip_delete_command (const char *arg, int from_tty)
{
  bool found = false;

  for (auto it = skiplist_entries.begin (),
	 end = skiplist_entries.end ();
       it != end;)
    {
      const skiplist_entry &e = *it;

      if (arg == NULL || number_is_in_list (arg, e.number ()))
	{
	  it = skiplist_entries.erase (it);
	  found = true;
	}
      else
	++it;
    }

  if (!found)
    error (_("No skiplist entries found with number %s."), arg);
}

bool
skiplist_entry::do_skip_file_p (const symtab_and_line &function_sal) const
{
  bool result;

  /* Check first sole SYMTAB->FILENAME.  It may not be a substring of
     symtab_to_fullname as it may contain "./" etc.  */
  if (compare_filenames_for_search (function_sal.symtab->filename (),
				    m_file.c_str ()))
    result = true;

  /* Before we invoke realpath, which can get expensive when many
     files are involved, do a quick comparison of the basenames.  */
  else if (!basenames_may_differ
	   && filename_cmp (lbasename (function_sal.symtab->filename ()),
			    lbasename (m_file.c_str ())) != 0)
    result = false;
  else
    {
      /* Note: symtab_to_fullname caches its result, thus we don't have to.  */
      const char *fullname = symtab_to_fullname (function_sal.symtab);

      result = compare_filenames_for_search (fullname, m_file.c_str ());
    }

  return result;
}

/* The implementation of skiplist_entry::do_skip_gfile_p.  This exists as a
   separate function so that this function can be unit tested.

   PATTERN is the user supplied pattern held within the skiplist_entry, and
   RE is the compiled regexp version of PATTERN, created when the
   skiplist_entry was created.

   The two callbacks GET_FILENAME and GET_FULLNAME return the result of
   symtab::filename and symtab_to_fullname respectively.  These are
   provided as callbacks though so that the self tests don't need to create
   fake symtabs.

   Returns true if PATTERN matches the filename or fullname, and false
   otherwise.

   This function tries to avoid calling GET_FULLNAME as this can be more
   expensive.  The PATTERN will first be matched against the result of
   calling GET_FILENAME if possible.  */

static bool
do_skip_gfile_p (const std::string &pattern, const compiled_regex &re,
		 gdb::function_view<const char * ()> get_filename,
		 gdb::function_view<const char * ()> get_fullname)
{
  /* If basenames don't match then the full pattern cannot match.  The
     gdb_filename_fnmatch already handles case insensitive filesystems, and
     as we're only checking the basenames here, directory separators are
     not a problem.  */
  if (!basenames_may_differ
      && gdb_filename_fnmatch (lbasename (pattern.c_str ()),
			       lbasename (get_filename ()),
			       FNM_FILE_NAME | FNM_NOESCAPE) != 0)
    return false;

  /* If the pattern is absolute, e.g. starts with '/', then we're going to
     have to compare against the full filename, we can skip the check
     against the symtab filename.  */
  bool is_absolute_pattern = IS_ABSOLUTE_PATH (pattern.c_str ());

  /* The symtab's filename might not be the full filename, this will depend
     on how the symtab was compiled and/or how the DWARF was generated.
     But with gcc at least, compiling a relative filename results in a
     symtab with a relative filename.  However, in many cases, the skip
     pattern is also only a partial filename, and matches against the end
     part of the symtab filename, so rather than the (relatively expensive)
     fullname lookup, check first against the symtab filename.  */
  if (!basenames_may_differ && !is_absolute_pattern)
    {
      if (re.exec (get_filename (), 0, nullptr, 0) == 0)
	return true;
    }

  return re.exec (get_fullname (), 0, nullptr, 0) == 0;
}

bool
skiplist_entry::do_skip_gfile_p (const symtab_and_line &function_sal) const
{
  gdb_assert (m_compiled_file_regexp.has_value ());
  return ::do_skip_gfile_p (m_file, m_compiled_file_regexp.value (),
			    [&] () {
			      return function_sal.symtab->filename ();
			    },
			    [&] () {
			      return symtab_to_fullname (function_sal.symtab);
			    });
}

bool
skiplist_entry::skip_file_p (const symtab_and_line &function_sal) const
{
  if (m_file.empty ())
    return false;

  if (function_sal.symtab == NULL)
    return false;

  skip_debug_printf ("checking if file %s matches %sglob %s",
		     function_sal.symtab->filename (),
		     (m_file_is_glob ? "" : "non-"), m_file.c_str ());

  bool result;
  if (m_file_is_glob)
    result = do_skip_gfile_p (function_sal);
  else
    result = do_skip_file_p (function_sal);

  skip_debug_printf (result ? "yes" : "no");

  return result;
}

bool
skiplist_entry::skip_function_p (const char *function_name) const
{
  if (m_function.empty ())
    return false;

  bool result;

  if (m_function_is_regexp)
    {
      skip_debug_printf ("checking if function %s matches regex %s",
			 function_name, m_function.c_str ());

      gdb_assert (m_compiled_function_regexp);
      result
	= (m_compiled_function_regexp->exec (function_name, 0, NULL, 0) == 0);
    }
  else
    {
      skip_debug_printf ("checking if function %s matches non-regex %s",
			 function_name, m_function.c_str ());
      result = (strcmp_iw (function_name, m_function.c_str ()) == 0);
    }

  skip_debug_printf (result ? "yes" : "no");

  return result;
}

/* See skip.h.  */

bool
function_name_is_marked_for_skip (const char *function_name,
				  const symtab_and_line &function_sal)
{
  if (function_name == NULL)
    return false;

  for (const skiplist_entry &e : skiplist_entries)
    {
      if (!e.enabled ())
	continue;

      bool skip_by_file = e.skip_file_p (function_sal);
      bool skip_by_function = e.skip_function_p (function_name);

      /* If both file and function must match, make sure we don't errantly
	 exit if only one of them match.  */
      if (!e.file ().empty () && !e.function ().empty ())
	{
	  if (skip_by_file && skip_by_function)
	    return true;
	}
      /* Only one of file/function is specified.  */
      else if (skip_by_file || skip_by_function)
	return true;
    }

  return false;
}

/* Completer for skip numbers.  */

static void
complete_skip_number (cmd_list_element *cmd,
		      completion_tracker &completer,
		      const char *text, const char *word)
{
  size_t word_len = strlen (word);

  for (const skiplist_entry &entry : skiplist_entries)
    {
      gdb::unique_xmalloc_ptr<char> name = xstrprintf ("%d", entry.number ());
      if (strncmp (word, name.get (), word_len) == 0)
	completer.add_completion (std::move (name));
    }
}

/* Implementation of 'save skip' command.  */

static void
save_skip_command (const char *filename, int from_tty)
{
  if (filename == nullptr || *filename == '\0')
    error (_("Argument required (file name in which to save)"));

  gdb::unique_xmalloc_ptr<char> expanded_filename (tilde_expand (filename));
  stdio_file fp;
  if (!fp.open (expanded_filename.get (), "w"))
    error (_("Unable to open file '%s' for saving (%s)"),
	   expanded_filename.get (), safe_strerror (errno));

  for (const auto &entry : skiplist_entries)
    entry.print_recreate (&fp);
}

#if GDB_SELF_TEST

namespace selftests {

/* Define a single test of the do_skip_gfile_p function.  */
struct skip_gfile_test
{
  /* The glob pattern to match against the filename.  */
  const char *pattern;

  /* The filename is split into PREFIX and SUFFIX.  This reflects how GDB
     stores only part of the filename (as found in the DWARF) within the
     symtab as the 'filename'.  The PREFIX is the part GDB figures out from
     the compilation directory.  The full filename is created by
     concatenating PREFIX to SUFFIX.  */
  const char *prefix;
  const char *suffix;

  /* True if we expect PATTERN to match against the filename created from
     PREFIX and SUFFIX.  */
  bool expect_match;
};

/* Some tests check that case sensitivity works, these tests expect a
   glob to not match a particular filename.  On case insensitive file
   systems these globs will match.  We define this constant to use for
   those tests.  */

#ifdef HAVE_CASE_INSENSITIVE_FILE_SYSTEM
static constexpr bool false_if_case_sensitive = true;
#else
static constexpr bool false_if_case_sensitive = false;
#endif /* ! HAVE_CASE_INSENSITIVE_FILE_SYSTEM */

/* List of all do_skip_gfile_p tests.  */
static constexpr std::initializer_list<skip_gfile_test> skip_gfile_tests = {
  /* Basic glob feature testing.  */
  { "*", "/tmp/", "foo/hello.c", true },
  { "xx", "/tmp/", "foo/hello.c", false },
  { "hello.?", "/tmp/", "foo/hello.c", true },
  { "hello.?", "/tmp/", "foo/hello.cc", false },
  { "aa/bb*/file*.*c*", "/tmp/", "aa/bb/file.c", true },
  { "*.c", "/tmp/", "aa/bb/file.c", true },
  { "bb/*.c", "/tmp/", "aa/bb/file.c", true },
  { "ee/*.c", "/tmp/", "ee/bb/file.c", false },
  { "b/*.c", "/tmp/", "aa/bb/file.c", false },

  /* Testing the globstar '**' feature.  */
  { "ee/**/*.c", "/tmp/", "ee/bb/file.c", true },
  { "dd/**/**/*.c", "/tmp/", "dd/file.c", true },
  { "dd/**/**/*.c", "/tmp/", "dd/aa/file.c", true },
  { "dd/**/**/*.c", "/tmp/", "dd/aa/bb/file.c", true },
  { "dd/**/**/*.c", "/tmp/", "dd/aa/bb/cc/file.c", true },
  { "dd/**/**/*.c", "/tmp/", "dd/aa/bb/cc/dd/file.c", true },
  { "**/**/**/*.c", "/tmp/", "file.c", true },
  { "**/**/**/*.c", "/tmp/aa/", "file.c", true },
  { "**/**/**/*.c", "/tmp/aa/bb/", "file.c", true },
  { "**/**/**/*.c", "/tmp/aa/bb/cc/", "file.c", true },
  { "**/**/**/*.c", "/tmp/aa/bb/cc/dd/", "file.c", true },
  { "/tmp/**", "/tmp/", "file.c", true },
  { "/tmp/**", "/tmp/aa/", "file.c", true },

  /* When '**' appears in the middle of a path component (not after
     '/' or at the start), it is not globstar but two regular '*'
     wildcards.  */
  { "a**b/*.c", "/tmp/", "ab/foo.c", true },
  { "a**b/*.c", "/tmp/", "axxb/foo.c", true },
  { "a**b/*.c", "/tmp/", "a/b/foo.c", false },

  { "[a-c]*.h", "/tmp/", "axxx.h", true },
  { "*[[:digit:]]*.h", "/tmp/", "xx1xx.h", true },
  { "[!a-c]*.h", "/tmp/", "axxx.h", false },
  { "*[]].h", "/tmp/", "xx].h", true },
  { "*[!]].h", "/tmp/", "xx].h", false },

  /* An absolute pattern must match the full filename.  */
  { "/tmp/foo.cc", "/blah/tmp/", "foo.cc", false },
  { "/blah/tmp/foo.cc", "/blah/tmp/", "foo.cc", true },

  /* Characters that are special in regular expressions but should be
     treated as literal characters in glob patterns.  */

  /* The '+' character means 'one or more' in a regular expression.  */
  { "file+.c", "/tmp/", "file+.c", true },
  { "dir+/*.c", "/tmp/", "dir+/foo.c", true },
  { "dir+/*.c", "/tmp/", "dirr/foo.c", false },
  { "dir+/*.c", "/tmp/", "dirrr/foo.c", false },

  /* The '(' and ')' characters form capture groups in regexp.  */
  { "(foo)/*.c", "/tmp/", "(foo)/bar.c", true },
  { "(foo)/*.c", "/tmp/", "foo/bar.c", false },

  /* The '|' character means alternation in a regexp.  */
  { "a|b/*.c", "/tmp/", "a|b/foo.c", true },
  { "a|b/*.c", "/tmp/", "b/foo.c", false },

  /* The '{' and '}' characters form interval expressions in regexp.
     (e.g., a{2} matches "aa").  */
  { "a{2}/*.c", "/tmp/", "a{2}/foo.c", true },
  { "a{2}/*.c", "/tmp/", "aa/foo.c", false },

  /* The '$' and '^' characters are anchors in regexp.  */
  { "file$.c", "/tmp/", "file$.c", true },
  { "^file.c", "/tmp/", "^file.c", true },

  /* Bracket expressions that mention a '/' are not valid within a glob and
     POSIX requires that they be treated as literal content.  */
  { "aa[.-/]bb/*.c", "/tmp/", "aa.bb/foo.c", false },
  { "aa[.-/]bb/*.c", "/tmp/", "aa[.-/]bb/foo.c", true },
  { "aa[/-/]bb/*.c", "/tmp/", "aa/bb/foo.c", false },
  { "aa[/-/]bb/*.c", "/tmp/", "aa[/-/]bb/foo.c", true },
  { "aa[/-1]bb/*.c", "/tmp/", "aa[/-1]bb/foo.c", true },
  { "aa[/-1]bb/*.c", "/tmp/", "aa0bb/foo.c", false },

  /* Within bracket expressions, ranges that span '/' should not match the
     '/' character.  */
  { "aa[.-0]bb/*.c", "/tmp/", "aa/bb/foo.c", false },
  { "aa[.-0]bb/*.c", "/tmp/", "aa.bb/foo.c", true },
  { "aa[.-0]bb/*.c", "/tmp/", "aa0bb/foo.c", true },
  { "aa[.-1]bb/*.c", "/tmp/", "aa.bb/foo.c", true },
  { "aa[.-1]bb/*.c", "/tmp/", "aa0bb/foo.c", true },
  { "aa[.-1]bb/*.c", "/tmp/", "aa1bb/foo.c", true },
  { "aa[.-1]bb/*.c", "/tmp/", "aa/bb/foo.c", false },
  { "aa[.-1]bb/*.c", "/tmp/", "aa2bb/foo.c", false },
  { "aa[,-0]bb/*.c", "/tmp/", "aa,bb/foo.c", true },
  { "aa[,-0]bb/*.c", "/tmp/", "aa-bb/foo.c", true },
  { "aa[,-0]bb/*.c", "/tmp/", "aa.bb/foo.c", true },
  { "aa[,-0]bb/*.c", "/tmp/", "aa0bb/foo.c", true },
  { "aa[,-0]bb/*.c", "/tmp/", "aa/bb/foo.c", false },
  { "aa[,-0]bb/*.c", "/tmp/", "aa1bb/foo.c", false },

  /* Negated bracket expressions should also not match '/'.  POSIX says
     that, within a glob, a bracket expression starting with '^' is
     undefined, but fnmatch (and bash) treat '^' the same as '!'.  */
  { "a[!a]b/*.c", "/tmp/", "a/b/foo.c", false },
  { "a[!a]b/*.c", "/tmp/", "axb/foo.c", true },
  { "a[^a]b/*.c", "/tmp/", "a/b/foo.c", false },
  { "a[^a]b/*.c", "/tmp/", "axb/foo.c", true },

  /* Historically GDB used fnmatch to perform glob matching, and passed
     FNM_NOESCAPE as an option to fnmatch, which means that '\' is not an
     escape character, but should be treated as a literal character.  When
     we switch to using regexp instead of fnmatch we preserved this
     behaviour.

     Given the above '\*' in a glob doesn't escape the '*', the '\' is
     literal and '*' is still a wildcard.  */
  { "a\\b.c", "/tmp/", "a\\b.c", true },
  { "a\\*.c", "/tmp/", "a\\foo.c", true },
  { "a\\*.c", "/tmp/", "a*.c", false },

  /* As with the previous test, historically FNM_PERIOD was not used, so
     '*' and '?' should match a leading period in a filename.  */
  { "*.c", "/tmp/", ".hidden.c", true },
  { "?idden.c", "/tmp/", ".idden.c", true },

  /* An unmatched '[' should be treated as a literal character.  */
  { "[.c", "/tmp/", "[.c", true },
  { "[.c", "/tmp/", "x.c", false },

  /* A '-' at the start or end of a bracket expression is literal.  */
  { "[-ab]*.c", "/tmp/", "-foo.c", true },
  { "[ab-]*.c", "/tmp/", "-foo.c", true },

  /* A '-' immediately after a range is literal, not a second range
     operator.  So [a-c-f] is the range 'a'-'c', a literal '-', and
     a literal 'f'.  Characters between 'c' and 'f' (like 'e') that
     are outside the range should not match.  */
  { "[a-c-f]dir/*.c", "/tmp/", "bdir/foo.c", true },
  { "[a-c-f]dir/*.c", "/tmp/", "-dir/foo.c", true },
  { "[a-c-f]dir/*.c", "/tmp/", "fdir/foo.c", true },
  { "[a-c-f]dir/*.c", "/tmp/", "edir/foo.c", false },

  /* In a POSIX glob bracket expression, a leading '^' is undefined.
     However fnmatch (and bash) treat this the same as '!', that is, as
     match negation.  GDB copies this behaviour.  */
  { "[^abc]dir/*.c", "/tmp/", "adir/foo.c", false },
  { "[^abc]dir/*.c", "/tmp/", "^dir/foo.c", true },
  { "[^abc]dir/*.c", "/tmp/", "xdir/foo.c", true },

  /* Due to the above '^^' within a bracket means match everything except
     '^' (and '/' of course).  */
  { "[^^]dir/*.c", "/tmp/", "^dir/foo.c", false },
  { "[^^]dir/*.c", "/tmp/", "xdir/foo.c", true },
  { "aa[^^]bb/*.c", "/tmp/", "aa/bb/foo.c", false },
  { "aa[^^]bb/*.c", "/tmp/", "aa^bb/foo.c", false },
  { "aa[^^]bb/*.c", "/tmp/", "aa-bb/foo.c", true },

  /* The glob '[!^]' is the same as the previous, but uses the official
     POSIX glob '!' character for negation.  */
  { "[!^]dir/*.c", "/tmp/", "^dir/foo.c", false },
  { "[!^]dir/*.c", "/tmp/", "adir/foo.c", true },

  /* The globs '[]' and '[!]' are invalid as the ']' is considered a
     character within the bracket expression, this means that the bracket
     expression is never terminated.  This is handled by treating the
     characters as literals.  */
  { "[!]dir/*.c", "/tmp/", "[!]dir/foo.c", true },
  { "[!]dir/*.c", "/tmp/", "adir/foo.c", false },
  { "[^]dir/*.c", "/tmp/", "[^]dir/foo.c", true },
  { "[^]dir/*.c", "/tmp/", "adir/foo.c", false },
  { "aa[]bb/*.c", "/tmp/", "aabb/foo.c", false },
  { "aa[]bb/*.c", "/tmp/", "aa[]bb/foo.c", true },

  /* When '^' is NOT the first character in a bracket expression, it is
     just a character to match or not match.  */
  { "[abc^]dir/*.c", "/tmp/", "^dir/foo.c", true },
  { "[abc^]dir/*.c", "/tmp/", "adir/foo.c", true },
  { "[abc^]dir/*.c", "/tmp/", "xdir/foo.c", false },
  { "[!abc^]dir/*.c", "/tmp/", "^dir/foo.c", false },
  { "[!abc^]dir/*.c", "/tmp/", "adir/foo.c", false },
  { "[!abc^]dir/*.c", "/tmp/", "xdir/foo.c", true },

  /* Negated bracket expression with ']' as the first element.  Test with
     both '!' and '^' for the reasons discussed above.  */
  { "[!]a]dir/*.c", "/tmp/", "bdir/foo.c", true },
  { "[!]a]dir/*.c", "/tmp/", "]dir/foo.c", false },
  { "[!]a]dir/*.c", "/tmp/", "adir/foo.c", false },
  { "[^]a]dir/*.c", "/tmp/", "bdir/foo.c", true },
  { "[^]a]dir/*.c", "/tmp/", "]dir/foo.c", false },
  { "[^]a]dir/*.c", "/tmp/", "adir/foo.c", false },

  /* Within a bracket expression, a range that starts with the negation
     character is not treated like a range, so [!-a] means match everything
     except '-' and 'a'.  Test with both '!' and '^' for the reasons
     discussed above.  */
  { "[!-a]dir/*.c", "/tmp/", "_dir/foo.c", true },
  { "[!-a]dir/*.c", "/tmp/", "^dir/foo.c", true },
  { "[!-a]dir/*.c", "/tmp/", "adir/foo.c", false },
  { "[!-a]dir/*.c", "/tmp/", "bdir/foo.c", true },
  { "[^-a]dir/*.c", "/tmp/", "_dir/foo.c", true },
  { "[^-a]dir/*.c", "/tmp/", "^dir/foo.c", true },
  { "[^-a]dir/*.c", "/tmp/", "adir/foo.c", false },
  { "[^-a]dir/*.c", "/tmp/", "bdir/foo.c", true },
  { "[!-]dir/*.c", "/tmp/", "adir/foo.c", true },
  { "[!-]dir/*.c", "/tmp/", "-dir/foo.c", false },
  { "[^-]dir/*.c", "/tmp/", "adir/foo.c", true },
  { "[^-]dir/*.c", "/tmp/", "-dir/foo.c", false },

  /* Within a bracket expression, test ranges starting and ending with
     '-'.  */
  { "aa[--0]bb/*.c", "/tmp/", "aa-bb/foo.c", true },
  { "aa[--0]bb/*.c", "/tmp/", "aa.bb/foo.c", true },
  { "aa[--0]bb/*.c", "/tmp/", "aa0bb/foo.c", true },
  { "aa[--0]bb/*.c", "/tmp/", "aa/bb/foo.c", false },
  { "aa[+--]bb/*.c", "/tmp/", "aa-bb/foo.c", true },
  { "aa[+--]bb/*.c", "/tmp/", "aa+bb/foo.c", true },
  { "aa[+--]bb/*.c", "/tmp/", "aa,bb/foo.c", true },
  { "aa[+--]bb/*.c", "/tmp/", "aa.bb/foo.c", false },

  /* Basic character class matching.  [[:digit:]] matches any decimal
     digit character.  */
  { "[[:digit:]]dir/*.c", "/tmp/", "1dir/foo.c", true },
  { "[[:digit:]]dir/*.c", "/tmp/", "adir/foo.c", false },
  { "adir/[[:digit:]]*.c", "/tmp/", "adir/1foo.c", true },
  { "[[:digit:]]*.c", "/tmp/", "adir/1foo.c", true },

  /* [[:alpha:]] matches any alphabetic character.  */
  { "[[:alpha:]]dir/*.c", "/tmp/", "adir/foo.c", true },
  { "[[:alpha:]]dir/*.c", "/tmp/", "1dir/foo.c", false },

  /* [[:upper:]] and [[:lower:]] match upper- and lowercase letters
     respectively.  */
  { "[[:upper:]]dir/*.c", "/tmp/", "Adir/foo.c", true },
  { "[[:upper:]]dir/*.c", "/tmp/", "adir/foo.c", false_if_case_sensitive },
  { "[[:lower:]]dir/*.c", "/tmp/", "adir/foo.c", true },
  { "[[:lower:]]dir/*.c", "/tmp/", "Adir/foo.c", false_if_case_sensitive },

  /* Negated character class: [![:digit:]] matches non-digit characters,
     but should not match '/'.  */
  { "a[![:digit:]]b/*.c", "/tmp/", "axb/foo.c", true },
  { "a[![:digit:]]b/*.c", "/tmp/", "a1b/foo.c", false },
  { "a[![:digit:]]b/*.c", "/tmp/", "a/b/foo.c", false },

  /* Character class combined with literal characters.  [[:digit:]ab]
     should match any digit, or 'a', or 'b'.  */
  { "[[:digit:]ab]dir/*.c", "/tmp/", "1dir/foo.c", true },
  { "[[:digit:]ab]dir/*.c", "/tmp/", "adir/foo.c", true },
  { "[[:digit:]ab]dir/*.c", "/tmp/", "bdir/foo.c", true },
  { "[[:digit:]ab]dir/*.c", "/tmp/", "xdir/foo.c", false },

  /* Character class combined with a character range.  [[:digit:]a-f]
     should match any digit or any letter from 'a' to 'f'.  */
  { "[[:digit:]a-f]dir/*.c", "/tmp/", "1dir/foo.c", true },
  { "[[:digit:]a-f]dir/*.c", "/tmp/", "cdir/foo.c", true },
  { "[[:digit:]a-f]dir/*.c", "/tmp/", "gdir/foo.c", false },

  /* Multiple character classes in one bracket expression.
     [[:digit:][:alpha:]] should match digits and letters.  */
  { "[[:digit:][:alpha:]]dir/*.c", "/tmp/", "1dir/foo.c", true },
  { "[[:digit:][:alpha:]]dir/*.c", "/tmp/", "adir/foo.c", true },
  { "[[:digit:][:alpha:]]dir/*.c", "/tmp/", "-dir/foo.c", false },

  /* Character class appearing in the basename portion of the
     pattern.  */
  { "dir/file[[:digit:]].c", "/tmp/", "dir/file1.c", true },
  { "dir/file[[:digit:]].c", "/tmp/", "dir/filea.c", false },

  { "[W-_]dir/*.c", "/tmp/", "_dir/foo.c", true },
  { "[A-C]dir/*.c", "/tmp/", "Adir/foo.c", true },
  { "[]-_]dir/*.c", "/tmp/", "^dir/foo.c", true },

#ifdef HAVE_CASE_INSENSITIVE_FILE_SYSTEM
  /* Basic case insensitive matching.  */
  { "Dir/*.c", "/tmp/", "dir/foo.c", true },
  { "DIR/*.c", "/tmp/", "dir/foo.c", true },
  { "Dir/*.c", "/tmp/", "other/foo.c", false },
  { "dir/*.c", "/tmp/", "dir/FOO.c", true },
  { "dir/*.c", "/tmp/", "DIR/FOO.c", true },

  /* Upper case character range [A-C] is converted to [a-c] on a case
     insensitive file system.  */
  { "[A-C]dir/*.c", "/tmp/", "adir/foo.c", true },
  { "[A-C]dir/*.c", "/tmp/", "cdir/foo.c", true },
  { "[A-C]dir/*.c", "/tmp/", "Bdir/foo.c", true },
  { "[A-C]dir/*.c", "/tmp/", "ddir/foo.c", false },

  /* A range that overlaps into the upper case characters [=-D] will be
     split into two: [=-@] and [A-D].  With REG_ICASE the [A-D] will match
     both [A-D] and [a-d].  */
  { "[=-D]dir/*.c", "/tmp/", "@dir/foo.c", true },
  { "[=-D]dir/*.c", "/tmp/", "=dir/foo.c", true },
  { "[=-D]dir/*.c", "/tmp/", "adir/foo.c", true },
  { "[=-D]dir/*.c", "/tmp/", "ddir/foo.c", true },
  { "[=-D]dir/*.c", "/tmp/", "Adir/foo.c", true },
  { "[=-D]dir/*.c", "/tmp/", "Ddir/foo.c", true },

  /* A similar thing happens with ranges that overlap the end of the
     upper case character range.  */
  { "[W-_]dir/*.c", "/tmp/", "[dir/foo.c", true },
  { "[W-_]dir/*.c", "/tmp/", "_dir/foo.c", true },
  { "[W-_]dir/*.c", "/tmp/", "wdir/foo.c", true },
  { "[W-_]dir/*.c", "/tmp/", "zdir/foo.c", true },
  { "[W-_]dir/*.c", "/tmp/", "Wdir/foo.c", true },
  { "[W-_]dir/*.c", "/tmp/", "Zdir/foo.c", true },

  /* A full upper case range [A-Z] is converted to [a-z].  */
  { "[A-Z]dir/*.c", "/tmp/", "mdir/foo.c", true },
  { "[A-Z]dir/*.c", "/tmp/", "1dir/foo.c", false },

  /* Negated bracket expression with an upper case range.  */
  { "[!A-C]dir/*.c", "/tmp/", "ddir/foo.c", true },
  { "[!A-C]dir/*.c", "/tmp/", "adir/foo.c", false },
  { "[!A-C]dir/*.c", "/tmp/", "Bdir/foo.c", false },

  /* Upper case literal characters in bracket expressions are
     converted to lower case.  */
  { "[ABCD]dir/*.c", "/tmp/", "adir/foo.c", true },
  { "[ABCD]dir/*.c", "/tmp/", "Cdir/foo.c", true },
  { "[ABCD]dir/*.c", "/tmp/", "edir/foo.c", false },

  { "[[:upper:]]dir/*.c", "/tmp/", "adir/foo.c", true },
#endif /* HAVE_CASE_INSENSITIVE_FILE_SYSTEM */

#ifdef HAVE_DOS_BASED_FILE_SYSTEM
  /* Tests for DOS-based file systems.  On a DOS-based file system
     the backslash is treated as a directory separator in addition to
     the forward slash.  */

  /* Backslash in the pattern is treated as a directory separator.  */
  { "dir\\*.c", "/tmp/", "dir\\foo.c", true },
  { "dir\\*.c", "/tmp/", "dir\\bar.c", true },
  { "dir\\*.c", "/tmp/", "other\\foo.c", false },

  /* The regexp should match both backslash and forward slashes.  */
  { "dir/*.c", "/tmp/", "dir\\foo.c", true },

  /* Globstar with backslash as the directory separator.  */
  { "dir\\**\\*.c", "/tmp/", "dir\\sub\\foo.c", true },
  { "dir\\**\\*.c", "/tmp/", "dir\\a\\b\\foo.c", true },

  /* An absolute DOS path with a drive letter.  */
  { "C:\\dir\\*.c", "", "C:\\dir\\foo.c", true },
  { "C:\\dir\\*.c", "", "C:\\other\\foo.c", false },
#endif /* HAVE_DOS_BASED_FILE_SYSTEM */

#if defined HAVE_DOS_BASED_FILE_SYSTEM \
  && defined HAVE_CASE_INSENSITIVE_FILE_SYSTEM
  /* Combine case insensitivity together with backslash directory
     separators.  The pattern has upper case and forward slash, the
     filename has mixed case and backslash.  */
  { "Dir/*.c", "/tmp/", "dir\\foo.c", true },
  { "dir/*.c", "/tmp/", "Dir\\foo.c", true },
  { "Dir\\sub\\*.c", "/tmp/", "Dir\\sub\\bar.c", true },
#endif /* HAVE_DOS_BASED_FILE_SYSTEM && HAVE_CASE_INSENSITIVE_FILE_SYSTEM */
};

/* The skip_gfile_matching unit tests.  */

static void
test_skip_gfile_matching ()
{
  int failure_count = 0;
  bool first = true;
  for (const auto &test : skip_gfile_tests)
    {
      std::string pattern (test.pattern);
      std::string filename (test.suffix);
      std::string fullname = std::string (test.prefix) + filename;

      if (run_verbose ())
	{
	  if (!first)
	    debug_printf ("\n");
	  else
	    first = false;
	  debug_printf ("Pattern (%s)\n", pattern.c_str ());
	  debug_printf ("Filename (%s)\n", filename.c_str ());
	  debug_printf ("Fullname (%s)\n", fullname.c_str ());
	}

      std::string file_re = glob_to_regexp (pattern);

      if (run_verbose ())
	debug_printf ("Regexp (%s)\n", file_re.c_str ());

      compiled_regex re (file_re.c_str (), file_glob_regexp_flags,
			 _("skip glob testing"));

      bool matched = do_skip_gfile_p (pattern, re,
				      [&] () { return filename.c_str (); },
				      [&] () { return fullname.c_str (); });

      bool success = matched == test.expect_match;

      if (run_verbose ())
	debug_printf ("Matched: %s%s\n", (matched ? "Yes" : "No"),
		      (success ? ""
		       : string_printf ("\t[Expected: %s]",
					(test.expect_match
					 ? "Yes" : "No")).c_str ()));

      if (matched != test.expect_match)
	failure_count++;
    }

  if (run_verbose ())
    debug_printf ("Failed: %d\n", failure_count);

  SELF_CHECK (failure_count == 0);
}

}  /* namespace selftests */

#endif /* GDB_SELF_TEST */

INIT_GDB_FILE (step_skip)
{
  static struct cmd_list_element *skiplist = NULL;
  struct cmd_list_element *c;

  c = add_prefix_cmd ("skip", class_breakpoint, skip_command, _("\
Ignore a function while stepping.\n\
\n\
Usage: skip [FUNCTION-NAME]\n\
       skip [FILE-SPEC] [FUNCTION-SPEC]\n\
If no arguments are given, ignore the current function.\n\
\n\
FILE-SPEC is one of:\n\
       -file FILE-NAME\n\
       -gfile GLOB-FILE-PATTERN\n\
FUNCTION-SPEC is one of:\n\
       -function FUNCTION-NAME\n\
       -rfunction FUNCTION-NAME-REGULAR-EXPRESSION"),
		  &skiplist, 1, &cmdlist);
  set_cmd_completer_handle_brkchars (c, skip_command_completer);

  c = add_cmd ("file", class_breakpoint, skip_file_command, _("\
Ignore a file while stepping.\n\
Usage: skip file [FILE-NAME]\n\
If no filename is given, ignore the current file."),
	       &skiplist);
  set_cmd_completer (c, deprecated_filename_completer);

  c = add_cmd ("function", class_breakpoint, skip_function_command, _("\
Ignore a function while stepping.\n\
Usage: skip function [FUNCTION-NAME]\n\
If no function name is given, skip the current function."),
	       &skiplist);
  set_cmd_completer (c, location_completer);

  c = add_cmd ("enable", class_breakpoint, skip_enable_command, _("\
Enable skip entries.\n\
Usage: skip enable [NUMBER | RANGE]...\n\
You can specify numbers (e.g. \"skip enable 1 3\"),\n\
ranges (e.g. \"skip enable 4-8\"), or both (e.g. \"skip enable 1 3 4-8\").\n\n\
If you don't specify any numbers or ranges, we'll enable all skip entries."),
	       &skiplist);
  set_cmd_completer (c, complete_skip_number);
  add_alias_cmd ("skip", c, class_breakpoint, 0, &enablelist);

  c = add_cmd ("disable", class_breakpoint, skip_disable_command, _("\
Disable skip entries.\n\
Usage: skip disable [NUMBER | RANGE]...\n\
You can specify numbers (e.g. \"skip disable 1 3\"),\n\
ranges (e.g. \"skip disable 4-8\"), or both (e.g. \"skip disable 1 3 4-8\").\n\n\
If you don't specify any numbers or ranges, we'll disable all skip entries."),
	       &skiplist);
  set_cmd_completer (c, complete_skip_number);
  add_alias_cmd ("skip", c, class_breakpoint, 0, &disablelist);

  c = add_cmd ("delete", class_breakpoint, skip_delete_command, _("\
Delete skip entries.\n\
Usage: skip delete [NUMBER | RANGES]...\n\
You can specify numbers (e.g. \"skip delete 1 3\"),\n\
ranges (e.g. \"skip delete 4-8\"), or both (e.g. \"skip delete 1 3 4-8\").\n\n\
If you don't specify any numbers or ranges, we'll delete all skip entries."),
	       &skiplist);
  set_cmd_completer (c, complete_skip_number);
  add_alias_cmd ("skip", c, class_breakpoint, 0, &deletelist);

  add_info ("skip", info_skip_command, _("\
Display the status of skips.\n\
Usage: info skip [NUMBER | RANGES]...\n\
You can specify numbers (e.g. \"info skip 1 3\"),\n\
ranges (e.g. \"info skip 4-8\"), or both (e.g. \"info skip 1 3 4-8\").\n\n\
If you don't specify any numbers or ranges, we'll show all skips."));
  set_cmd_completer (c, complete_skip_number);

  add_setshow_boolean_cmd ("skip", class_maintenance,
			   &debug_skip, _("\
Set whether to print the debug output about skipping files and functions."),
			   _("\
Show whether the debug output about skipping files and functions is printed."),
			   _("\
When non-zero, debug output about skipping files and functions is displayed."),
			   NULL, NULL,
			   &setdebuglist, &showdebuglist);

  c = add_cmd ("skip", no_class, save_skip_command, _("\
Save current skips as a script.\n\
Usage: save skip FILE\n\
Use the 'source' command in another debug session to restore them."),
	       &save_cmdlist);
  set_cmd_completer (c, deprecated_filename_completer);

#if GDB_SELF_TEST
  selftests::register_test ("skip_gfile_matching",
			    selftests::test_skip_gfile_matching);
#endif
}
