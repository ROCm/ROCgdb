/* Support code for the Ada expression parser, for GDB.

   Copyright (C) 1986-2026 Free Software Foundation, Inc.

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

#include "ada-exp-parser.h"
#include "ada-exp-parser-gen.h"
#include "ada-lang.h"
#include "block.h"
#include "cli/cli-style.h"
#include "gdbsupport/gdb_regex.h"
#include "gdbsupport/scoped_restore.h"
#include "gdbsupport/selftest.h"
#include "parser-defs.h"
#include "inferior.h"

using namespace expr;

/* The entry point of the bison/yacc-generated parser, defined in
   ada-exp-parser-gen.c.  Bison produces a declaration for ada_yyparse in
   ada-exp-parser-gen.h, but byacc does not, hence this declaration.  */

int ada_yyparse ();

/* Likewise, byacc does not produce a declaration for ada_yydebug.  */

extern int ada_yydebug;

/* A regular expression matching Ada numbers.  */

static const char number_regex_text[] =
  /* We have to match the first character.  */
  "^"

  /* Hex number.  */
  "(0x[0-9a-f][0-9a-f]*)"
#define HEX_NUMBER 1
  "|"

  /* Floating point number with positive exponent.  */
  "([0-9][0-9_]*e[+]?[0-9][0-9_]*)"
#define FLOAT_POSEXP 2
  "|"

  /* Floating point number with optional exponent.  */
  "([0-9][0-9_]*\\.[0-9][0-9_]*(e[+-][0-9][0-9_]*)?)"
#define FLOAT_OPTEXP 3
  "|"

  /* Floating point numbers but in another base.  The "llf" is a gdb
     extension, see lex_number.  The regexp is a little funny so that
     it can match only 0, 1, or 2 "l"s.  */
  "([0-9][0-9_]*(l?l?f)?#[0-9a-f][0-9a-f_]*#(e[+]?[0-9][0-9_]*)?)"
#define FLOAT_EXTENSION 5
#define EXPONENT 7
  "|"

  /* Based real literals.  */
  "([0-9][0-9_]*#[0-9a-f][0-9a-f_]*\\.[0-9a-f][0-9a-f_]*#(e[+-][0-9][0-9_]*)?)"
#define BASED_REAL 8
  "|"

  /* Lowly integers.  */
  "[0-9][0-9_]*";

/* The number of subexpressions to allocate space for, including the
   "0th" whole match subexpression.  */
#define NUM_SUBEXPRESSIONS 10

/* The compiled number-matching regex.  */

static compiled_regex number_regex (number_regex_text,
				    REG_EXTENDED | REG_ICASE,
				    _("internal error"));

/* A regular expression matching an Ada operator, surrounded by
   quotes.  */

static const char operator_regex_text[] =
  "^\"(<=|>=|\\*\\*|/=|and|or|xor|not|mod|rem|abs|[-+*/=<>&])\"";

/* The number of subexpressions to allocate space for, including the
   "0th" whole match subexpression.  */
#define NUM_OPER_SUBEXPRESSIONS 2

/* The compiled operator-matching regex.  */

static compiled_regex operator_regex (operator_regex_text,
				      REG_EXTENDED | REG_ICASE,
				      _("internal error"));

namespace ada_exp_parser
{

/* Map a string to a code.  */
struct code_entry
{
  const char *name;
  int code;
};

static const struct code_entry attributes[] =
{
  { "address", TICK_ADDRESS },
  { "unchecked_access", TICK_ACCESS },
  { "unrestricted_access", TICK_ACCESS },
  { "access", TICK_ACCESS },
  { "first", TICK_FIRST },
  { "last", TICK_LAST },
  { "length", TICK_LENGTH },
  { "max", TICK_MAX },
  { "min", TICK_MIN },
  { "modulus", TICK_MODULUS },
  { "object_size", TICK_OBJECT_SIZE },
  { "pos", TICK_POS },
  { "range", TICK_RANGE },
  { "size", TICK_SIZE },
  { "tag", TICK_TAG },
  { "val", TICK_VAL },
  { "enum_rep", TICK_ENUM_REP },
  { "enum_val", TICK_ENUM_VAL },
};

/* Expression completer for attributes.  */
struct ada_tick_completer : public expr_completion_base
{
  explicit ada_tick_completer (std::string &&name)
    : m_name (std::move (name))
  {
  }

  bool complete (struct expression *exp,
		 completion_tracker &tracker) override
  {
    completion_list output;
    for (const auto &item : attributes)
      {
	if (strncasecmp (item.name, m_name.c_str (), m_name.length ()) == 0)
	  output.push_back (make_unique_xstrdup (item.name));
      }
    tracker.add_completions (std::move (output));
    return true;
  }

private:

  std::string m_name;
};

/* See ada-exp-parser.h.  */

struct parser_state *pstate;

/* See ada-exp-parser.h.  */

struct ada_parse_state *ada_parser;

/* Copy S2 to S1, removing all underscores, and downcasing all
   letters.  Stops at END.  */

static void
canonicalize_numeral (char *s1, const char *s2, const char *end)
{
  for (; s2 != end; s2 += 1)
    {
      if (*s2 != '_')
	{
	  *s1 = c_tolower(*s2);
	  s1 += 1;
	}
    }
  s1[0] = '\000';
}

/* See ada-exp-parser.h.  */

int
ada_parse_state::process_int (const char *base_str, const char *num,
			      const char *exp_str)
{
  long exp;
  int base;
  /* For the based literal with an "f" prefix, we'll return a
     floating-point number.  This counts the number of "l"s seen,
     to decide the width of the floating-point number to return.  -1
     means no "f".  */
  int floating_point_l_count = -1;

  if (base_str == NULL)
    base = 10;
  else
    {
      char *end_of_base;
      base = strtol (base_str, &end_of_base, 10);
      if (base < 2 || base > 16)
	error (_("Invalid base: %d."), base);
      while (*end_of_base == 'l')
	{
	  ++floating_point_l_count;
	  ++end_of_base;
	}
      /* This assertion is ensured by the pattern.  */
      gdb_assert (floating_point_l_count == -1 || *end_of_base == 'f');
      if (*end_of_base == 'f')
	{
	  ++end_of_base;
	  ++floating_point_l_count;
	}
      /* This assertion is ensured by the pattern.  */
      gdb_assert (*end_of_base == '#');
    }

  if (exp_str == NULL)
    exp = 0;
  else
    exp = strtol (exp_str, nullptr, 10);

  gdb_mpz result;
  while (c_isxdigit (*num))
    {
      int dig = fromhex (*num);
      if (dig >= base)
	error (_("Invalid digit \"%c\" in based literal"), *num);
      result *= base;
      result += dig;
      ++num;
    }

  while (exp > 0)
    {
      result *= base;
      exp -= 1;
    }

  if (floating_point_l_count > -1)
    {
      struct type *fp_type;
      if (floating_point_l_count == 0)
	fp_type = language_lookup_primitive_type (pstate->language (),
						  pstate->gdbarch (),
						  "float");
      else if (floating_point_l_count == 1)
	fp_type = language_lookup_primitive_type (pstate->language (),
						  pstate->gdbarch (),
						  "long_float");
      else
	{
	  /* This assertion is ensured by the pattern.  */
	  gdb_assert (floating_point_l_count == 2);
	  fp_type = language_lookup_primitive_type (pstate->language (),
						    pstate->gdbarch (),
						    "long_long_float");
	}

      ada_yylval.typed_val_float.type = fp_type;
      result.write (gdb::make_span (ada_yylval.typed_val_float.val,
				    fp_type->length ()),
		    type_byte_order (fp_type),
		    true);

      return FLOAT;
    }

  const gdb_mpz *value = push_integer (std::move (result));

  int int_bits = gdbarch_int_bit (pstate->gdbarch ());
  int long_bits = gdbarch_long_bit (pstate->gdbarch ());
  int long_long_bits = gdbarch_long_long_bit (pstate->gdbarch ());

  if (fits_in_type (1, *value, int_bits, true))
    ada_yylval.typed_val.type = parse_type (pstate)->builtin_int;
  else if (fits_in_type (1, *value, long_bits, true))
    ada_yylval.typed_val.type = parse_type (pstate)->builtin_long;
  else if (fits_in_type (1, *value, long_bits, false))
    ada_yylval.typed_val.type
      = builtin_type (pstate->gdbarch ())->builtin_unsigned_long;
  else if (fits_in_type (1, *value, long_long_bits, true))
    ada_yylval.typed_val.type = parse_type (pstate)->builtin_long_long;
  else if (fits_in_type (1, *value, long_long_bits, false))
    ada_yylval.typed_val.type
      = builtin_type (pstate->gdbarch ())->builtin_unsigned_long_long;
  else if (fits_in_type (1, *value, 128, true))
    ada_yylval.typed_val.type
      = language_lookup_primitive_type (pstate->language (),
					pstate->gdbarch (),
					"long_long_long_integer");
  else if (fits_in_type (1, *value, 128, false))
    ada_yylval.typed_val.type
      = language_lookup_primitive_type (pstate->language (),
					pstate->gdbarch (),
					"unsigned_long_long_long_integer");
  else
    error (_("Integer literal out of range"));

  ada_yylval.typed_val.val = value;
  return INT;
}

/* See ada-exp-parser.h.  */

int
ada_parse_state::process_real (const char *num0)
{
  ada_yylval.typed_val_float.type = parse_type (pstate)->builtin_long_double;

  bool parsed = parse_float (num0, strlen (num0),
			     ada_yylval.typed_val_float.type,
			     ada_yylval.typed_val_float.val);
  gdb_assert (parsed);
  return FLOAT;
}


/* See ada-exp-parser.h.  */

struct stoken
ada_parse_state::process_id (std::string_view name0)
{
  char *name = (char *) obstack_alloc (&temp_space, name0.size () + 11);
  struct stoken result;

  result.ptr = name;
  while (!name0.empty () && c_isspace (name0.back ()))
    name0.remove_suffix (1);

  if (!name0.empty ()
      && (name0[0] == '<' || name0.find ("___") != std::string_view::npos))
    {
      size_t len = name0.size ();
      strncpy (name, name0.data (), len);
      name[len] = '\000';
      result.length = len;
      return result;
    }

  bool in_quotes = false;
  int i = 0;
  for (char c : name0)
    {
      if (in_quotes)
	name[i++] = c;
      else if (c_isalnum (c))
	name[i++] = c_tolower (c);
      else if (c_isspace (c))
	{
	  /* Nothing.  */
	}
      else if (c == '\'')
	{
	  /* Copy the starting quote, but not the ending quote.  */
	  if (!in_quotes)
	    name[i++] = c;
	  in_quotes = !in_quotes;
	}
      else
	name[i++] = c;
    }
  name[i] = '\000';

  result.length = i;
  return result;
}

/* See ada-exp-parser.h.  */

struct stoken
ada_parse_state::process_string (const char *text, const char *lim)
{
  const char *p;
  char *q;
  struct stoken result;

  q = (char *) obstack_alloc (&temp_space, lim - text);
  result.ptr = q;
  p = text;
  while (p < lim)
    {
      if (p[0] == '[' && p[1] == '"' && p+2 < lim)
	{
	  if (p[2] == '"')  /* "...["""]... */
	    {
	      *q = '"';
	      p += 4;
	    }
	  else
	    {
	      const char *end;
	      ULONGEST chr = strtoulst (p + 2, &end, 16);
	      if (chr > 0xff)
		error (_("wide strings are not yet supported"));
	      *q = (char) chr;
	      p = end + 1;
	    }
	}
      else
	*q = *p;
      q += 1;
      p += 1;
    }
  result.length = q - result.ptr;
  return result;
}

/* Returns non-zero iff string SUBSEQ matches a subsequence of STR, ignoring
   case.  */

static int
subseq_match (const char *subseq, const char *str)
{
  if (subseq[0] == '\0')
    return 1;
  else if (str[0] == '\0')
    return 0;
  else if (c_tolower (subseq[0]) == c_tolower (str[0]))
    return subseq_match (subseq+1, str+1) || subseq_match (subseq, str+1);
  else
    return subseq_match (subseq, str+1);
}

/* See ada-exp-parser.h.  */

int
ada_parse_state::process_attribute (const std::string &str, bool completing)
{
  size_t len = str.length ();
  if (completing)
    {
      ada_yylval.sval.ptr = obstack_strndup (&temp_space, str.c_str (), len);
      ada_yylval.sval.length = len;
      return TICK_COMPLETE;
    }

  for (const auto &item : attributes)
    if (strcasecmp (str.c_str (), item.name) == 0)
      return item.code;

  std::optional<int> found;
  for (const auto &item : attributes)
    if (subseq_match (str.c_str (), item.name))
      {
	if (!found.has_value ())
	  found = item.code;
	else
	  error (_("ambiguous attribute name: \"%s\""), str.c_str ());
      }
  if (!found.has_value ())
    error (_("unrecognized attribute: \"%s\""), str.c_str ());

  return *found;
}

int
ada_parse_state::lex_number ()
{
  const char *num_start = pstate->lexptr;
  gdb_assert (c_isdigit (*num_start));

  regmatch_t subexps[NUM_SUBEXPRESSIONS];
  int match = number_regex.exec (num_start, ARRAY_SIZE (subexps), subexps, 0);
  /* Failure means the regexp is broken.  */
  gdb_assert (match == 0);

  pstate->lexptr += subexps[0].rm_eo;

  char *numbuf = (char *) obstack_alloc (&temp_space,
					 pstate->lexptr - num_start);
  if (subexps[HEX_NUMBER].rm_so != -1)
    {
      canonicalize_numeral (numbuf, num_start + 2, pstate->lexptr);
      return process_int ("16#", numbuf, nullptr);
    }
  else if (subexps[FLOAT_POSEXP].rm_so != -1)
    {
      canonicalize_numeral (numbuf, num_start, pstate->lexptr);
      char *e_ptr = strrchr (numbuf, 'e');
      *e_ptr = '\0';
      return process_int (nullptr, numbuf, e_ptr + 1);
    }
  else if (subexps[FLOAT_OPTEXP].rm_so != -1)
    {
      canonicalize_numeral (numbuf, num_start, pstate->lexptr);
      return process_real (numbuf);
    }
  else if (subexps[FLOAT_EXTENSION].rm_so != -1)
    {
      canonicalize_numeral (numbuf, num_start, pstate->lexptr);
      char *e_ptr = nullptr;
      if (subexps[EXPONENT].rm_so != -1)
	{
	  /* We can't use the 'rm_so' here because
	     canonicalize_numeral might have removed some underscores.
	     But we know the 'e' appears after the '#' so we can
	     search for it and not get one of the (possibly hex)
	     numerals.  */
	  e_ptr = strrchr (numbuf, 'e');
	  gdb_assert (e_ptr != nullptr);
	  *e_ptr = '\0';
	  ++e_ptr;
	}
      return process_int (numbuf, strchr (numbuf, '#') + 1, e_ptr);
    }
  else if (subexps[BASED_REAL].rm_so != -1)
    error (_("Based real literals not implemented yet."));

  /* Must be a simple integer.  */
  canonicalize_numeral (numbuf, num_start, pstate->lexptr);
  return process_int (nullptr, numbuf, nullptr);
}

static const code_entry keywords[] =
{
  { "abs", ABS },
  { "and", _AND_ },
  { "delta", DELTA },
  { "else", ELSE },
  { "for", FOR },
  { "in", IN },
  { "mod", MOD },
  { "new", NEW },
  { "not", NOT },
  { "null", NULL_PTR },
  { "or", OR },
  { "others", OTHERS },
  { "rem", REM },
  { "then", THEN },
  { "with", WITH },
  { "xor", XOR },
  /* True and False are not keywords in Ada, but rather enumeration
     constants.  However, the boolean type is no longer represented as
     an enum, so True and False are no longer defined in symbol
     tables.  We compromise by making them keywords (when bare). */
  { "true", TRUEKEYWORD },
  { "false", FALSEKEYWORD },
  { "if", 0 },
  { "task", 0 },
  { "thread", -1 },
};

bool
ada_parse_state::check_thread_keyword (const char *text)
{
  /* "thread" has to be followed by whitespace and then a digit in
     order to be treated as a keyword.  */
  if (!c_isspace (*text))
    return false;
  const char *after = skip_spaces (text);
  return c_isdigit (*after);
}

std::optional<int>
ada_parse_state::check_keyword (size_t len)
{
  const char *start = pstate->lexptr;
  for (const code_entry &entry : keywords)
    {
      if (strlen (entry.name) == len
	  && strncasecmp (entry.name, start, len) == 0)
	{
	  int result = entry.code;
	  if (result == -1)
	    {
	      /* Special handling for the "thread" keyword.  */
	      if (!check_thread_keyword (pstate->lexptr + len))
		continue;
	      result = 0;
	    }
	  return result;
	}
    }

  return std::nullopt;
}

const char *
ada_parse_state::lex_operator (const char *start)
{
  gdb_assert (*start == '"');
  regmatch_t subexps[NUM_OPER_SUBEXPRESSIONS];
  int match = operator_regex.exec (start, ARRAY_SIZE (subexps), subexps, 0);
  if (match != 0)
    return nullptr;
  return start + subexps[0].rm_eo;
}

int
ada_parse_state::lex_identifier (const char *id_end)
{
  bool is_quoted = *pstate->lexptr == '<';
  bool completing = *id_end == '\0' && pstate->parse_completion;

  /* Only check keywords when the identifier is unquoted, and when not
     completing.  */
  if (!is_quoted && !completing)
    {
      std::optional<int> result = check_keyword (id_end - pstate->lexptr);
      if (result.has_value ())
	{
	  /* If the keyword results in an EOF, leave the parser's
	     lexptr pointing at the start of the keyword.  Otherwise,
	     update to the end.  */
	  if (*result != 0)
	    pstate->lexptr = id_end;
	  return *result;
	}
    }

  while (true)
    {
      const char *iter = skip_spaces (id_end);
      if (*iter != '.')
	break;
      iter = skip_spaces (iter + 1);

      const char *new_id_end = nullptr;
      if (*iter == '"')
	new_id_end = lex_operator (iter);
      else if (c_isalpha (*iter) || *iter == '_' || *iter == '<')
	new_id_end = lex_simple_identifier (iter);
      if (new_id_end == nullptr)
	break;

      /* Skip ".all" and leave it for a later DOT_ID lex.  */
      if (new_id_end - iter == 3 && strncasecmp (iter, "all", 3) == 0)
	break;

      id_end = new_id_end;
      completing = *id_end == '\0' && pstate->parse_completion;
    }

  ada_yylval.sval = process_id ({ pstate->lexptr,
				  (size_t) (id_end - pstate->lexptr) });

  id_end = skip_spaces (id_end);
  if (*id_end == '\'')
    m_before_qual_quote = true;

  pstate->lexptr = id_end;
  return completing ? NAME_COMPLETE : NAME;
}

void
ada_parse_state::invalid_char ()
{
  error (_("Invalid character '%c' in expression."), *pstate->lexptr);
}

const char *
ada_parse_state::lex_simple_identifier (const char *text)
{
  gdb_assert (c_isalpha (*text) || *text == '_' || *text == '<');

  const char *iter = text;
  bool quoted = (*iter == '<');
  if (quoted)
    {
      ++iter;
      if (!c_isalpha (*iter) && *iter != '_')
	return nullptr;
      ++iter;
    }

  auto as_unsigned = [] (const char *ptr)
    {
      return *(const unsigned char *) ptr;
    };

  while (c_isalnum (*iter) || *iter == '_'
	 || (!quoted && as_unsigned (iter) >= 0x80
	     && as_unsigned (iter) <= 0xff))
    ++iter;

  if (quoted)
    {
      /* A quirk of the flex lexer was that completion didn't work for
	 Ada-quoted symbols like <some_encoded_name>.  Perhaps TAB
	 somewhere before the '>' ought to work.  */
      if (*iter != '>')
	return nullptr;
      ++iter;
    }

  return iter;
}

int
ada_parse_state::lex_dot ()
{
  gdb_assert (*pstate->lexptr == '.');
  const char *save = pstate->lexptr;
  pstate->lexptr = skip_spaces (pstate->lexptr + 1);

  int name_token = 0;
  if (*pstate->lexptr == '<' || *pstate->lexptr == '_'
      || c_isalpha (*pstate->lexptr))
    {
      const char *id_end = lex_simple_identifier (pstate->lexptr);
      if (id_end != nullptr)
	{
	  pstate->lexptr = id_end;
	  name_token = NAME;
	}
    }

  if (*pstate->lexptr == '\0' && pstate->parse_completion)
    name_token = NAME_COMPLETE;

  if (name_token != NAME && name_token != NAME_COMPLETE)
    {
      pstate->lexptr = save;
      invalid_char ();
    }

  ada_yylval.sval = process_id ({ save + 1,
				(size_t) (pstate->lexptr - save - 1) });
  return name_token == NAME_COMPLETE ? DOT_COMPLETE : DOT_ID;
}

const char *
ada_parse_state::recognize_encoded_char (const char *bracket)
{
  if (bracket[0] == '[' && bracket[1] == '"')
    {
      const char *start = bracket + 2;
      const char *iter = start;

      /* Handle the weird case of ["""], which just means a double
	 quote.  */
      if (*iter == '"')
	++iter;
      else
	{
	  while (c_isxdigit (*iter))
	    ++iter;
	  if (iter - start < 2)
	    return nullptr;
	}

      if (iter[0] == '"' && iter[1] == ']')
	{
	  /* Ada-style quoted character.  */
	  return iter + 2;
	}
    }
  return nullptr;
}

int
ada_parse_state::lex_single_quote ()
{
  gdb_assert (*pstate->lexptr == '\'');

  bool before_qual_quote = m_before_qual_quote;
  m_before_qual_quote = false;

  if (!before_qual_quote)
    {
      /* Look for the GDB block extension "'STUFF'::".  */
      const char *block_iter = pstate->lexptr + 1;
      while (*block_iter != '\0' && *block_iter != '\'')
	++block_iter;
      /* Require a non-empty block.  */
      if (block_iter[0] == '\'' && block_iter - pstate->lexptr - 1 > 0)
	{
	  const char *block_end = skip_spaces (block_iter + 1);
	  if (block_end[0] == ':' && block_end[1] == ':')
	    {
	      ada_yylval.sval
		= process_id ({ pstate->lexptr,
				(size_t) (block_end - pstate->lexptr) });
	      /* Leave the "::" to be lexed later.  */
	      pstate->lexptr = block_end;
	      return NAME;
	    }
	}

      if (c_isprint (pstate->lexptr[1]) && pstate->lexptr[2] == '\'')
	{
	  /* Ordinary character literal.  */
	  int the_char = pstate->lexptr[1];
	  ada_yylval.typed_char.val = the_char;
	  ada_yylval.typed_char.type = type_for_char (pstate, the_char);
	  pstate->lexptr += 3;
	  return CHARLIT;
	}

      const char *end_bracket = recognize_encoded_char (pstate->lexptr + 1);
      if (end_bracket != nullptr)
	{
	  if (*end_bracket != '\'')
	    {
	      /* We saw something pretty odd.  */
	      error (_("Unterminated character literal"));
	    }
	  /* The character starts at an offset from the initial single
	     quote.  */
	  ULONGEST v = strtoulst (pstate->lexptr + 3, nullptr, 16);
	  ada_yylval.typed_char.val = v;
	  ada_yylval.typed_char.type = type_for_char (pstate, v);
	  pstate->lexptr = end_bracket + 1;
	  return CHARLIT;
	}
    }

  /* Handle 'ATTRIBUTE; with a special case when completing just after
     a single quote.  */
  const char *attr_start = skip_spaces (pstate->lexptr + 1);
  if (c_isalpha (*attr_start)
      || (pstate->parse_completion && *attr_start == '\0'))
    {
      const char *iter = attr_start;
      while (*iter == '_' || c_isalpha (*iter))
	++iter;
      bool completing = pstate->parse_completion && *iter == '\0';
      pstate->lexptr = iter;
      return process_attribute (std::string (attr_start, iter - attr_start),
				completing);
    }

  ++pstate->lexptr;
  return '\'';
}

int
ada_parse_state::lex_double_quote ()
{
  gdb_assert (*pstate->lexptr == '"');
  const char *start = pstate->lexptr + 1;

  const char *iter = start;
  while (*iter != '"' && *iter != '\0')
    {
      /* Note that we don't handle bracket sequences of more than 2
	 digits here.  Currently there's no support for wide or
	 wide-wide strings.  */
      const char *end_bracket = recognize_encoded_char (iter);
      if (end_bracket != nullptr)
	iter = end_bracket;
      else if (c_isprint (*iter))
	++iter;
      else
	break;
    }

  if (*iter != '"')
    error (_("ill-formed or non-terminated string literal"));

  ada_yylval.sval = process_string (start, iter);
  pstate->lexptr = iter + 1;
  return STRING;
}

int
ada_parse_state::lex_dollar ()
{
  gdb_assert (*pstate->lexptr == '$');
  const char *start = pstate->lexptr++;
  while (c_isalnum (*pstate->lexptr) || *pstate->lexptr == '_'
	 || *pstate->lexptr == '$')
    ++pstate->lexptr;
  ada_yylval.sval.ptr = start;
  ada_yylval.sval.length = pstate->lexptr - start;
  return DOLLAR_VARIABLE;
}

static const code_entry punct[] =
{
  { "=>", ARROW },
  { "..", DOTDOT },
  { "**", STARSTAR },
  { ":=", ASSIGN },
  { "/=", NOTEQUAL },
  { "<=", LEQ },
  { ">=", GEQ },
  { "::", COLONCOLON },
  /* "--" is documented in the manual as being a comment.  */
  { "--", 0 },
};

int
ada_parse_state::lex ()
{
  /* If m_before_qual_quote is set, we must be examining a single
     quote.  */
  gdb_assert (!m_before_qual_quote || *pstate->lexptr == '\'');

  pstate->lexptr = skip_spaces (pstate->lexptr);

  if (pstate->lexptr[0] != '\0' && pstate->lexptr[1] != '\0')
    for (const code_entry &entry : punct)
      {
	size_t len = strlen (entry.name);
	gdb_assert (len == 2);
	if (strncmp (entry.name, pstate->lexptr, len) == 0)
	  {
	    /* Handle the "--" comment case specially.  */
	    if (entry.code == 0)
	      {
		size_t rest = strlen (pstate->lexptr);
		pstate->lexptr += rest;
	      }
	    else
	      pstate->lexptr += len;
	    return entry.code;
	  }
      }

  switch (*pstate->lexptr)
    {
    case '\0':
      return 0;

    case '-':
    case '=':
    case '*':
    case ':':
    case '&':
    case '+':
    case '{':
    case '}':
    case '@':
    case '/':
    case '>':
    case '|':
    case ';':
    case '[':
    case ']':
      return *pstate->lexptr++;

    case ',':
      if (m_paren_depth == 0 && pstate->comma_terminates)
	return 0;
      else
	return *pstate->lexptr++;

    case '(':
      ++m_paren_depth;
      return *pstate->lexptr++;

    case ')':
      if (m_paren_depth == 0)
	return 0;
      --m_paren_depth;
      return *pstate->lexptr++;

    case '\'':
      return lex_single_quote ();

    case '"':
      return lex_double_quote ();

    case '$':
      return lex_dollar ();

    case '.':
      return lex_dot ();
    }

  if (c_isalpha (*pstate->lexptr) || *pstate->lexptr == '_'
      || *pstate->lexptr == '<')
    {
      const char *id_end = lex_simple_identifier (pstate->lexptr);
      if (id_end == nullptr)
	{
	  /* It shouldn't be possible to reach here when starting with
	     a letter or underscore.  */
	  gdb_assert (*pstate->lexptr == '<');
	  return *pstate->lexptr++;
	}
      return lex_identifier (id_end);
    }

  if (c_isdigit (*pstate->lexptr))
    return lex_number ();

  invalid_char ();
}

#if GDB_SELF_TEST

namespace selftests
{

/* Helper that checks that a given lex causes an exception.  */
static void
ada_lex_test_error (const char *input)
{
  /* We need a dummy parser.  */
  parser_state ps (language_def (language_ada), current_inferior ()->arch (),
		   nullptr, 0, 0, input, false, nullptr);
  scoped_restore save_parser = make_scoped_restore (&pstate, &ps);

  ada_parse_state ada_ps (input);
  /* This is probably not needed, but also it doesn't hurt.  */
  scoped_restore save_aps = make_scoped_restore (&ada_parser, &ada_ps);

  bool saw_failure = false;
  try
    {
      ada_ps.lex ();
    }
  catch (const gdb_exception_error &err)
    {
      saw_failure = true;
    }
  SELF_CHECK (saw_failure);
}

/* Helper to check a lexing sequence, also checking string values of
   tokens.  */
static void
ada_lex_test_sequence (const char *input,
		       gdb::span<const code_entry> expected,
		       bool completion = false)
{
  /* We need a dummy parser.  */
  parser_state ps (language_def (language_ada), current_inferior ()->arch (),
		   nullptr, 0, 0, input, completion, nullptr);
  scoped_restore save_parser = make_scoped_restore (&pstate, &ps);

  ada_parse_state ada_ps (input);
  /* This is probably not needed, but also it doesn't hurt.  */
  scoped_restore save_aps = make_scoped_restore (&ada_parser, &ada_ps);
  for (const code_entry &entry : expected)
    {
      int token = ada_ps.lex ();
      SELF_CHECK (token == entry.code);

      bool has_string_value = (token == NAME || token == NAME_COMPLETE
			       || token == DOT_ID || token == DOT_COMPLETE
			       || token == STRING
			       || token == DOLLAR_VARIABLE
			       || token == TICK_COMPLETE);
      if (has_string_value)
	{
	  std::string_view contents (ada_yylval.sval.ptr,
				     ada_yylval.sval.length);
	  SELF_CHECK (contents == entry.name);
	}
      else
	SELF_CHECK (entry.name == nullptr);
    }

  /* Test that all the input was used.  */
  SELF_CHECK (*ps.lexptr == '\0');
}

/* Helper that checks an integer value.  */
static void
ada_lex_test_integer (const char *input, int value)
{
  /* We need a dummy parser.  */
  parser_state ps (language_def (language_ada), current_inferior ()->arch (),
		   nullptr, 0, 0, input, false, nullptr);
  scoped_restore save_parser = make_scoped_restore (&pstate, &ps);

  ada_parse_state ada_ps (input);
  /* This is probably not needed, but also it doesn't hurt.  */
  scoped_restore save_aps = make_scoped_restore (&ada_parser, &ada_ps);

  int token = ada_ps.lex ();
  SELF_CHECK (token == INT);
  SELF_CHECK (*ada_yylval.typed_val.val == value);
}

static void
ada_lex_tests ()
{
  const code_entry cc_test[] =
    {
      { "blockname", NAME },
      { nullptr, COLONCOLON },
      { "duck", NAME },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("blockname::duck", cc_test);

  const code_entry ccq_test[] =
    {
      { "'blockname'", NAME },
      { nullptr, COLONCOLON },
      { "duck", NAME },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("'blockname'::duck", ccq_test);
  ada_lex_test_sequence ("'blockname'  ::  duck", ccq_test);

  /* This should not lex as a character literal.  */
  const code_entry ccq2_test[] =
    {
      { "'a'", NAME },
      { nullptr, COLONCOLON },
      { "duck", NAME },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("'a'::duck", ccq2_test);

  const code_entry comment_test[] =
    {
      { "variable", NAME },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("variable -- comment text here", comment_test);

  const code_entry oper_test[] =
    {
      { "name.\"mod\"", NAME },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("   name . \"MoD\"   ", oper_test);

  const code_entry quote_test[] =
    {
      { "<some__Weird_Name>.whatever", NAME },
      { "all", DOT_ID },
      { "rest", DOT_ID },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("<some__Weird_Name>.whatever.all.rest", quote_test);

  const code_entry uquote_test[] =
    {
      { "<__leading_uscore>", NAME },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("<__leading_uscore>", uquote_test);

  const code_entry qual_test[] =
    {
      { "char_enum", NAME },
      { nullptr, '\'' },
      { nullptr, '(' },
      { nullptr, CHARLIT },
      { nullptr, ')' },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("  Char_Enum ' ( 'B' )  ", qual_test);

  /* This failed historically since the TAB character is not
     "printable".  */
  ada_lex_test_error ("\"\t\"");

  /* This should fail due to unbalanced quotes.  */
  ada_lex_test_error ("'[\"61\"]x");

  const code_entry char_test[] =
    {
      { nullptr, CHARLIT },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("'[\"61\"]'", char_test);
  ada_lex_test_sequence ("'\"'", char_test);
  ada_lex_test_sequence ("' '", char_test);

  const code_entry str_test[] =
    {
      { "a string \"", STRING },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("\"[\"61\"] string [\"\"\"]\"", str_test);

  const code_entry compl_test[] =
    {
      { "duck.duck", NAME },
      { "goose", TICK_COMPLETE },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("duck.duck'goose", compl_test, true);

  const code_entry compl2_test[] =
    {
      { "pck.inner", NAME },
      { "", DOT_COMPLETE },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("pck.inner.", compl2_test, true);

  const code_entry compl3_test[] =
    {
      { nullptr, '(' },
      { "name", NAME },
      { nullptr, ')' },
      { "", DOT_COMPLETE },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("(name).", compl3_test, true);

  const code_entry compl4_test[] =
    {
      { nullptr, '(' },
      { "name", NAME },
      { nullptr, ')' },
      { "fl", DOT_COMPLETE },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("(name).fl", compl4_test, true);

  const code_entry tick_test[] =
    {
      { "duck.duck", NAME },
      { nullptr, TICK_MIN },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("duck.duck  '  min", tick_test);

  const code_entry us3_test[] =
    {
      { "x", NAME },
      { nullptr, '+' },
      { "a___b", NAME },
      { nullptr, 0 }
    };
  ada_lex_test_sequence ("X + a___b", us3_test);

  ada_lex_test_integer ("0xff", 255);
  ada_lex_test_integer ("0xffff", 65535);
  ada_lex_test_integer ("019", 19);
  ada_lex_test_integer ("0", 0);
  ada_lex_test_integer ("16#ff#", 255);
  ada_lex_test_integer ("16#ee#", 238);
  ada_lex_test_integer ("16#EE#", 238);
  ada_lex_test_integer ("8#77#", 63);

  /* Based reals aren't handled.  */
  ada_lex_test_error ("16#abcd.ef01#");
  ada_lex_test_error ("16#abcd.ef01#e+23");
}

} /* namespace selftests */

#endif /* GDB_SELF_TEST */

/* See ada-exp-parser.h.  */

operation_up
resolve (operation_up &&op, bool deprocedure_p, struct type *context_type)
{
  operation_up result = std::move (op);
  ada_resolvable *res = dynamic_cast<ada_resolvable *> (result.get ());
  if (res != nullptr)
    return res->replace (std::move (result),
			 pstate->expout.get (),
			 deprocedure_p,
			 pstate->parse_completion,
			 pstate->block_tracker,
			 context_type);
  return result;
}

/* See ada-exp-parser.h.  */

operation_up
ada_pop (bool deprocedure_p, struct type *context_type)
{
  /* Of course it's ok to call parser_state::pop here... */
  return resolve (pstate->pop (), deprocedure_p, context_type);
}

/* See ada-exp-parser.h.  */

void
ada_addrof (struct type *type)
{
  operation_up arg = ada_pop (false);
  operation_up addr = make_operation<unop_addr_operation> (std::move (arg));
  operation_up wrapped
    = make_operation<ada_wrapped_operation> (std::move (addr));
  if (type != nullptr)
    wrapped = make_operation<unop_cast_operation> (std::move (wrapped), type);
  pstate->push (std::move (wrapped));
}

/* See ada-exp-parser.h.  */

operation_up
maybe_overload (enum exp_opcode op, operation_up &lhs, operation_up &rhs)
{
  struct value *args[2];

  int nargs = 1;
  args[0] = lhs->evaluate (nullptr, pstate->expout.get (),
			   EVAL_AVOID_SIDE_EFFECTS);
  if (rhs == nullptr)
    args[1] = nullptr;
  else
    {
      args[1] = rhs->evaluate (nullptr, pstate->expout.get (),
			       EVAL_AVOID_SIDE_EFFECTS);
      ++nargs;
    }

  block_symbol fn = ada_find_operator_symbol (op, pstate->parse_completion,
					      nargs, args);
  if (fn.symbol == nullptr)
    return {};

  if (symbol_read_needs_frame (fn.symbol))
    pstate->block_tracker->update (fn.block, INNERMOST_BLOCK_FOR_SYMBOLS);
  operation_up callee = make_operation<ada_var_value_operation> (fn);

  std::vector<operation_up> argvec;
  argvec.push_back (std::move (lhs));
  if (rhs != nullptr)
    argvec.push_back (std::move (rhs));
  return make_operation<ada_funcall_operation> (std::move (callee),
						std::move (argvec));
}

/* See ada-exp-parser.h.  */

void
ada_funcall (int nargs)
{
  /* We use the ordinary pop here, because we're going to do
     resolution in a separate step, in order to handle array
     indices.  */
  std::vector<operation_up> args = pstate->pop_vector (nargs);
  /* Call parser_state::pop here, because we don't want to
     function-convert the callee slot of a call we're already
     constructing.  */
  operation_up callee = pstate->pop ();

  ada_var_value_operation *vvo
    = dynamic_cast<ada_var_value_operation *> (callee.get ());
  int array_arity = 0;
  struct type *callee_t = nullptr;
  if (vvo == nullptr
      || vvo->get_symbol ()->domain () != UNDEF_DOMAIN)
    {
      struct value *callee_v = callee->evaluate (nullptr,
						 pstate->expout.get (),
						 EVAL_AVOID_SIDE_EFFECTS);
      callee_t = ada_check_typedef (callee_v->type ());
      array_arity = ada_array_arity (callee_t);
    }

  for (int i = 0; i < nargs; ++i)
    {
      struct type *subtype = nullptr;
      if (i < array_arity)
	subtype = ada_index_type (callee_t, i + 1, "array type");
      args[i] = resolve (std::move (args[i]), true, subtype);
    }

  std::unique_ptr<ada_funcall_operation> funcall
    (new ada_funcall_operation (std::move (callee), std::move (args)));
  funcall->resolve (pstate->expout.get (), true, pstate->parse_completion,
		    pstate->block_tracker, nullptr);
  pstate->push (std::move (funcall));
}

/* See ada-exp-parser.h.  */

ada_choices_component *
choice_component ()
{
  ada_component *last = ada_parser->components.back ().get ();
  return gdb::checked_static_cast<ada_choices_component *> (last);
}

/* See ada-exp-parser.h.  */

ada_component_up
pop_component ()
{
  ada_component_up result = std::move (ada_parser->components.back ());
  ada_parser->components.pop_back ();
  return result;
}

/* See ada-exp-parser.h.  */

std::vector<ada_component_up>
pop_components (int n)
{
  std::vector<ada_component_up> result (n);
  for (int i = 1; i <= n; ++i)
    result[n - i] = pop_component ();
  return result;
}

/* Pop the most recent association from the global stack, and return
   it.  */
static ada_association_up
pop_association ()
{
  ada_association_up result = std::move (ada_parser->associations.back ());
  ada_parser->associations.pop_back ();
  return result;
}

/* See ada-exp-parser.h.  */

std::vector<ada_association_up>
pop_associations (int n)
{
  std::vector<ada_association_up> result (n);
  for (int i = 1; i <= n; ++i)
    result[n - i] = pop_association ();
  return result;
}

/* See ada-exp-parser.h.  */

std::unique_ptr<expr_completion_base>
make_tick_completer (struct stoken tok)
{
  return (std::unique_ptr<expr_completion_base>
	  (new ada_tick_completer (std::string (tok.ptr, tok.length))));
}

/* Emit expression to access an instance of SYM, in block BLOCK (if
   non-NULL).  */

static void
write_var_from_sym (struct parser_state *par_state, block_symbol sym)
{
  if (symbol_read_needs_frame (sym.symbol))
    par_state->block_tracker->update (sym.block, INNERMOST_BLOCK_FOR_SYMBOLS);

  par_state->push_new<ada_var_value_operation> (sym);
}

/* See ada-exp-parser.h.  */

void
write_int (struct parser_state *par_state, LONGEST arg, struct type *type)
{
  pstate->push_new<long_const_operation> (type, arg);
  ada_wrap<ada_wrapped_operation> ();
}

/* Emit expression corresponding to the renamed object named
   designated by RENAMED_ENTITY[0 .. RENAMED_ENTITY_LEN-1] in the
   context of ORIG_LEFT_CONTEXT, to which is applied the operations
   encoded by RENAMING_EXPR.  MAX_DEPTH is the maximum number of
   cascaded renamings to allow.  If ORIG_LEFT_CONTEXT is null, it
   defaults to the currently selected block. ORIG_SYMBOL is the
   symbol that originally encoded the renaming.  It is needed only
   because its prefix also qualifies any index variables used to index
   or slice an array.  It should not be necessary once we go to the
   new encoding entirely (FIXME pnh 7/20/2007).  */

static void
write_object_renaming (struct parser_state *par_state,
		       const struct block *orig_left_context,
		       const char *renamed_entity, int renamed_entity_len,
		       const char *renaming_expr, int max_depth)
{
  char *name;
  enum { SIMPLE_INDEX, LOWER_BOUND, UPPER_BOUND } slice_state;

  if (max_depth <= 0)
    error (_("Could not find renamed symbol"));

  if (orig_left_context == NULL)
    orig_left_context = get_selected_block ();

  name = obstack_strndup (&ada_parser->temp_space, renamed_entity,
			  renamed_entity_len);
  block_symbol sym_info = ada_lookup_encoded_symbol (name, orig_left_context,
						     SEARCH_VFT);
  if (sym_info.symbol == NULL)
    error (_("Could not find renamed variable: %ps"),
	   styled_string (variable_name_style.style (),
			  ada_decode (name).c_str ()));
  else if (sym_info.symbol->loc_class () == LOC_TYPEDEF)
    /* We have a renaming of an old-style renaming symbol.  Don't
       trust the block information.  */
    sym_info.block = orig_left_context;

  {
    const char *inner_renamed_entity;
    int inner_renamed_entity_len;
    const char *inner_renaming_expr;

    switch (ada_parse_renaming (sym_info.symbol, &inner_renamed_entity,
				&inner_renamed_entity_len,
				&inner_renaming_expr))
      {
      case ADA_NOT_RENAMING:
	write_var_from_sym (par_state, sym_info);
	break;
      case ADA_OBJECT_RENAMING:
	write_object_renaming (par_state, sym_info.block,
			       inner_renamed_entity, inner_renamed_entity_len,
			       inner_renaming_expr, max_depth - 1);
	break;
      default:
	goto BadEncoding;
      }
  }

  slice_state = SIMPLE_INDEX;
  while (*renaming_expr == 'X')
    {
      renaming_expr += 1;

      switch (*renaming_expr) {
      case 'A':
	renaming_expr += 1;
	ada_wrap<ada_unop_ind_operation> ();
	break;
      case 'L':
	slice_state = LOWER_BOUND;
	[[fallthrough]];
      case 'S':
	renaming_expr += 1;
	if (c_isdigit (*renaming_expr))
	  {
	    char *next;
	    long val = strtol (renaming_expr, &next, 10);
	    if (next == renaming_expr)
	      goto BadEncoding;
	    renaming_expr = next;
	    write_int (par_state, val, parse_type (par_state)->builtin_int);
	  }
	else
	  {
	    const char *end;
	    char *index_name;

	    end = strchr (renaming_expr, 'X');
	    if (end == NULL)
	      end = renaming_expr + strlen (renaming_expr);

	    index_name = obstack_strndup (&ada_parser->temp_space,
					  renaming_expr,
					  end - renaming_expr);
	    renaming_expr = end;

	    block_symbol index_sym_info
	      = ada_lookup_encoded_symbol (index_name, orig_left_context,
					   SEARCH_VFT);
	    if (index_sym_info.symbol == NULL)
	      error (_("Could not find %s"), index_name);
	    else if (index_sym_info.symbol->loc_class () == LOC_TYPEDEF)
	      /* Index is an old-style renaming symbol.  */
	      index_sym_info.block = orig_left_context;
	    write_var_from_sym (par_state, index_sym_info);
	  }
	if (slice_state == SIMPLE_INDEX)
	  ada_funcall (1);
	else if (slice_state == LOWER_BOUND)
	  slice_state = UPPER_BOUND;
	else if (slice_state == UPPER_BOUND)
	  {
	    ada_wrap3<ada_ternop_slice_operation> ();
	    slice_state = SIMPLE_INDEX;
	  }
	break;

      case 'R':
	{
	  const char *end;

	  renaming_expr += 1;

	  if (slice_state != SIMPLE_INDEX)
	    goto BadEncoding;
	  end = strchr (renaming_expr, 'X');
	  if (end == NULL)
	    end = renaming_expr + strlen (renaming_expr);

	  operation_up arg = ada_pop ();
	  pstate->push_new<ada_structop_operation>
	    (std::move (arg), std::string (renaming_expr,
					   end - renaming_expr));
	  renaming_expr = end;
	  break;
	}

      default:
	goto BadEncoding;
      }
    }
  if (slice_state == SIMPLE_INDEX)
    return;

 BadEncoding:
  error (_("Internal error in encoding of renaming declaration"));
}

/* See ada-exp-parser.h.  */

const struct block*
block_lookup (const struct block *context, const char *raw_name)
{
  const char *name;
  struct symtab *symtab;
  const struct block *result = NULL;

  std::string name_storage;
  if (raw_name[0] == '\'')
    {
      raw_name += 1;
      name = raw_name;
    }
  else
    {
      name_storage = ada_encode (raw_name);
      name = name_storage.c_str ();
    }

  std::vector<struct block_symbol> syms
    = ada_lookup_symbol_list (name, context, SEARCH_FUNCTION_DOMAIN);

  if (context == NULL
      && (syms.empty () || syms[0].symbol->loc_class () != LOC_BLOCK))
    symtab = lookup_symtab (current_program_space, name);
  else
    symtab = NULL;

  if (symtab != NULL)
    result = symtab->compunit ().blockvector ()->static_block ();
  else if (syms.empty () || syms[0].symbol->loc_class () != LOC_BLOCK)
    {
      if (context == NULL)
	error (_("No file or function \"%s\"."), raw_name);
      else
	error (_("No function \"%s\" in specified context."), raw_name);
    }
  else
    {
      if (syms.size () > 1)
	warning (_("Function name \"%s\" ambiguous here"), raw_name);
      result = syms[0].symbol->value_block ();
    }

  return result;
}

static struct symbol*
select_possible_type_sym (const std::vector<struct block_symbol> &syms)
{
  int i;
  int preferred_index;
  struct type *preferred_type;

  preferred_index = -1; preferred_type = NULL;
  for (i = 0; i < syms.size (); i += 1)
    switch (syms[i].symbol->loc_class ())
      {
      case LOC_TYPEDEF:
	if (ada_prefer_type (syms[i].symbol->type (), preferred_type))
	  {
	    preferred_index = i;
	    preferred_type = syms[i].symbol->type ();
	  }
	break;
      case LOC_REGISTER:
      case LOC_ARG:
      case LOC_REF_ARG:
      case LOC_REGPARM_ADDR:
      case LOC_LOCAL:
      case LOC_COMPUTED:
	return NULL;
      default:
	break;
      }
  if (preferred_type == NULL)
    return NULL;
  return syms[preferred_index].symbol;
}

static struct type*
find_primitive_type (struct parser_state *par_state, const char *name)
{
  struct type *type;
  type = language_lookup_primitive_type (par_state->language (),
					 par_state->gdbarch (),
					 name);
  if (type == NULL && streq ("system__address", name))
    type = type_system_address (par_state);

  if (type != NULL)
    {
      /* Check to see if we have a regular definition of this
	 type that just didn't happen to have been read yet.  */
      struct symbol *sym;
      char *expanded_name =
	(char *) alloca (strlen (name) + sizeof ("standard__"));
      strcpy (expanded_name, "standard__");
      strcat (expanded_name, name);
      sym = ada_lookup_symbol (expanded_name, NULL, SEARCH_TYPE_DOMAIN).symbol;
      if (sym != NULL && sym->loc_class () == LOC_TYPEDEF)
	type = sym->type ();
    }

  return type;
}

static int
chop_selector (const char *name, int end)
{
  int i;
  for (i = end - 1; i > 0; i -= 1)
    if (name[i] == '.' || (name[i] == '_' && name[i+1] == '_'))
      return i;
  return -1;
}

/* If NAME is a string beginning with a separator (either '__', or
   '.'), chop this separator and return the result; else, return
   NAME.  */

static const char *
chop_separator (const char *name)
{
  if (*name == '.')
   return name + 1;

  if (name[0] == '_' && name[1] == '_')
    return name + 2;

  return name;
}

/* Given that SELS is a string of the form (<sep><identifier>)*, where
   <sep> is '__' or '.', write the indicated sequence of
   STRUCTOP_STRUCT expression operators.  Returns a pointer to the
   last operation that was pushed.  */
static ada_structop_operation *
write_selectors (struct parser_state *par_state, const char *sels)
{
  ada_structop_operation *result = nullptr;
  while (*sels != '\0')
    {
      const char *p = chop_separator (sels);
      sels = p;
      while (*sels != '\0' && *sels != '.'
	     && (sels[0] != '_' || sels[1] != '_'))
	sels += 1;
      operation_up arg = ada_pop ();
      result = new ada_structop_operation (std::move (arg),
					   std::string (p, sels - p));
      pstate->push (operation_up (result));
    }
  return result;
}

/* Write a variable access (OP_VAR_VALUE) to ambiguous encoded name
   NAME[0..LEN-1], in block context BLOCK, to be resolved later.  Writes
   a temporary symbol that is valid until the next call to ada_parse.
   */
static void
write_ambiguous_var (struct parser_state *par_state,
		     const struct block *block, const char *name, int len)
{
  struct symbol *sym = new (&ada_parser->temp_space) symbol ();

  sym->set_domain (UNDEF_DOMAIN);
  sym->set_linkage_name (obstack_strndup (&ada_parser->temp_space, name, len));
  sym->set_language (language_ada, nullptr);

  block_symbol bsym { sym, block };
  par_state->push_new<ada_var_value_operation> (bsym);
}

/* A convenient wrapper around ada_get_field_index that takes
   a non NUL-terminated FIELD_NAME0 and a FIELD_NAME_LEN instead
   of a NUL-terminated field name.  */

static int
ada_nget_field_index (const struct type *type, const char *field_name0,
		      int field_name_len, int maybe_missing)
{
  char *field_name = (char *) alloca ((field_name_len + 1) * sizeof (char));

  strncpy (field_name, field_name0, field_name_len);
  field_name[field_name_len] = '\0';
  return ada_get_field_index (type, field_name, maybe_missing);
}

/* If encoded_field_name is the name of a field inside symbol SYM,
   then return the type of that field.  Otherwise, return NULL.

   This function is actually recursive, so if ENCODED_FIELD_NAME
   doesn't match one of the fields of our symbol, then try to see
   if ENCODED_FIELD_NAME could not be a succession of field names
   (in other words, the user entered an expression of the form
   TYPE_NAME.FIELD1.FIELD2.FIELD3), in which case we evaluate
   each field name sequentially to obtain the desired field type.
   In case of failure, we return NULL.  */

static struct type *
get_symbol_field_type (struct symbol *sym, const char *encoded_field_name)
{
  const char *field_name = encoded_field_name;
  const char *subfield_name;
  struct type *type = sym->type ();
  int fieldno;

  if (type == NULL || field_name == NULL)
    return NULL;
  type = check_typedef (type);

  while (field_name[0] != '\0')
    {
      field_name = chop_separator (field_name);

      fieldno = ada_get_field_index (type, field_name, 1);
      if (fieldno >= 0)
	return type->field (fieldno).type ();

      subfield_name = field_name;
      while (*subfield_name != '\0' && *subfield_name != '.'
	     && (subfield_name[0] != '_' || subfield_name[1] != '_'))
	subfield_name += 1;

      if (subfield_name[0] == '\0')
	return NULL;

      fieldno = ada_nget_field_index (type, field_name,
				      subfield_name - field_name, 1);
      if (fieldno < 0)
	return NULL;

      type = type->field (fieldno).type ();
      field_name = subfield_name;
    }

  return NULL;
}

/* See ada-exp-parser.h.  */

struct type*
write_var_or_type (struct parser_state *par_state,
		   const struct block *block, struct stoken name0)
{
  int depth;
  char *encoded_name;
  int name_len;

  std::string name_storage = ada_encode (name0.ptr);

  if (block == nullptr)
    {
      auto iter = ada_parser->iterated_associations.find (name_storage);
      if (iter != ada_parser->iterated_associations.end ())
	{
	  auto op = std::make_unique<ada_index_var_operation> ();
	  iter->second.push_back (op.get ());
	  par_state->push (std::move (op));
	  return nullptr;
	}

      block = par_state->expression_context_block;
    }

  name_len = name_storage.size ();
  encoded_name = obstack_strndup (&ada_parser->temp_space,
				  name_storage.c_str (),
				  name_len);
  for (depth = 0; depth < MAX_RENAMING_CHAIN_LENGTH; depth += 1)
    {
      int tail_index;

      tail_index = name_len;
      while (tail_index > 0)
	{
	  struct symbol *type_sym;
	  struct symbol *renaming_sym;
	  const char* renaming;
	  int renaming_len;
	  const char* renaming_expr;
	  int terminator = encoded_name[tail_index];

	  encoded_name[tail_index] = '\0';
	  /* In order to avoid double-encoding, we want to only pass
	     the decoded form to lookup functions.  */
	  std::string decoded_name = ada_decode (encoded_name);
	  encoded_name[tail_index] = terminator;

	  std::vector<struct block_symbol> syms
	    = ada_lookup_symbol_list (decoded_name.c_str (), block,
				      SEARCH_VFT);

	  type_sym = select_possible_type_sym (syms);

	  if (type_sym != NULL)
	    renaming_sym = type_sym;
	  else if (syms.size () == 1)
	    renaming_sym = syms[0].symbol;
	  else
	    renaming_sym = NULL;

	  switch (ada_parse_renaming (renaming_sym, &renaming,
				      &renaming_len, &renaming_expr))
	    {
	    case ADA_NOT_RENAMING:
	      break;
	    case ADA_PACKAGE_RENAMING:
	    case ADA_EXCEPTION_RENAMING:
	    case ADA_SUBPROGRAM_RENAMING:
	      {
		int alloc_len = renaming_len + name_len - tail_index + 1;
		char *new_name
		  = (char *) obstack_alloc (&ada_parser->temp_space,
					    alloc_len);
		strncpy (new_name, renaming, renaming_len);
		strcpy (new_name + renaming_len, encoded_name + tail_index);
		encoded_name = new_name;
		name_len = renaming_len + name_len - tail_index;
		goto TryAfterRenaming;
	      }
	    case ADA_OBJECT_RENAMING:
	      write_object_renaming (par_state, block, renaming, renaming_len,
				     renaming_expr, MAX_RENAMING_CHAIN_LENGTH);
	      write_selectors (par_state, encoded_name + tail_index);
	      return NULL;
	    default:
	      internal_error (_("impossible value from ada_parse_renaming"));
	    }

	  if (type_sym != NULL)
	    {
	      struct type *field_type;

	      if (tail_index == name_len)
		return type_sym->type ();

	      /* We have some extraneous characters after the type name.
		 If this is an expression "TYPE_NAME.FIELD0.[...].FIELDN",
		 then try to get the type of FIELDN.  */
	      field_type
		= get_symbol_field_type (type_sym, encoded_name + tail_index);
	      if (field_type != NULL)
		return field_type;
	      else
		error (_("Invalid attempt to select from type: \"%s\"."),
		       name0.ptr);
	    }
	  else if (tail_index == name_len && syms.empty ())
	    {
	      struct type *type = find_primitive_type (par_state,
						       encoded_name);

	      if (type != NULL)
		return type;
	    }

	  if (syms.size () == 1)
	    {
	      write_var_from_sym (par_state, syms[0]);
	      write_selectors (par_state, encoded_name + tail_index);
	      return NULL;
	    }
	  else if (syms.empty ())
	    {
	      struct objfile *objfile = nullptr;
	      if (block != nullptr)
		objfile = block->objfile ();

	      bound_minimal_symbol msym
		= ada_lookup_simple_minsym (decoded_name.c_str (), objfile);
	      if (msym.minsym != NULL)
		{
		  par_state->push_new<ada_var_msym_value_operation> (msym);
		  /* Maybe cause error here rather than later? FIXME? */
		  write_selectors (par_state, encoded_name + tail_index);
		  return NULL;
		}

	      if (tail_index == name_len
		  && strncmp (encoded_name, "standard__",
			      sizeof ("standard__") - 1) == 0)
		error (_("No definition of \"%s\" found."), name0.ptr);

	      tail_index = chop_selector (encoded_name, tail_index);
	    }
	  else
	    {
	      write_ambiguous_var (par_state, block, encoded_name,
				   tail_index);
	      write_selectors (par_state, encoded_name + tail_index);
	      return NULL;
	    }
	}

      if (!current_program_space->has_full_symbols ()
	  && !current_program_space->has_partial_symbols ()
	  && block == NULL)
	error (_("No symbol table is loaded.  Use the \"%ps\" command."),
	       styled_string (command_style.style (), "file"));
      if (block == par_state->expression_context_block)
	error (_("No definition of \"%s\" in current context."), name0.ptr);
      else
	error (_("No definition of \"%s\" in specified context."), name0.ptr);

    TryAfterRenaming: ;
    }

  error (_("Could not find renamed symbol \"%s\""), name0.ptr);

}

/* Because ada_completer_word_break_characters does not contain '.' --
   and it cannot easily be added, this breaks other completions -- we
   have to recreate the completion word-splitting here, so that we can
   provide a prefix that is then used when completing field names.
   Without this, an attempt like "complete print abc.d" will give a
   result like "print def" rather than "print abc.def".  */

std::string
ada_parse_state::find_completion_bounds ()
{
  const char *end = pstate->lexptr;
  /* First the end of the prefix.  Here we stop at the token start or
     at '.' or space.  */
  for (; end > m_original_expr && end[-1] != '.' && !c_isspace (end[-1]); --end)
    {
      /* Nothing.  */
    }
  /* Now find the start of the prefix.  */
  const char *ptr = end;
  /* Here we allow '.'.  */
  for (;
       ptr > m_original_expr && (ptr[-1] == '.'
				 || ptr[-1] == '_'
				 || (ptr[-1] >= 'a' && ptr[-1] <= 'z')
				 || (ptr[-1] >= 'A' && ptr[-1] <= 'Z')
				 || (ptr[-1] & 0xff) >= 0x80);
       --ptr)
    {
      /* Nothing.  */
    }
  /* ... except, skip leading spaces.  */
  ptr = skip_spaces (ptr);

  return std::string (ptr, end);
}

/* See ada-exp-parser.h.  */

struct type *
write_var_or_type_completion (struct parser_state *par_state,
			      const struct block *block, struct stoken name0)
{
  int tail_index = chop_selector (name0.ptr, name0.length);
  /* If there's no separator, just defer to ordinary symbol
     completion.  */
  if (tail_index == -1)
    return write_var_or_type (par_state, block, name0);

  std::string copy (name0.ptr, tail_index);
  struct type *type = write_var_or_type (par_state, block,
					 { copy.c_str (),
					   (int) copy.length () });
  /* For completion purposes, it's enough that we return a type
     here.  */
  if (type != nullptr)
    return type;

  ada_structop_operation *op = write_selectors (par_state,
						name0.ptr + tail_index);
  op->set_prefix (ada_parser->find_completion_bounds ());
  par_state->mark_struct_expression (op);
  return nullptr;
}

/* See ada-exp-parser.h.  */

void
write_name_assoc (struct parser_state *par_state, struct stoken name)
{
  if (strchr (name.ptr, '.') == NULL)
    {
      std::vector<struct block_symbol> syms
	= ada_lookup_symbol_list (name.ptr,
				  par_state->expression_context_block,
				  SEARCH_VFT);

      if (syms.size () != 1 || syms[0].symbol->loc_class () == LOC_TYPEDEF)
	pstate->push_new<ada_string_operation> (copy_name (name));
      else
	write_var_from_sym (par_state, syms[0]);
    }
  else
    if (write_var_or_type (par_state, NULL, name) != NULL)
      error (_("Invalid use of type."));

  push_association<ada_name_association> (ada_pop ());
}

/* See ada-exp-parser.h.  */

struct type *
type_for_char (struct parser_state *par_state, ULONGEST value)
{
  if (value <= 0xff)
    return language_string_char_type (par_state->language (),
				      par_state->gdbarch ());
  else if (value <= 0xffff)
    return language_lookup_primitive_type (par_state->language (),
					   par_state->gdbarch (),
					   "wide_character");
  return language_lookup_primitive_type (par_state->language (),
					 par_state->gdbarch (),
					 "wide_wide_character");
}

/* See ada-exp-parser.h.  */

struct type *
type_system_address (struct parser_state *par_state)
{
  struct type *type
    = language_lookup_primitive_type (par_state->language (),
				      par_state->gdbarch (),
				      "system__address");
  return  type != NULL ? type : parse_type (par_state)->builtin_data_ptr;
}

/* See ada-exp-parser.h.  */

void
ada_yyerror (const char *msg)
{
  ada_exp_parser::pstate->parse_error (msg);
}

} /* namespace ada_exp_parser */

/* See ada-exp-parser.h.  */

int
ada_parse (struct parser_state *par_state)
{
  using namespace ada_exp_parser;

  /* Setting up the parser state.  */
  scoped_restore pstate_restore = make_scoped_restore (&pstate, par_state);
  gdb_assert (par_state != NULL);

  ada_parse_state parser (par_state->lexptr);
  scoped_restore parser_restore = make_scoped_restore (&ada_parser, &parser);

  scoped_restore restore_yydebug = make_scoped_restore (&ada_yydebug,
							par_state->debug);

  int result = ada_yyparse ();
  if (!result)
    {
      struct type *context_type = nullptr;
      if (par_state->void_context_p)
	context_type = parse_type (par_state)->builtin_void;
      pstate->set_operation (ada_pop (true, context_type));
    }
  return result;
}

INIT_GDB_FILE (ada_exp_parser)
{
#if GDB_SELF_TEST
  selftests::register_test ("ada-lex", ada_exp_parser::selftests::ada_lex_tests);
#endif
}
