/* Helper functions for the C++ name parser, for GDB.

   Copyright (C) 2003-2026 Free Software Foundation, Inc.

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

#include "cp-name-parser.h"
#include "cp-name-parser-gen.h"
#include "c-support.h"
#include "cp-support.h"
#include "gdbsupport/gdb_obstack.h"
#include "gdbsupport/selftest.h"

/* The entry point of the bison/yacc-generated parser, defined in
   cp-name-parser-gen.c.  Bison produces a declaration for yyparse in
   cp-name-parser-gen.h, but byacc does not, hence this declaration.  */

int cpname_yyparse (cp_name_parser::cpname_state *state);

namespace cp_name_parser {

struct demangle_component *
cpname_state::d_grab ()
{
  return obstack_new<demangle_component> (&demangle_info->obstack);
}

/* Helper functions.  These wrap the demangler tree interface, handle
   allocation from our global store, and return the allocated component.  */

struct demangle_component *
cpname_state::fill_comp (enum demangle_component_type d_type,
			 struct demangle_component *lhs,
			 struct demangle_component *rhs)
{
  struct demangle_component *ret = d_grab ();
  int i;

  i = cplus_demangle_fill_component (ret, d_type, lhs, rhs);
  gdb_assert (i);

  return ret;
}

struct demangle_component *
cpname_state::make_operator (const char *name, int args)
{
  struct demangle_component *ret = d_grab ();
  int i;

  i = cplus_demangle_fill_operator (ret, name, args);
  gdb_assert (i);

  return ret;
}

struct demangle_component *
cpname_state::make_dtor (enum gnu_v3_dtor_kinds kind,
			 struct demangle_component *name)
{
  struct demangle_component *ret = d_grab ();
  int i;

  i = cplus_demangle_fill_dtor (ret, kind, name);
  gdb_assert (i);

  return ret;
}

struct demangle_component *
cpname_state::make_builtin_type (const char *name)
{
  struct demangle_component *ret = d_grab ();
  int i;

  i = cplus_demangle_fill_builtin_type (ret, name);
  gdb_assert (i);

  return ret;
}

struct demangle_component *
cpname_state::make_name (const char *name, int len)
{
  struct demangle_component *ret = d_grab ();
  int i;

  i = cplus_demangle_fill_name (ret, name, len);
  gdb_assert (i);

  return ret;
}

/* Apply QUALIFIERS to LHS and return a qualified component.  IS_METHOD
   is set if LHS is a method, in which case the qualifiers are logically
   applied to "this".  We apply qualifiers in a consistent order; LHS
   may already be qualified; duplicate qualifiers are not created.  */

struct demangle_component *
cpname_state::d_qualify (struct demangle_component *lhs, int qualifiers,
			 int is_method)
{
  struct demangle_component **inner_p;
  enum demangle_component_type type;

  /* For now the order is CONST (innermost), VOLATILE, RESTRICT.  */

#define HANDLE_QUAL(TYPE, MTYPE, QUAL)				\
  if ((qualifiers & QUAL) && (type != TYPE) && (type != MTYPE))	\
    {								\
      *inner_p = fill_comp (is_method ? MTYPE : TYPE,		\
			    *inner_p, NULL);			\
      inner_p = &d_left (*inner_p);				\
      type = (*inner_p)->type;					\
    }								\
  else if (type == TYPE || type == MTYPE)			\
    {								\
      inner_p = &d_left (*inner_p);				\
      type = (*inner_p)->type;					\
    }

  inner_p = &lhs;

  type = (*inner_p)->type;

  HANDLE_QUAL (DEMANGLE_COMPONENT_RESTRICT, DEMANGLE_COMPONENT_RESTRICT_THIS, QUAL_RESTRICT);
  HANDLE_QUAL (DEMANGLE_COMPONENT_VOLATILE, DEMANGLE_COMPONENT_VOLATILE_THIS, QUAL_VOLATILE);
  HANDLE_QUAL (DEMANGLE_COMPONENT_CONST, DEMANGLE_COMPONENT_CONST_THIS, QUAL_CONST);
#undef HANDLE_QUAL

  return lhs;
}

/* Return a builtin type corresponding to FLAGS.  */

struct demangle_component *
cpname_state::d_int_type (int flags)
{
  const char *name;

  switch (flags)
    {
    case INT_SIGNED | INT_CHAR:
      name = "signed char";
      break;
    case INT_CHAR:
      name = "char";
      break;
    case INT_UNSIGNED | INT_CHAR:
      name = "unsigned char";
      break;
    case 0:
    case INT_SIGNED:
      name = "int";
      break;
    case INT_UNSIGNED:
      name = "unsigned int";
      break;
    case INT_LONG:
    case INT_SIGNED | INT_LONG:
      name = "long";
      break;
    case INT_UNSIGNED | INT_LONG:
      name = "unsigned long";
      break;
    case INT_SHORT:
    case INT_SIGNED | INT_SHORT:
      name = "short";
      break;
    case INT_UNSIGNED | INT_SHORT:
      name = "unsigned short";
      break;
    case INT_LLONG | INT_LONG:
    case INT_SIGNED | INT_LLONG | INT_LONG:
      name = "long long";
      break;
    case INT_UNSIGNED | INT_LLONG | INT_LONG:
      name = "unsigned long long";
      break;
    default:
      return NULL;
    }

  return make_builtin_type (name);
}

/* Wrapper to create a unary operation.  */

struct demangle_component *
cpname_state::d_unary (const char *name, struct demangle_component *lhs)
{
  return fill_comp (DEMANGLE_COMPONENT_UNARY, make_operator (name, 1), lhs);
}

/* Wrapper to create a binary operation.  */

struct demangle_component *
cpname_state::d_binary (const char *name, struct demangle_component *lhs,
			struct demangle_component *rhs)
{
  return fill_comp (DEMANGLE_COMPONENT_BINARY, make_operator (name, 2),
		    fill_comp (DEMANGLE_COMPONENT_BINARY_ARGS, lhs, rhs));
}

/* Take care of parsing a number (anything that starts with a digit).
   The number starts at P and contains LEN characters.  Store the result in
   YYLVAL.  */

int
cpname_state::parse_number (const char *p, int len, int parsed_float,
			    cp_name_parser_YYSTYPE *lvalp)
{
  int unsigned_p = 0;

  /* Number of "L" suffixes encountered.  */
  int long_p = 0;

  struct demangle_component *type, *name;
  enum demangle_component_type literal_type;

  if (p[0] == '-')
    {
      literal_type = DEMANGLE_COMPONENT_LITERAL_NEG;
      p++;
      len--;
    }
  else
    literal_type = DEMANGLE_COMPONENT_LITERAL;

  if (parsed_float)
    {
      /* It's a float since it contains a point or an exponent.  */
      char c;

      /* The GDB lexer checks the result of scanf at this point.  Not doing
	 this leaves our error checking slightly weaker but only for invalid
	 data.  */

      /* See if it has `f' or `l' suffix (float or long double).  */

      c = c_tolower (p[len - 1]);

      if (c == 'f')
	{
	  len--;
	  type = make_builtin_type ("float");
	}
      else if (c == 'l')
	{
	  len--;
	  type = make_builtin_type ("long double");
	}
      else if (c_isdigit (c) || c == '.')
	type = make_builtin_type ("double");
      else
	return ERROR;

      name = make_name (p, len);
      lvalp->comp = fill_comp (literal_type, type, name);

      return FLOAT;
    }

  /* Note that we do not automatically generate unsigned types.  This
     can't be done because we don't have access to the gdbarch
     here.  */

  int base = 10;
  if (len > 1 && p[0] == '0')
    {
      if (p[1] == 'x' || p[1] == 'X')
	{
	  base = 16;
	  p += 2;
	  len -= 2;
	}
      else if (p[1] == 'b' || p[1] == 'B')
	{
	  base = 2;
	  p += 2;
	  len -= 2;
	}
      else if (p[1] == 'd' || p[1] == 'D' || p[1] == 't' || p[1] == 'T')
	{
	  /* Apparently gdb extensions.  */
	  base = 10;
	  p += 2;
	  len -= 2;
	}
      else
	base = 8;
    }

  long_p = 0;
  unsigned_p = 0;
  while (len > 0)
    {
      if (p[len - 1] == 'l' || p[len - 1] == 'L')
	{
	  len--;
	  long_p++;
	  continue;
	}
      if (p[len - 1] == 'u' || p[len - 1] == 'U')
	{
	  len--;
	  unsigned_p++;
	  continue;
	}
      break;
    }

  /* Use gdb_mpz here in case a 128-bit value appears.  */
  gdb_mpz value (0);
  for (int off = 0; off < len; ++off)
    {
      int dig;

      if (c_isdigit (p[off]))
	dig = p[off] - '0';
      else
	dig = c_tolower (p[off]) - 'a' + 10;

      if (dig >= base)
	return ERROR;

      value *= base;
      value += dig;
    }

  std::string printed = value.str ();
  const char *copy = obstack_strdup (&demangle_info->obstack, printed);

  if (long_p == 0)
    {
      if (unsigned_p)
	type = make_builtin_type ("unsigned int");
      else
	type = make_builtin_type ("int");
    }
  else if (long_p == 1)
    {
      if (unsigned_p)
	type = make_builtin_type ("unsigned long");
      else
	type = make_builtin_type ("long");
    }
  else
    {
      if (unsigned_p)
	type = make_builtin_type ("unsigned long long");
      else
	type = make_builtin_type ("long long");
    }

  name = make_name (copy, strlen (copy));
  lvalp->comp = fill_comp (literal_type, type, name);

  return INT;
}

} /* namespace cp_name_parser */

/* See cp-support.h.  */

gdb::unique_xmalloc_ptr<char>
cp_comp_to_string (struct demangle_component *result, int estimated_len)
{
  size_t err;

  char *res = gdb_cplus_demangle_print (DMGL_PARAMS | DMGL_ANSI,
					result, estimated_len, &err);
  return gdb::unique_xmalloc_ptr<char> (res);
}

/* Merge the two parse trees given by DEST and SRC.  The parse tree
   in SRC is attached to DEST at the node represented by TARGET.

   NOTE 1: Since there is no API to merge obstacks, this function does
   even attempt to try it.  Fortunately, we do not (yet?) need this ability.
   The code will assert if SRC->obstack is not empty.

   NOTE 2: The string from which SRC was parsed must not be freed, since
   this function will place pointers to that string into DEST.  */

void
cp_merge_demangle_parse_infos (struct demangle_parse_info *dest,
			       struct demangle_component *target,
			       demangle_parse_info_up src)
{
  /* Copy the SRC's parse data into DEST.  */
  *target = *src->tree;

  /* Make sure SRC is owned by DEST.  */
  dest->infos.push_back (std::move (src));
}

/* Convert a demangled name to a demangle_component tree.  On success,
   a structure containing the root of the new tree is returned.  On
   error, NULL is returned, and an error message will be set in
   *ERRMSG.  */

demangle_parse_info_up
cp_demangled_name_to_comp (const char *demangled_name,
			   std::string *errmsg)
{
  auto result = std::make_unique<demangle_parse_info> ();
  cp_name_parser::cpname_state state (demangled_name, result.get ());

  /* Note that we can't set yydebug here, as is done in the other
     parsers.  Bison implements yydebug as a global, even with a pure
     parser, and this parser is run from worker threads.  So, changing
     yydebug causes TSan reports.  If you need to debug this parser,
     debug gdb and set the global from the outer gdb.  */
  if (cpname_yyparse (&state))
    {
      if (state.global_errmsg && errmsg)
	*errmsg = state.global_errmsg;
      return NULL;
    }

  result->tree = state.global_result;

  return result;
}

/* Find the end of a symbol name starting at LEXPTR.  */

static const char *
symbol_end (const char *lexptr)
{
  const char *p = lexptr;

  while (*p && (c_ident_is_alnum (*p) || *p == '_' || *p == '$' || *p == '.'))
    p++;

  return p;
}

static const char backslashable[] = "abefnrtv";
static const char represented[] = "\a\b\e\f\n\r\t\v";

/* Translate the backslash the way we would in the host character set.  */
static int
c_parse_backslash (int host_char, int *target_char)
{
  const char *ix;
  ix = strchr (backslashable, host_char);
  if (! ix)
    return 0;
  else
    *target_char = represented[ix - backslashable];
  return 1;
}

/* Parse a C escape sequence.  STRING_PTR points to a variable
   containing a pointer to the string to parse.  That pointer
   should point to the character after the \.  That pointer
   is updated past the characters we use.  The value of the
   escape sequence is returned.

   A negative value means the sequence \ newline was seen,
   which is supposed to be equivalent to nothing at all.

   If \ is followed by a null character, we return a negative
   value and leave the string pointer pointing at the null character.

   If \ is followed by 000, we return 0 and leave the string pointer
   after the zeros.  A value of 0 does not mean end of string.  */

static int
cp_parse_escape (const char **string_ptr)
{
  int target_char;
  int c = *(*string_ptr)++;
  if (c_parse_backslash (c, &target_char))
    return target_char;
  else
    switch (c)
      {
      case '\n':
	return -2;
      case 0:
	(*string_ptr)--;
	return 0;
      case '^':
	{
	  c = *(*string_ptr)++;

	  if (c == '?')
	    return 0177;
	  else if (c == '\\')
	    target_char = cp_parse_escape (string_ptr);
	  else
	    target_char = c;

	  /* Now target_char is something like `c', and we want to find
	     its control-character equivalent.  */
	  target_char = target_char & 037;

	  return target_char;
	}

      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
	{
	  int i = c - '0';
	  int count = 0;
	  while (++count < 3)
	    {
	      c = (**string_ptr);
	      if (c >= '0' && c <= '7')
		{
		  (*string_ptr)++;
		  i *= 8;
		  i += c - '0';
		}
	      else
		{
		  break;
		}
	    }
	  return i;
	}
      default:
	return c;
      }
}

#define HANDLE_SPECIAL(string, comp)				\
  if (startswith (tokstart, string))				\
    {								\
      state->lexptr = tokstart + sizeof (string) - 1;			\
      lvalp->lval = comp;					\
      return DEMANGLER_SPECIAL;					\
    }

#define HANDLE_TOKEN2(string, token)			\
  if (state->lexptr[1] == string[1])				\
    {							\
      state->lexptr += 2;					\
      lvalp->opname = string;				\
      return token;					\
    }

#define HANDLE_TOKEN3(string, token)			\
  if (state->lexptr[1] == string[1] && state->lexptr[2] == string[2])	\
    {							\
      state->lexptr += 3;					\
      lvalp->opname = string;				\
      return token;					\
    }

void
cpname_yyerror (cp_name_parser::cpname_state *state, const char *msg)
{
  if (state->global_errmsg)
    return;

  state->error_lexptr = state->prev_lexptr;
  state->global_errmsg = msg ? msg : "parse error";
}

/* Read one token, getting characters through LEXPTR.  */

int
cpname_yylex (cp_name_parser_YYSTYPE *lvalp,
	      cp_name_parser::cpname_state *state)
{
  int c;
  int namelen;
  const char *tokstart;
  char *copy;

 retry:
  state->prev_lexptr = state->lexptr;
  tokstart = state->lexptr;

  switch (c = *tokstart)
    {
    case 0:
      return 0;

    case ' ':
    case '\t':
    case '\n':
      state->lexptr++;
      goto retry;

    case '\'':
      /* We either have a character constant ('0' or '\177' for example)
	 or we have a quoted symbol reference ('foo(int,int)' in C++
	 for example). */
      state->lexptr++;
      c = *state->lexptr++;
      if (c == '\\')
	c = cp_parse_escape (&state->lexptr);
      else if (c == '\'')
	{
	  cpname_yyerror (state, _("empty character constant"));
	  return ERROR;
	}

      /* We over-allocate here, but it doesn't really matter . */
      copy = (char *) obstack_alloc (&state->demangle_info->obstack, 30);
      xsnprintf (copy, 30, "%d", c);

      c = *state->lexptr++;
      if (c != '\'')
	{
	  cpname_yyerror (state, _("invalid character constant"));
	  return ERROR;
	}

      lvalp->comp
	= state->fill_comp (DEMANGLE_COMPONENT_LITERAL,
			    state->make_builtin_type ("char"),
			    state->make_name (copy, strlen (copy)));

      return INT;

    case '(':
      if (startswith (tokstart, "(anonymous namespace)"))
	{
	  state->lexptr += 21;
	  lvalp->comp = state->make_name ("(anonymous namespace)",
					  sizeof "(anonymous namespace)" - 1);
	  return NAME;
	}
	[[fallthrough]];

    case ')':
    case ',':
      state->lexptr++;
      return c;

    case '.':
      if (state->lexptr[1] == '.' && state->lexptr[2] == '.')
	{
	  state->lexptr += 3;
	  return ELLIPSIS;
	}

      /* Might be a floating point number.  */
      if (state->lexptr[1] < '0' || state->lexptr[1] > '9')
	goto symbol;		/* Nope, must be a symbol. */

      goto try_number;

    case '-':
      HANDLE_TOKEN2 ("-=", ASSIGN_MODIFY);
      HANDLE_TOKEN2 ("--", DECREMENT);
      HANDLE_TOKEN2 ("->", ARROW);

      /* For construction vtables.  This is kind of hokey.  */
      if (startswith (tokstart, "-in-"))
	{
	  state->lexptr += 4;
	  return CONSTRUCTION_IN;
	}

      if (state->lexptr[1] < '0' || state->lexptr[1] > '9')
	{
	  state->lexptr++;
	  return '-';
	}

    try_number:
      [[fallthrough]];
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      {
	/* It's a number.  */
	int got_dot = 0, got_e = 0, toktype;
	const char *p = tokstart;
	int hex = 0;

	if (c == '-')
	  p++;

	if (c == '0' && (p[1] == 'x' || p[1] == 'X'))
	  {
	    p += 2;
	    hex = 1;
	  }
	else if (c == '0' && (p[1]=='t' || p[1]=='T' || p[1]=='d' || p[1]=='D'))
	  {
	    p += 2;
	    hex = 0;
	  }

	/* If the token includes the C++14 digits separator, we make a
	   copy so that we don't have to handle the separator in
	   parse_number.  */
	std::optional<std::string> no_tick;
	for (;; ++p)
	  {
	    /* This test includes !hex because 'e' is a valid hex digit
	       and thus does not indicate a floating point number when
	       the radix is hex.  */
	    if (!hex && !got_e && (*p == 'e' || *p == 'E'))
	      got_dot = got_e = 1;
	    /* This test does not include !hex, because a '.' always indicates
	       a decimal floating point number regardless of the radix.

	       NOTE drow/2005-03-09: This comment is not accurate in C99;
	       however, it's not clear that all the floating point support
	       in this file is doing any good here.  */
	    else if (!got_dot && *p == '.')
	      got_dot = 1;
	    else if (got_e && (p[-1] == 'e' || p[-1] == 'E')
		     && (*p == '-' || *p == '+'))
	      {
		/* This is the sign of the exponent, not the end of
		   the number.  */
	      }
	    /* C++14 allows a separator.  */
	    else if (*p == '\'')
	      {
		if (!no_tick.has_value ())
		  no_tick.emplace (tokstart, p);
		continue;
	      }
	    /* We will take any letters or digits.  parse_number will
	       complain if past the radix, or if L or U are not final.  */
	    else if (! c_isalnum (*p))
	      break;
	    if (no_tick.has_value ())
	      no_tick->push_back (*p);
	  }
	if (no_tick.has_value ())
	  toktype = state->parse_number (no_tick->c_str (), no_tick->length (),
					 got_dot | got_e, lvalp);
	else
	  toktype = state->parse_number (tokstart, p - tokstart,
					 got_dot | got_e, lvalp);
	if (toktype == ERROR)
	  {
	    cpname_yyerror (state, _("invalid number"));
	    return ERROR;
	  }
	state->lexptr = p;
	return toktype;
      }

    case '+':
      HANDLE_TOKEN2 ("+=", ASSIGN_MODIFY);
      HANDLE_TOKEN2 ("++", INCREMENT);
      state->lexptr++;
      return c;
    case '*':
      HANDLE_TOKEN2 ("*=", ASSIGN_MODIFY);
      state->lexptr++;
      return c;
    case '/':
      HANDLE_TOKEN2 ("/=", ASSIGN_MODIFY);
      state->lexptr++;
      return c;
    case '%':
      HANDLE_TOKEN2 ("%=", ASSIGN_MODIFY);
      state->lexptr++;
      return c;
    case '|':
      HANDLE_TOKEN2 ("|=", ASSIGN_MODIFY);
      HANDLE_TOKEN2 ("||", OROR);
      state->lexptr++;
      return c;
    case '&':
      HANDLE_TOKEN2 ("&=", ASSIGN_MODIFY);
      HANDLE_TOKEN2 ("&&", ANDAND);
      state->lexptr++;
      return c;
    case '^':
      HANDLE_TOKEN2 ("^=", ASSIGN_MODIFY);
      state->lexptr++;
      return c;
    case '!':
      HANDLE_TOKEN2 ("!=", NOTEQUAL);
      state->lexptr++;
      return c;
    case '<':
      HANDLE_TOKEN3 ("<<=", ASSIGN_MODIFY);
      HANDLE_TOKEN3 ("<=>", SPACESHIP);
      HANDLE_TOKEN2 ("<=", LEQ);
      HANDLE_TOKEN2 ("<<", LSH);
      state->lexptr++;
      return c;
    case '>':
      HANDLE_TOKEN3 (">>=", ASSIGN_MODIFY);
      HANDLE_TOKEN2 (">=", GEQ);
      HANDLE_TOKEN2 (">>", RSH);
      state->lexptr++;
      return c;
    case '=':
      HANDLE_TOKEN2 ("==", EQUAL);
      state->lexptr++;
      return c;
    case ':':
      HANDLE_TOKEN2 ("::", COLONCOLON);
      state->lexptr++;
      return c;

    case '[':
    case ']':
    case '?':
    case '@':
    case '~':
    case '{':
    case '}':
    symbol:
      state->lexptr++;
      return c;

    case '"':
      /* These can't occur in C++ names.  */
      cpname_yyerror (state, _("unexpected string literal"));
      return ERROR;
    }

  if (!(c == '_' || c == '$' || c_ident_is_alpha (c)))
    {
      /* We must have come across a bad character (e.g. ';').  */
      cpname_yyerror (state, _("invalid character"));
      return ERROR;
    }

  /* It's a name.  See how long it is.  */
  namelen = 0;
  do
    c = tokstart[++namelen];
  while (c_ident_is_alnum (c) || c == '_' || c == '$');

  state->lexptr += namelen;

  /* Catch specific keywords.  Notice that some of the keywords contain
     spaces, and are sorted by the length of the first word.  They must
     all include a trailing space in the string comparison.  */
  switch (namelen)
    {
    case 16:
      if (startswith (tokstart, "reinterpret_cast"))
	return REINTERPRET_CAST;
      break;
    case 12:
      if (startswith (tokstart, "construction vtable for "))
	{
	  state->lexptr = tokstart + 24;
	  return CONSTRUCTION_VTABLE;
	}
      if (startswith (tokstart, "dynamic_cast"))
	return DYNAMIC_CAST;
      break;
    case 11:
      if (startswith (tokstart, "static_cast"))
	return STATIC_CAST;
      break;
    case 9:
      HANDLE_SPECIAL ("covariant return thunk to ", DEMANGLE_COMPONENT_COVARIANT_THUNK);
      HANDLE_SPECIAL ("reference temporary for ", DEMANGLE_COMPONENT_REFTEMP);
      break;
    case 8:
      HANDLE_SPECIAL ("typeinfo for ", DEMANGLE_COMPONENT_TYPEINFO);
      HANDLE_SPECIAL ("typeinfo fn for ", DEMANGLE_COMPONENT_TYPEINFO_FN);
      HANDLE_SPECIAL ("typeinfo name for ", DEMANGLE_COMPONENT_TYPEINFO_NAME);
      if (startswith (tokstart, "operator"))
	return OPERATOR;
      if (startswith (tokstart, "restrict"))
	return RESTRICT;
      if (startswith (tokstart, "unsigned"))
	return UNSIGNED;
      if (startswith (tokstart, "template"))
	return TEMPLATE;
      if (startswith (tokstart, "volatile"))
	return VOLATILE_KEYWORD;
      break;
    case 7:
      HANDLE_SPECIAL ("virtual thunk to ", DEMANGLE_COMPONENT_VIRTUAL_THUNK);
      if (startswith (tokstart, "wchar_t"))
	return WCHAR_T;
      break;
    case 6:
      if (startswith (tokstart, "global constructors keyed to "))
	{
	  const char *p;
	  state->lexptr = tokstart + 29;
	  lvalp->lval = DEMANGLE_COMPONENT_GLOBAL_CONSTRUCTORS;
	  /* Find the end of the symbol.  */
	  p = symbol_end (state->lexptr);
	  lvalp->comp = state->make_name (state->lexptr, p - state->lexptr);
	  state->lexptr = p;
	  return DEMANGLER_SPECIAL;
	}
      if (startswith (tokstart, "global destructors keyed to "))
	{
	  const char *p;
	  state->lexptr = tokstart + 28;
	  lvalp->lval = DEMANGLE_COMPONENT_GLOBAL_DESTRUCTORS;
	  /* Find the end of the symbol.  */
	  p = symbol_end (state->lexptr);
	  lvalp->comp = state->make_name (state->lexptr, p - state->lexptr);
	  state->lexptr = p;
	  return DEMANGLER_SPECIAL;
	}

      HANDLE_SPECIAL ("vtable for ", DEMANGLE_COMPONENT_VTABLE);
      if (startswith (tokstart, "delete"))
	return DELETE;
      if (startswith (tokstart, "struct"))
	return STRUCT;
      if (startswith (tokstart, "signed"))
	return SIGNED_KEYWORD;
      if (startswith (tokstart, "sizeof"))
	return SIZEOF;
      if (startswith (tokstart, "double"))
	return DOUBLE_KEYWORD;
      break;
    case 5:
      HANDLE_SPECIAL ("guard variable for ", DEMANGLE_COMPONENT_GUARD);
      if (startswith (tokstart, "false"))
	return FALSEKEYWORD;
      if (startswith (tokstart, "class"))
	return CLASS;
      if (startswith (tokstart, "union"))
	return UNION;
      if (startswith (tokstart, "float"))
	return FLOAT_KEYWORD;
      if (startswith (tokstart, "short"))
	return SHORT;
      if (startswith (tokstart, "const"))
	return CONST_KEYWORD;
      break;
    case 4:
      if (startswith (tokstart, "void"))
	return VOID;
      if (startswith (tokstart, "bool"))
	return BOOL;
      if (startswith (tokstart, "char"))
	return CHAR;
      if (startswith (tokstart, "enum"))
	return ENUM;
      if (startswith (tokstart, "long"))
	return LONG;
      if (startswith (tokstart, "true"))
	return TRUEKEYWORD;
      break;
    case 3:
      HANDLE_SPECIAL ("VTT for ", DEMANGLE_COMPONENT_VTT);
      HANDLE_SPECIAL ("non-virtual thunk to ", DEMANGLE_COMPONENT_THUNK);
      if (startswith (tokstart, "new"))
	return NEW;
      if (startswith (tokstart, "int"))
	return INT_KEYWORD;
      break;
    default:
      break;
    }

  lvalp->comp = state->make_name (tokstart, namelen);
  return NAME;
}

#if GDB_SELF_TEST

static void
should_be_the_same (const char *one, const char *two)
{
  gdb::unique_xmalloc_ptr<char> cpone = cp_canonicalize_string (one);
  gdb::unique_xmalloc_ptr<char> cptwo = cp_canonicalize_string (two);

  if (cpone != nullptr)
    one = cpone.get ();
  if (cptwo != nullptr)
    two = cptwo.get ();

  SELF_CHECK (streq (one, two));
}

static void
should_parse (const char *name)
{
  auto parsed = cp_demangled_name_to_comp (name);
  SELF_CHECK (parsed != nullptr);
}

static void
canonicalize_tests ()
{
  should_be_the_same ("short int", "short");
  should_be_the_same ("int short", "short");

  should_be_the_same ("C<(char) 1>::m()", "C<(char) '\\001'>::m()");
  should_be_the_same ("x::y::z<1>", "x::y::z<0x01>");
  should_be_the_same ("x::y::z<1>", "x::y::z<01>");
  should_be_the_same ("x::y::z<(unsigned long long) 1>", "x::y::z<01ull>");
  should_be_the_same ("x::y::z<0b111>", "x::y::z<7>");
  should_be_the_same ("x::y::z<0b111>", "x::y::z<0t7>");
  should_be_the_same ("x::y::z<0b111>", "x::y::z<0D7>");

  should_be_the_same ("x::y::z<0xff'ff>", "x::y::z<65535>");

  should_be_the_same ("something<void ()>", "something<  void()  >");
  should_be_the_same ("something<void ()>", "something<void (void)>");

  should_parse ("void whatever::operator<=><int, int>");

  should_be_the_same ("Foozle<int>::fogey<Empty<int> > (Empty<int>)",
		      "Foozle<int>::fogey<Empty<int>> (Empty<int>)");

  should_be_the_same ("something :: operator new [ ]",
		      "something::operator new[]");
  should_be_the_same ("something :: operator   new",
		      "something::operator new");
  should_be_the_same ("operator()", "operator ()");
}

#endif

INIT_GDB_FILE (cp_name_parser)
{
#if GDB_SELF_TEST
  selftests::register_test ("canonicalize", canonicalize_tests);
#endif
}
