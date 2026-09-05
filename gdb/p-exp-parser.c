/* YACC parser support code for Pascal expressions, for GDB.

   Copyright (C) 2000-2026 Free Software Foundation, Inc.

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

#include "p-exp-parser.h"
#include "p-exp-parser-gen.h"
#include "block.h"
#include "expression.h"
#include "language.h"
#include "value.h"

/* The entry point of the bison/yacc-generated parser, defined in
   p-exp-parser-gen.c.  Bison produces a declaration for pascal_yyparse in
   p-exp-parser-gen.h, but byacc does not, hence this declaration.  */

int pascal_yyparse ();

/* Likewise, byacc does not produce a declaration for pascal_yydebug.  */

extern int pascal_yydebug;

namespace p_exp_parser
{

/* See p-exp-parser.h.  */

parser_state *pstate;

/* Depth of parentheses.  */

static int paren_depth;

/* See p-exp-parser.h.  */

struct type *current_type;

/* See p-exp-parser.h.  */

int leftdiv_is_integer;

/* See p-exp-parser.h.  */

int search_field;

/* See p-exp-parser.h.  */

int
parse_number (struct parser_state *par_state,
	      const char *p, int len, int parsed_float, p_exp_parser_YYSTYPE *putithere)
{
  ULONGEST n = 0;
  ULONGEST prevn = 0;

  int i = 0;
  int c;
  int base = input_radix;
  int unsigned_p = 0;

  /* Number of "L" suffixes encountered.  */
  int long_p = 0;

  /* We have found a "L" or "U" suffix.  */
  int found_suffix = 0;

  if (parsed_float)
    {
      /* Handle suffixes: 'f' for float, 'l' for long double.
	 FIXME: This appears to be an extension -- do we want this?  */
      if (len >= 1 && c_tolower (p[len - 1]) == 'f')
	{
	  putithere->typed_val_float.type
	    = parse_type (par_state)->builtin_float;
	  len--;
	}
      else if (len >= 1 && c_tolower (p[len - 1]) == 'l')
	{
	  putithere->typed_val_float.type
	    = parse_type (par_state)->builtin_long_double;
	  len--;
	}
      /* Default type for floating-point literals is double.  */
      else
	{
	  putithere->typed_val_float.type
	    = parse_type (par_state)->builtin_double;
	}

      if (!parse_float (p, len,
			putithere->typed_val_float.type,
			putithere->typed_val_float.val))
	return ERROR;
      return FLOAT;
    }

  /* Handle base-switching prefixes 0x, 0t, 0d, 0.  */
  if (p[0] == '0' && len > 1)
    switch (p[1])
      {
      case 'x':
      case 'X':
	if (len >= 3)
	  {
	    p += 2;
	    base = 16;
	    len -= 2;
	  }
	break;

      case 't':
      case 'T':
      case 'd':
      case 'D':
	if (len >= 3)
	  {
	    p += 2;
	    base = 10;
	    len -= 2;
	  }
	break;

      default:
	base = 8;
	break;
      }

  while (len-- > 0)
    {
      c = *p++;
      if (c >= 'A' && c <= 'Z')
	c += 'a' - 'A';
      if (c != 'l' && c != 'u')
	n *= base;
      if (c >= '0' && c <= '9')
	{
	  if (found_suffix)
	    return ERROR;
	  n += i = c - '0';
	}
      else
	{
	  if (base > 10 && c >= 'a' && c <= 'f')
	    {
	      if (found_suffix)
		return ERROR;
	      n += i = c - 'a' + 10;
	    }
	  else if (c == 'l')
	    {
	      ++long_p;
	      found_suffix = 1;
	    }
	  else if (c == 'u')
	    {
	      unsigned_p = 1;
	      found_suffix = 1;
	    }
	  else
	    return ERROR;	/* Char not a digit */
	}
      if (i >= base)
	return ERROR;		/* Invalid digit in this base.  */

      if (c != 'l' && c != 'u')
	{
	  /* Test for overflow.  */
	  if (prevn == 0 && n == 0)
	    ;
	  else if (prevn >= n)
	    error (_("Numeric constant too large."));
	}
      prevn = n;
    }

  /* An integer constant is an int, a long, or a long long.  An L
     suffix forces it to be long; an LL suffix forces it to be long
     long.  If not forced to a larger size, it gets the first type of
     the above that it fits in.  To figure out whether it fits, we
     shift it right and see whether anything remains.  Note that we
     can't shift sizeof (LONGEST) * HOST_CHAR_BIT bits or more in one
     operation, because many compilers will warn about such a shift
     (which always produces a zero result).  Sometimes gdbarch_int_bit
     or gdbarch_long_bit will be that big, sometimes not.  To deal with
     the case where it is we just always shift the value more than
     once, with fewer bits each time.  */

  int int_bits = gdbarch_int_bit (par_state->gdbarch ());
  int long_bits = gdbarch_long_bit (par_state->gdbarch ());
  int long_long_bits = gdbarch_long_long_bit (par_state->gdbarch ());
  bool have_signed = !unsigned_p;
  bool have_int = long_p == 0;
  bool have_long = long_p <= 1;
  if (have_int && have_signed && fits_in_type (1, n, int_bits, true))
    putithere->typed_val_int.type = parse_type (par_state)->builtin_int;
  else if (have_int && fits_in_type (1, n, int_bits, false))
    putithere->typed_val_int.type
      = parse_type (par_state)->builtin_unsigned_int;
  else if (have_long && have_signed && fits_in_type (1, n, long_bits, true))
    putithere->typed_val_int.type = parse_type (par_state)->builtin_long;
  else if (have_long && fits_in_type (1, n, long_bits, false))
    putithere->typed_val_int.type
      = parse_type (par_state)->builtin_unsigned_long;
  else if (have_signed && fits_in_type (1, n, long_long_bits, true))
    putithere->typed_val_int.type
      = parse_type (par_state)->builtin_long_long;
  else if (fits_in_type (1, n, long_long_bits, false))
    putithere->typed_val_int.type
      = parse_type (par_state)->builtin_unsigned_long_long;
  else
    error (_("Numeric constant too large."));
  putithere->typed_val_int.val = n;

   return INT;
}


struct type_push
{
  struct type *stored;
  struct type_push *next;
};

static struct type_push *tp_top = NULL;

/* See p-exp-parser.h.  */

void
push_current_type (void)
{
  struct type_push *tpnew;
  tpnew = (struct type_push *) xmalloc (sizeof (struct type_push));
  tpnew->next = tp_top;
  tpnew->stored = current_type;
  current_type = NULL;
  tp_top = tpnew;
}

/* See p-exp-parser.h.  */

void
pop_current_type (void)
{
  struct type_push *tp = tp_top;
  if (tp)
    {
      current_type = tp->stored;
      tp_top = tp->next;
      xfree (tp);
    }
}

struct p_token
{
  const char *oper;
  int token;
  enum exp_opcode opcode;
};

static const struct p_token tokentab3[] =
  {
    {"shr", RSH, OP_NULL},
    {"shl", LSH, OP_NULL},
    {"and", ANDAND, OP_NULL},
    {"div", DIV, OP_NULL},
    {"not", NOT, OP_NULL},
    {"mod", MOD, OP_NULL},
    {"inc", INCREMENT, OP_NULL},
    {"dec", DECREMENT, OP_NULL},
    {"xor", XOR, OP_NULL}
  };

static const struct p_token tokentab2[] =
  {
    {"or", OR, OP_NULL},
    {"<>", NOTEQUAL, OP_NULL},
    {"<=", LEQ, OP_NULL},
    {">=", GEQ, OP_NULL},
    {":=", ASSIGN, OP_NULL},
    {"::", COLONCOLON, OP_NULL} };

/* Allocate uppercased var: */
/* make an uppercased copy of tokstart.  */
static char *
uptok (const char *tokstart, int namelen)
{
  int i;
  char *uptokstart = (char *)xmalloc(namelen+1);
  for (i = 0;i <= namelen;i++)
    {
      if ((tokstart[i]>='a' && tokstart[i]<='z'))
	uptokstart[i] = tokstart[i]-('a'-'A');
      else
	uptokstart[i] = tokstart[i];
    }
  uptokstart[namelen]='\0';
  return uptokstart;
}

/* Skip over a Pascal string.  STR must point to the opening single quote
   character.  This function returns a pointer to the character after the
   closing single quote character.

   This function does not support embedded, escaped single quotes, which
   is done by placing two consecutive single quotes into a string.
   Support for this would be easy to add, but this function is only used
   from the Python expression parser, and if we did skip over escaped
   quotes then the rest of the expression parser wouldn't handle them
   correctly.  */
static const char *
pascal_skip_string (const char *str)
{
  gdb_assert (*str == '\'');

  do
    ++str;
  while (*str != '\0' && *str != '\'');

  return str;
}

/* See p-exp-parser.h.  */

int
pascal_yylex (void)
{
  int c;
  int namelen;
  const char *tokstart;
  char *uptokstart;
  const char *tokptr;
  int explen, tempbufindex;
  static char *tempbuf;
  static int tempbufsize;

 retry:

  pstate->prev_lexptr = pstate->lexptr;

  tokstart = pstate->lexptr;
  explen = strlen (pstate->lexptr);

  /* See if it is a special token of length 3.  */
  if (explen > 2)
    for (const auto &token : tokentab3)
      if (strncasecmp (tokstart, token.oper, 3) == 0
	  && (!c_isalpha (token.oper[0]) || explen == 3
	      || (!c_isalpha (tokstart[3])
		  && !c_isdigit (tokstart[3]) && tokstart[3] != '_')))
	{
	  pstate->lexptr += 3;
	  pascal_yylval.opcode = token.opcode;
	  return token.token;
	}

  /* See if it is a special token of length 2.  */
  if (explen > 1)
    for (const auto &token : tokentab2)
      if (strncasecmp (tokstart, token.oper, 2) == 0
	  && (!c_isalpha (token.oper[0]) || explen == 2
	      || (!c_isalpha (tokstart[2])
		  && !c_isdigit (tokstart[2]) && tokstart[2] != '_')))
	{
	  pstate->lexptr += 2;
	  pascal_yylval.opcode = token.opcode;
	  return token.token;
	}

  switch (c = *tokstart)
    {
    case 0:
      if (search_field && pstate->parse_completion)
	return COMPLETE;
      else
       return 0;

    case ' ':
    case '\t':
    case '\n':
      pstate->lexptr++;
      goto retry;

    case '\'':
      /* We either have a character constant ('0' or '\177' for example)
	 or we have a quoted symbol reference ('foo(int,int)' in object pascal
	 for example).  */
      pstate->lexptr++;
      c = *pstate->lexptr++;
      if (c == '\\')
	c = parse_escape (pstate->gdbarch (), &pstate->lexptr);
      else if (c == '\'')
	error (_("Empty character constant."));

      pascal_yylval.typed_val_int.val = c;
      pascal_yylval.typed_val_int.type = parse_type (pstate)->builtin_char;

      c = *pstate->lexptr++;
      if (c != '\'')
	{
	  namelen = pascal_skip_string (tokstart) - tokstart;
	  if (namelen > 2)
	    {
	      pstate->lexptr = tokstart + namelen;
	      if (pstate->lexptr[-1] != '\'')
		error (_("Unmatched single quote."));
	      namelen -= 2;
	      tokstart++;
	      uptokstart = uptok(tokstart,namelen);
	      goto tryname;
	    }
	  error (_("Invalid character constant."));
	}
      return INT;

    case '(':
      paren_depth++;
      pstate->lexptr++;
      return c;

    case ')':
      if (paren_depth == 0)
	return 0;
      paren_depth--;
      pstate->lexptr++;
      return c;

    case ',':
      if (pstate->comma_terminates && paren_depth == 0)
	return 0;
      pstate->lexptr++;
      return c;

    case '.':
      /* Might be a floating point number.  */
      if (pstate->lexptr[1] < '0' || pstate->lexptr[1] > '9')
	{
	  goto symbol;		/* Nope, must be a symbol.  */
	}

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
	int hex = input_radix > 10;

	if (c == '0' && (p[1] == 'x' || p[1] == 'X'))
	  {
	    p += 2;
	    hex = 1;
	  }
	else if (c == '0' && (p[1]=='t' || p[1]=='T'
			      || p[1]=='d' || p[1]=='D'))
	  {
	    p += 2;
	    hex = 0;
	  }

	for (;; ++p)
	  {
	    /* This test includes !hex because 'e' is a valid hex digit
	       and thus does not indicate a floating point number when
	       the radix is hex.  */
	    if (!hex && !got_e && (*p == 'e' || *p == 'E'))
	      got_dot = got_e = 1;
	    /* This test does not include !hex, because a '.' always indicates
	       a decimal floating point number regardless of the radix.  */
	    else if (!got_dot && *p == '.')
	      got_dot = 1;
	    else if (got_e && (p[-1] == 'e' || p[-1] == 'E')
		     && (*p == '-' || *p == '+'))
	      /* This is the sign of the exponent, not the end of the
		 number.  */
	      continue;
	    /* We will take any letters or digits.  parse_number will
	       complain if past the radix, or if L or U are not final.  */
	    else if ((*p < '0' || *p > '9')
		     && ((*p < 'a' || *p > 'z')
				  && (*p < 'A' || *p > 'Z')))
	      break;
	  }
	toktype = parse_number (pstate, tokstart,
				p - tokstart, got_dot | got_e, &pascal_yylval);
	if (toktype == ERROR)
	  error (_("Invalid number \"%.*s\"."), (int) (p - tokstart),
		 tokstart);
	pstate->lexptr = p;
	return toktype;
      }

    case '+':
    case '-':
    case '*':
    case '/':
    case '|':
    case '&':
    case '^':
    case '~':
    case '!':
    case '@':
    case '<':
    case '>':
    case '[':
    case ']':
    case '?':
    case ':':
    case '=':
    case '{':
    case '}':
    symbol:
      pstate->lexptr++;
      return c;

    case '"':

      /* Build the gdb internal form of the input string in tempbuf,
	 translating any standard C escape forms seen.  Note that the
	 buffer is null byte terminated *only* for the convenience of
	 debugging gdb itself and printing the buffer contents when
	 the buffer contains no embedded nulls.  Gdb does not depend
	 upon the buffer being null byte terminated, it uses the length
	 string instead.  This allows gdb to handle C strings (as well
	 as strings in other languages) with embedded null bytes.  */

      tokptr = ++tokstart;
      tempbufindex = 0;

      do {
	/* Grow the static temp buffer if necessary, including allocating
	   the first one on demand.  */
	if (tempbufindex + 1 >= tempbufsize)
	  {
	    tempbuf = (char *) xrealloc (tempbuf, tempbufsize += 64);
	  }

	switch (*tokptr)
	  {
	  case '\0':
	  case '"':
	    /* Do nothing, loop will terminate.  */
	    break;
	  case '\\':
	    ++tokptr;
	    c = parse_escape (pstate->gdbarch (), &tokptr);
	    if (c == -1)
	      {
		continue;
	      }
	    tempbuf[tempbufindex++] = c;
	    break;
	  default:
	    tempbuf[tempbufindex++] = *tokptr++;
	    break;
	  }
      } while ((*tokptr != '"') && (*tokptr != '\0'));
      if (*tokptr++ != '"')
	{
	  error (_("Unterminated string in expression."));
	}
      tempbuf[tempbufindex] = '\0';	/* See note above.  */
      pascal_yylval.sval.ptr = tempbuf;
      pascal_yylval.sval.length = tempbufindex;
      pstate->lexptr = tokptr;
      return (STRING);
    }

  if (!(c == '_' || c == '$'
	|| (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')))
    /* We must have come across a bad character (e.g. ';').  */
    error (_("Invalid character '%c' in expression."), c);

  /* It's a name.  See how long it is.  */
  namelen = 0;
  for (c = tokstart[namelen];
       (c == '_' || c == '$' || (c >= '0' && c <= '9')
	|| (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '<');)
    {
      /* Template parameter lists are part of the name.
	 FIXME: This mishandles `print $a<4&&$a>3'.  */
      if (c == '<')
	{
	  int i = namelen;
	  int nesting_level = 1;
	  while (tokstart[++i])
	    {
	      if (tokstart[i] == '<')
		nesting_level++;
	      else if (tokstart[i] == '>')
		{
		  if (--nesting_level == 0)
		    break;
		}
	    }
	  if (tokstart[i] == '>')
	    namelen = i;
	  else
	    break;
	}

      /* do NOT uppercase internals because of registers !!!  */
      c = tokstart[++namelen];
    }

  uptokstart = uptok(tokstart,namelen);

  /* The token "if" terminates the expression and is NOT
     removed from the input stream.  */
  if (namelen == 2 && uptokstart[0] == 'I' && uptokstart[1] == 'F')
    {
      xfree (uptokstart);
      return 0;
    }

  pstate->lexptr += namelen;

  tryname:

  /* Catch specific keywords.  Should be done with a data structure.  */
  switch (namelen)
    {
    case 6:
      if (streq (uptokstart, "OBJECT"))
	{
	  xfree (uptokstart);
	  return CLASS;
	}
      if (streq (uptokstart, "RECORD"))
	{
	  xfree (uptokstart);
	  return STRUCT;
	}
      if (streq (uptokstart, "SIZEOF"))
	{
	  xfree (uptokstart);
	  return SIZEOF;
	}
      break;
    case 5:
      if (streq (uptokstart, "CLASS"))
	{
	  xfree (uptokstart);
	  return CLASS;
	}
      if (streq (uptokstart, "FALSE"))
	{
	  pascal_yylval.lval = 0;
	  xfree (uptokstart);
	  return FALSEKEYWORD;
	}
      break;
    case 4:
      if (streq (uptokstart, "TRUE"))
	{
	  pascal_yylval.lval = 1;
	  xfree (uptokstart);
	  return TRUEKEYWORD;
	}
      if (streq (uptokstart, "SELF"))
	{
	  /* Here we search for 'this' like
	     inserted in FPC stabs debug info.  */
	  static const char this_name[] = "this";

	  if (lookup_symbol (this_name, pstate->expression_context_block,
			     SEARCH_VFT, NULL).symbol)
	    {
	      xfree (uptokstart);
	      return THIS;
	    }
	}
      break;
    default:
      break;
    }

  pascal_yylval.sval.ptr = tokstart;
  pascal_yylval.sval.length = namelen;

  if (*tokstart == '$')
    {
      xfree (uptokstart);
      return DOLLAR_VARIABLE;
    }

  /* Use token-type BLOCKNAME for symbols that happen to be defined as
     functions or symtabs.  If this is not so, then ...
     Use token-type TYPENAME for symbols that happen to be defined
     currently as names of types; NAME for other symbols.
     The caller is not constrained to care about the distinction.  */
  {
    std::string tmp = copy_name (pascal_yylval.sval);
    struct symbol *sym;
    struct field_of_this_result is_a_field_of_this;
    int is_a_field = 0;
    int hextype;

    is_a_field_of_this.type = NULL;
    if (search_field && current_type)
      is_a_field = (lookup_struct_elt_type (current_type,
					    tmp.c_str (), 1) != NULL);
    if (is_a_field)
      sym = NULL;
    else
      sym = lookup_symbol (tmp.c_str (), pstate->expression_context_block,
			   SEARCH_VFT, &is_a_field_of_this).symbol;
    /* second chance uppercased (as Free Pascal does).  */
    if (!sym && is_a_field_of_this.type == NULL && !is_a_field)
      {
       for (int i = 0; i <= namelen; i++)
	 {
	   if ((tmp[i] >= 'a' && tmp[i] <= 'z'))
	     tmp[i] -= ('a'-'A');
	 }
       if (search_field && current_type)
	 is_a_field = (lookup_struct_elt_type (current_type,
					       tmp.c_str (), 1) != NULL);
       if (is_a_field)
	 sym = NULL;
       else
	 sym = lookup_symbol (tmp.c_str (), pstate->expression_context_block,
			      SEARCH_VFT, &is_a_field_of_this).symbol;
      }
    /* Third chance Capitalized (as GPC does).  */
    if (!sym && is_a_field_of_this.type == NULL && !is_a_field)
      {
       for (int i = 0; i <= namelen; i++)
	 {
	   if (i == 0)
	     {
	      if ((tmp[i] >= 'a' && tmp[i] <= 'z'))
		tmp[i] -= ('a'-'A');
	     }
	   else
	   if ((tmp[i] >= 'A' && tmp[i] <= 'Z'))
	     tmp[i] -= ('A'-'a');
	  }
       if (search_field && current_type)
	 is_a_field = (lookup_struct_elt_type (current_type,
					       tmp.c_str (), 1) != NULL);
       if (is_a_field)
	 sym = NULL;
       else
	 sym = lookup_symbol (tmp.c_str (), pstate->expression_context_block,
			      SEARCH_VFT, &is_a_field_of_this).symbol;
      }

    if (is_a_field || (is_a_field_of_this.type != NULL))
      {
	tempbuf = (char *) xrealloc (tempbuf, namelen + 1);
	strncpy (tempbuf, tmp.c_str (), namelen);
	tempbuf [namelen] = 0;
	pascal_yylval.sval.ptr = tempbuf;
	pascal_yylval.sval.length = namelen;
	pascal_yylval.ssym.sym.symbol = NULL;
	pascal_yylval.ssym.sym.block = NULL;
	xfree (uptokstart);
	pascal_yylval.ssym.is_a_field_of_this = is_a_field_of_this.type != NULL;
	if (is_a_field)
	  return FIELDNAME;
	else
	  return NAME;
      }
    /* Call lookup_symtab, not lookup_partial_symtab, in case there are
       no psymtabs (coff, xcoff, or some future change to blow away the
       psymtabs once once symbols are read).  */
    if ((sym && sym->loc_class () == LOC_BLOCK)
	|| lookup_symtab (current_program_space, tmp.c_str ()))
      {
	pascal_yylval.ssym.sym.symbol = sym;
	pascal_yylval.ssym.sym.block = NULL;
	pascal_yylval.ssym.is_a_field_of_this = is_a_field_of_this.type != NULL;
	xfree (uptokstart);
	return BLOCKNAME;
      }
    if (sym && sym->loc_class () == LOC_TYPEDEF)
	{
#if 1
	  /* Despite the following flaw, we need to keep this code enabled.
	     Because we can get called from check_stub_method, if we don't
	     handle nested types then it screws many operations in any
	     program which uses nested types.  */
	  /* In "A::x", if x is a member function of A and there happens
	     to be a type (nested or not, since the stabs don't make that
	     distinction) named x, then this code incorrectly thinks we
	     are dealing with nested types rather than a member function.  */

	  const char *p;
	  const char *namestart;
	  struct symbol *best_sym;

	  /* Look ahead to detect nested types.  This probably should be
	     done in the grammar, but trying seemed to introduce a lot
	     of shift/reduce and reduce/reduce conflicts.  It's possible
	     that it could be done, though.  Or perhaps a non-grammar, but
	     less ad hoc, approach would work well.  */

	  /* Since we do not currently have any way of distinguishing
	     a nested type from a non-nested one (the stabs don't tell
	     us whether a type is nested), we just ignore the
	     containing type.  */

	  p = pstate->lexptr;
	  best_sym = sym;
	  while (1)
	    {
	      /* Skip whitespace.  */
	      p = skip_spaces (p);
	      if (*p == ':' && p[1] == ':')
		{
		  /* Skip the `::'.  */
		  p += 2;
		  /* Skip whitespace.  */
		  p = skip_spaces (p);
		  namestart = p;
		  while (*p == '_' || *p == '$' || (*p >= '0' && *p <= '9')
			 || (*p >= 'a' && *p <= 'z')
			 || (*p >= 'A' && *p <= 'Z'))
		    ++p;
		  if (p != namestart)
		    {
		      struct symbol *cur_sym;
		      /* As big as the whole rest of the expression, which is
			 at least big enough.  */
		      char *ncopy
			= (char *) alloca (tmp.size () + strlen (namestart)
					   + 3);
		      char *tmp1;

		      tmp1 = ncopy;
		      memcpy (tmp1, tmp.c_str (), tmp.size ());
		      tmp1 += tmp.size ();
		      memcpy (tmp1, "::", 2);
		      tmp1 += 2;
		      memcpy (tmp1, namestart, p - namestart);
		      tmp1[p - namestart] = '\0';
		      cur_sym
			= lookup_symbol (ncopy,
					 pstate->expression_context_block,
					 SEARCH_VFT, NULL).symbol;
		      if (cur_sym)
			{
			  if (cur_sym->loc_class () == LOC_TYPEDEF)
			    {
			      best_sym = cur_sym;
			      pstate->lexptr = p;
			    }
			  else
			    break;
			}
		      else
			break;
		    }
		  else
		    break;
		}
	      else
		break;
	    }

	  pascal_yylval.tsym.type = best_sym->type ();
#else /* not 0 */
	  pascal_yylval.tsym.type = sym->type ();
#endif /* not 0 */
	  xfree (uptokstart);
	  return TYPENAME;
	}
    pascal_yylval.tsym.type
      = language_lookup_primitive_type (pstate->language (),
					pstate->gdbarch (), tmp.c_str ());
    if (pascal_yylval.tsym.type != NULL)
      {
	xfree (uptokstart);
	return TYPENAME;
      }

    /* Input names that aren't symbols but ARE valid hex numbers,
       when the input radix permits them, can be names or numbers
       depending on the parse.  Note we support radixes > 16 here.  */
    if (!sym
	&& ((tokstart[0] >= 'a' && tokstart[0] < 'a' + input_radix - 10)
	    || (tokstart[0] >= 'A' && tokstart[0] < 'A' + input_radix - 10)))
      {
	p_exp_parser_YYSTYPE newlval;	/* Its value is ignored.  */
	hextype = parse_number (pstate, tokstart, namelen, 0, &newlval);
	if (hextype == INT)
	  {
	    pascal_yylval.ssym.sym.symbol = sym;
	    pascal_yylval.ssym.sym.block = NULL;
	    pascal_yylval.ssym.is_a_field_of_this = is_a_field_of_this.type != NULL;
	    xfree (uptokstart);
	    return NAME_OR_INT;
	  }
      }

    xfree(uptokstart);
    /* Any other kind of symbol.  */
    pascal_yylval.ssym.sym.symbol = sym;
    pascal_yylval.ssym.sym.block = NULL;
    return NAME;
  }
}

/* See p-exp-parser.h.  */

void
pascal_yyerror (const char *msg)
{
  pstate->parse_error (msg);
}

} /* namespace p_exp_parser */

/* See p-exp-parser.h.  */

int
pascal_parse (struct parser_state *par_state)
{
  using namespace p_exp_parser;

  /* Setting up the parser state.  */
  scoped_restore pstate_restore = make_scoped_restore (&pstate);
  gdb_assert (par_state != NULL);
  pstate = par_state;
  paren_depth = 0;

  scoped_restore restore_yydebug = make_scoped_restore (&pascal_yydebug,
							par_state->debug);

  int result = pascal_yyparse ();
  if (!result)
    pstate->set_operation (pstate->pop ());
  return result;
}
