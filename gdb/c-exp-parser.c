/* Support code for the C expression parser, for GDB.

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

#include "c-exp-parser.h"
#include "block.h"
#include "c-exp-parser-gen.h"
#include "c-support.h"
#include "charset.h"
#include "cp-support.h"
#include "macroexp.h"
#include "macroscope.h"
#include "objc-lang.h"

/* The entry point of the bison/yacc-generated parser, defined in
   c-exp-parser-gen.c.  Bison produces a declaration for c_yyparse in
   c-exp-parser-gen.h, but byacc does not, hence this declaration.  */

int c_yyparse ();

/* Likewise, byacc does not produce a declaration for c_yydebug.  */

extern int c_yydebug;

namespace c_exp_parser
{

/* See c-exp-parser.h.  */

c_parse_state *cpstate;

/* See c-exp-parser.h.  */

parser_state *pstate;

/* See c-exp-parser.h.  */

struct stoken
operator_stoken (const char *op)
{
  struct stoken st = { NULL, 0 };
  char *buf;

  st.length = CP_OPERATOR_LEN + strlen (op);
  buf = (char *) xmalloc (st.length + 1);
  strcpy (buf, CP_OPERATOR_STR);
  strcat (buf, op);
  st.ptr = buf;

  /* The toplevel (c_parse) will free the memory allocated here.  */
  cpstate->strings.emplace_back (buf);
  return st;
};

/* See c-exp-parser.h.  */

qualified_name_token
typename_stoken (const char *type)
{
  return qualified_name_token { nullptr, type, false };
};

/* See c-exp-parser.h.  */

int
type_aggregate_p (struct type *type)
{
  return (type->code () == TYPE_CODE_STRUCT
	  || type->code () == TYPE_CODE_UNION
	  || type->code () == TYPE_CODE_NAMESPACE
	  || (type->code () == TYPE_CODE_ENUM
	      && type->is_declared_class ()));
}

/* See c-exp-parser.h.  */

void
check_parameter_typelist (std::vector<struct type *> *params)
{
  struct type *type;
  int ix;

  for (ix = 0; ix < params->size (); ++ix)
    {
      type = (*params)[ix];
      if (type != NULL && check_typedef (type)->code () == TYPE_CODE_VOID)
	{
	  if (ix == 0)
	    {
	      if (params->size () == 1)
		{
		  /* Ok.  */
		  break;
		}
	      error (_("parameter types following 'void'"));
	    }
	  else
	    error (_("'void' invalid as parameter type"));
	}
    }
}

/* See c-exp-parser.h.  */

int
parse_number (struct parser_state *par_state, const char *buf, int len,
	      int parsed_float, c_exp_parser_YYSTYPE *putithere)
{
  ULONGEST n = 0;
  ULONGEST prevn = 0;

  int i = 0;
  int c;
  int base = input_radix;
  int unsigned_p = 0;

  /* Number of "L" suffixes encountered.  */
  int long_p = 0;

  /* Imaginary number.  */
  bool imaginary_p = false;

  /* We have found a "L" or "U" (or "i") suffix.  */
  int found_suffix = 0;

  if (parsed_float)
    {
      if (len >= 1 && buf[len - 1] == 'i')
	{
	  imaginary_p = true;
	  --len;
	}

      /* Handle suffixes for decimal floating-point: "df", "dd" or "dl".  */
      if (len >= 2 && buf[len - 2] == 'd' && buf[len - 1] == 'f')
	{
	  putithere->typed_val_float.type
	    = parse_type (par_state)->builtin_decfloat;
	  len -= 2;
	}
      else if (len >= 2 && buf[len - 2] == 'd' && buf[len - 1] == 'd')
	{
	  putithere->typed_val_float.type
	    = parse_type (par_state)->builtin_decdouble;
	  len -= 2;
	}
      else if (len >= 2 && buf[len - 2] == 'd' && buf[len - 1] == 'l')
	{
	  putithere->typed_val_float.type
	    = parse_type (par_state)->builtin_declong;
	  len -= 2;
	}
      /* Handle suffixes: 'f' for float, 'l' for long double.  */
      else if (len >= 1 && c_tolower (buf[len - 1]) == 'f')
	{
	  putithere->typed_val_float.type
	    = parse_type (par_state)->builtin_float;
	  len -= 1;
	}
      else if (len >= 1 && c_tolower (buf[len - 1]) == 'l')
	{
	  putithere->typed_val_float.type
	    = parse_type (par_state)->builtin_long_double;
	  len -= 1;
	}
      /* Default type for floating-point literals is double.  */
      else
	{
	  putithere->typed_val_float.type
	    = parse_type (par_state)->builtin_double;
	}

      if (!parse_float (buf, len,
			putithere->typed_val_float.type,
			putithere->typed_val_float.val))
	return ERROR;

      if (imaginary_p)
	putithere->typed_val_float.type
	  = init_complex_type (nullptr, putithere->typed_val_float.type);

      return imaginary_p ? COMPLEX_FLOAT : FLOAT;
    }

  /* Handle base-switching prefixes 0x, 0t, 0d, 0 */
  if (buf[0] == '0' && len > 1)
    switch (buf[1])
      {
      case 'x':
      case 'X':
	if (len >= 3)
	  {
	    buf += 2;
	    base = 16;
	    len -= 2;
	  }
	break;

      case 'b':
      case 'B':
	if (len >= 3)
	  {
	    buf += 2;
	    base = 2;
	    len -= 2;
	  }
	break;

      case 't':
      case 'T':
      case 'd':
      case 'D':
	if (len >= 3)
	  {
	    buf += 2;
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
      c = *buf++;
      if (c >= 'A' && c <= 'Z')
	c += 'a' - 'A';
      if (c != 'l' && c != 'u' && c != 'i')
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
	  else if (c == 'i')
	    {
	      imaginary_p = true;
	      found_suffix = 1;
	    }
	  else
	    return ERROR;	/* Char not a digit */
	}
      if (i >= base)
	return ERROR;		/* Invalid digit in this base */

      if (c != 'l' && c != 'u' && c != 'i')
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
  bool have_signed
    /* No 'u' suffix.  */
    = !unsigned_p;
  bool have_unsigned
    = ((/* 'u' suffix.  */
	unsigned_p)
       || (/* Not a decimal.  */
	   base != 10)
       || (/* Allowed as a convenience, in case decimal doesn't fit in largest
	      signed type.  */
	   !fits_in_type (1, n, long_long_bits, true)));
  bool have_int
    /* No 'l' or 'll' suffix.  */
    = long_p == 0;
  bool have_long
    /* No 'll' suffix.  */
    = long_p <= 1;
  if (have_int && have_signed && fits_in_type (1, n, int_bits, true))
    putithere->typed_val_int.type = parse_type (par_state)->builtin_int;
  else if (have_int && have_unsigned && fits_in_type (1, n, int_bits, false))
    putithere->typed_val_int.type
      = parse_type (par_state)->builtin_unsigned_int;
  else if (have_long && have_signed && fits_in_type (1, n, long_bits, true))
    putithere->typed_val_int.type = parse_type (par_state)->builtin_long;
  else if (have_long && have_unsigned && fits_in_type (1, n, long_bits, false))
    putithere->typed_val_int.type
      = parse_type (par_state)->builtin_unsigned_long;
  else if (have_signed && fits_in_type (1, n, long_long_bits, true))
    putithere->typed_val_int.type
      = parse_type (par_state)->builtin_long_long;
  else if (have_unsigned && fits_in_type (1, n, long_long_bits, false))
    putithere->typed_val_int.type
      = parse_type (par_state)->builtin_unsigned_long_long;
  else
    error (_("Numeric constant too large."));
  putithere->typed_val_int.val = n;

   if (imaginary_p)
     putithere->typed_val_int.type
       = init_complex_type (nullptr, putithere->typed_val_int.type);

   return imaginary_p ? COMPLEX_INT : INT;
}

/* Temporary obstack used for holding strings.  */
static struct obstack tempbuf;
static int tempbuf_init;

/* Parse a string or character literal from TOKPTR.  The string or
   character may be wide or unicode.  *OUTPTR is set to just after the
   end of the literal in the input string.  The resulting token is
   stored in VALUE.  This returns a token value, either STRING or
   CHAR, depending on what was parsed.  *HOST_CHARS is set to the
   number of host characters in the literal.  */

static int
parse_string_or_char (const char *tokptr, const char **outptr,
		      struct typed_stoken *value, int *host_chars)
{
  int quote;
  c_string_type type;
  int is_objc = 0;

  /* Build the gdb internal form of the input string in tempbuf.  Note
     that the buffer is null byte terminated *only* for the
     convenience of debugging gdb itself and printing the buffer
     contents when the buffer contains no embedded nulls.  Gdb does
     not depend upon the buffer being null byte terminated, it uses
     the length string instead.  This allows gdb to handle C strings
     (as well as strings in other languages) with embedded null
     bytes */

  if (!tempbuf_init)
    tempbuf_init = 1;
  else
    obstack_free (&tempbuf, NULL);
  obstack_init (&tempbuf);

  /* Record the string type.  */
  if (*tokptr == 'L')
    {
      type = C_WIDE_STRING;
      ++tokptr;
    }
  else if (*tokptr == 'u')
    {
      type = C_STRING_16;
      ++tokptr;
    }
  else if (*tokptr == 'U')
    {
      type = C_STRING_32;
      ++tokptr;
    }
  else if (*tokptr == '@')
    {
      /* An Objective C string.  */
      is_objc = 1;
      type = C_STRING;
      ++tokptr;
    }
  else
    type = C_STRING;

  /* Skip the quote.  */
  quote = *tokptr;
  if (quote == '\'')
    type |= C_CHAR;
  ++tokptr;

  *host_chars = 0;

  while (*tokptr)
    {
      char c = *tokptr;
      if (c == '\\')
	{
	  ++tokptr;
	  *host_chars += c_parse_escape (&tokptr, &tempbuf);
	}
      else if (c == quote)
	break;
      else
	{
	  obstack_1grow (&tempbuf, c);
	  ++tokptr;
	  /* FIXME: this does the wrong thing with multi-byte host
	     characters.  We could use mbrlen here, but that would
	     make "set host-charset" a bit less useful.  */
	  ++*host_chars;
	}
    }

  if (*tokptr != quote)
    {
      if (quote == '"')
	error (_("Unterminated string in expression."));
      else
	error (_("Unmatched single quote."));
    }
  ++tokptr;

  value->type = type;
  value->ptr = (char *) obstack_base (&tempbuf);
  value->length = obstack_object_size (&tempbuf);

  *outptr = tokptr;

  return quote == '"' ? (is_objc ? NSSTRING : STRING) : CHAR;
}

/* This is used to associate some attributes with a token.  */

enum token_flag
{
  /* If this bit is set, the token is C++-only.  */

  FLAG_CXX = 1,

  /* If this bit is set, the token is C-only.  */

  FLAG_C = 2,

  /* If this bit is set, the token is conditional: if there is a
     symbol of the same name, then the token is a symbol; otherwise,
     the token is a keyword.  */

  FLAG_SHADOW = 4
};
DEF_ENUM_FLAGS_TYPE (enum token_flag, token_flags);

struct c_token
{
  const char *oper;
  int token;
  enum exp_opcode opcode;
  token_flags flags;
};

static const struct c_token tokentab3[] =
  {
    {">>=", ASSIGN_MODIFY, BINOP_RSH, 0},
    {"<<=", ASSIGN_MODIFY, BINOP_LSH, 0},
    {"->*", ARROW_STAR, OP_NULL, FLAG_CXX},
    {"...", DOTDOTDOT, OP_NULL, 0}
  };

static const struct c_token tokentab2[] =
  {
    {"+=", ASSIGN_MODIFY, BINOP_ADD, 0},
    {"-=", ASSIGN_MODIFY, BINOP_SUB, 0},
    {"*=", ASSIGN_MODIFY, BINOP_MUL, 0},
    {"/=", ASSIGN_MODIFY, BINOP_DIV, 0},
    {"%=", ASSIGN_MODIFY, BINOP_REM, 0},
    {"|=", ASSIGN_MODIFY, BINOP_BITWISE_IOR, 0},
    {"&=", ASSIGN_MODIFY, BINOP_BITWISE_AND, 0},
    {"^=", ASSIGN_MODIFY, BINOP_BITWISE_XOR, 0},
    {"++", INCREMENT, OP_NULL, 0},
    {"--", DECREMENT, OP_NULL, 0},
    {"->", ARROW, OP_NULL, 0},
    {"&&", ANDAND, OP_NULL, 0},
    {"||", OROR, OP_NULL, 0},
    /* "::" is *not* only C++: gdb overrides its meaning in several
       different ways, e.g., 'filename'::func, function::variable.  */
    {"::", COLONCOLON, OP_NULL, 0},
    {"<<", LSH, OP_NULL, 0},
    {">>", RSH, OP_NULL, 0},
    {"==", EQUAL, OP_NULL, 0},
    {"!=", NOTEQUAL, OP_NULL, 0},
    {"<=", LEQ, OP_NULL, 0},
    {">=", GEQ, OP_NULL, 0},
    {".*", DOT_STAR, OP_NULL, FLAG_CXX}
  };

/* Identifier-like tokens.  Only type-specifiers than can appear in
   multi-word type names (for example 'double' can appear in 'long
   double') need to be listed here.  type-specifiers that are only ever
   single word (like 'char') are handled by the classify_name function.  */
static const struct c_token ident_tokens[] =
  {
    {"unsigned", UNSIGNED, OP_NULL, 0},
    {"template", TEMPLATE, OP_NULL, FLAG_CXX},
    {"volatile", VOLATILE_KEYWORD, OP_NULL, 0},
    {"struct", STRUCT, OP_NULL, 0},
    {"signed", SIGNED_KEYWORD, OP_NULL, 0},
    {"sizeof", SIZEOF, OP_NULL, 0},
    {"_Alignof", ALIGNOF, OP_NULL, 0},
    {"alignof", ALIGNOF, OP_NULL, FLAG_CXX},
    {"double", DOUBLE_KEYWORD, OP_NULL, 0},
    {"float", FLOAT_KEYWORD, OP_NULL, 0},
    {"false", FALSEKEYWORD, OP_NULL, FLAG_CXX},
    {"class", CLASS, OP_NULL, FLAG_CXX},
    {"union", UNION, OP_NULL, 0},
    {"short", SHORT, OP_NULL, 0},
    {"const", CONST_KEYWORD, OP_NULL, 0},
    {"restrict", RESTRICT, OP_NULL, FLAG_C | FLAG_SHADOW},
    {"__restrict__", RESTRICT, OP_NULL, 0},
    {"__restrict", RESTRICT, OP_NULL, 0},
    {"_Atomic", ATOMIC, OP_NULL, 0},
    {"enum", ENUM, OP_NULL, 0},
    {"long", LONG, OP_NULL, 0},
    {"_Complex", COMPLEX, OP_NULL, 0},
    {"__complex__", COMPLEX, OP_NULL, 0},

    {"true", TRUEKEYWORD, OP_NULL, FLAG_CXX},
    {"int", INT_KEYWORD, OP_NULL, 0},
    {"new", NEW, OP_NULL, FLAG_CXX},
    {"delete", DELETE, OP_NULL, FLAG_CXX},
    {"operator", OPERATOR, OP_NULL, FLAG_CXX},

    {"and", ANDAND, OP_NULL, FLAG_CXX},
    {"and_eq", ASSIGN_MODIFY, BINOP_BITWISE_AND, FLAG_CXX},
    {"bitand", '&', OP_NULL, FLAG_CXX},
    {"bitor", '|', OP_NULL, FLAG_CXX},
    {"compl", '~', OP_NULL, FLAG_CXX},
    {"not", '!', OP_NULL, FLAG_CXX},
    {"not_eq", NOTEQUAL, OP_NULL, FLAG_CXX},
    {"or", OROR, OP_NULL, FLAG_CXX},
    {"or_eq", ASSIGN_MODIFY, BINOP_BITWISE_IOR, FLAG_CXX},
    {"xor", '^', OP_NULL, FLAG_CXX},
    {"xor_eq", ASSIGN_MODIFY, BINOP_BITWISE_XOR, FLAG_CXX},

    {"const_cast", CONST_CAST, OP_NULL, FLAG_CXX },
    {"dynamic_cast", DYNAMIC_CAST, OP_NULL, FLAG_CXX },
    {"static_cast", STATIC_CAST, OP_NULL, FLAG_CXX },
    {"reinterpret_cast", REINTERPRET_CAST, OP_NULL, FLAG_CXX },

    {"__typeof__", TYPEOF, OP_TYPEOF, 0 },
    {"__typeof", TYPEOF, OP_TYPEOF, 0 },
    {"typeof", TYPEOF, OP_TYPEOF, FLAG_SHADOW },
    {"__decltype", DECLTYPE, OP_DECLTYPE, FLAG_CXX },
    {"decltype", DECLTYPE, OP_DECLTYPE, FLAG_CXX | FLAG_SHADOW },

    {"typeid", TYPEID, OP_TYPEID, FLAG_CXX}
  };


static void
scan_macro_expansion (const char *expansion)
{
  /* We'd better not be trying to push the stack twice.  */
  gdb_assert (! cpstate->macro_original_text);

  /* Copy to the obstack.  */
  const char *copy = obstack_strdup (&cpstate->expansion_obstack, expansion);

  /* Save the old lexptr value, so we can return to it when we're done
     parsing the expanded text.  */
  cpstate->macro_original_text = pstate->lexptr;
  pstate->lexptr = copy;
}

static int
scanning_macro_expansion (void)
{
  return cpstate->macro_original_text != 0;
}

static void
finished_macro_expansion (void)
{
  /* There'd better be something to pop back to.  */
  gdb_assert (cpstate->macro_original_text);

  /* Pop back to the original text.  */
  pstate->lexptr = cpstate->macro_original_text;
  cpstate->macro_original_text = 0;
}

/* Return true iff the token represents a C++ cast operator.  */

static int
is_cast_operator (const char *token, int len)
{
  return (! strncmp (token, "dynamic_cast", len)
	  || ! strncmp (token, "static_cast", len)
	  || ! strncmp (token, "reinterpret_cast", len)
	  || ! strncmp (token, "const_cast", len));
}

/* The scope used for macro expansion.  */
static struct macro_scope *expression_macro_scope;

/* This is set if a NAME token appeared at the very end of the input
   string, with no whitespace separating the name from the EOF.  This
   is used only when parsing to do field name completion.  */
static int saw_name_at_eof;

/* This is set if the previously-returned token was a structure
   operator -- either '.' or ARROW.  */
static bool last_was_structop;

/* Depth of parentheses.  */
static int paren_depth;

/* Lex an Objective-C @selector.  Return true if lexed.  In this case,
   sets the resulting token and updates the lex pointer.  Otherwise
   returns false and updates nothing.  */

static bool
lex_selector (const char **lex_ptr, struct stoken *token)
{
  const char *p = *lex_ptr;

  if (!startswith (p, "selector"))
    return false;

  p += strlen ("selector");
  p = skip_spaces (p);
  if (*p != '(')
    return false;
  ++p;

  /* The selector name matches [A-Za-z0-9:_-]+.  We could probably be
     a bit more refined but meh.  */
  const char *start = p;
  while (c_isalnum (*p) || *p == ':' || *p == '_' || *p == '-')
    ++p;
  if (p == start)
    return false;
  const char *end = p;

  p = skip_spaces (p);
  if (*p != ')')
    return false;
  ++p;

  *lex_ptr = p;
  *token = { start, (int) (end - start) };
  return true;
}

/* Read one token, getting characters through lexptr.  */

static int
lex_one_token (struct parser_state *par_state, bool *is_quoted_name)
{
  int c;
  int namelen;
  const char *tokstart;
  bool saw_structop = last_was_structop;

  last_was_structop = false;
  *is_quoted_name = false;

 retry:

  /* Check if this is a macro invocation that we need to expand.  */
  if (! scanning_macro_expansion ())
    {
      gdb::unique_xmalloc_ptr<char> expanded
	= macro_expand_next (&pstate->lexptr, *expression_macro_scope);

      if (expanded != nullptr)
	scan_macro_expansion (expanded.get ());
    }

  pstate->prev_lexptr = pstate->lexptr;

  tokstart = pstate->lexptr;
  /* See if it is a special token of length 3.  */
  for (const auto &token : tokentab3)
    if (strncmp (tokstart, token.oper, 3) == 0)
      {
	if ((token.flags & FLAG_CXX) != 0
	    && par_state->language ()->la_language != language_cplus)
	  break;
	gdb_assert ((token.flags & FLAG_C) == 0);

	pstate->lexptr += 3;
	c_yylval.opcode = token.opcode;
	return token.token;
      }

  /* See if it is a special token of length 2.  */
  for (const auto &token : tokentab2)
    if (strncmp (tokstart, token.oper, 2) == 0)
      {
	if ((token.flags & FLAG_CXX) != 0
	    && par_state->language ()->la_language != language_cplus)
	  break;
	gdb_assert ((token.flags & FLAG_C) == 0);

	pstate->lexptr += 2;
	c_yylval.opcode = token.opcode;
	if (token.token == ARROW)
	  last_was_structop = 1;
	return token.token;
      }

  switch (c = *tokstart)
    {
    case 0:
      /* If we were just scanning the result of a macro expansion,
	 then we need to resume scanning the original text.
	 If we're parsing for field name completion, and the previous
	 token allows such completion, return a COMPLETE token.
	 Otherwise, we were already scanning the original text, and
	 we're really done.  */
      if (scanning_macro_expansion ())
	{
	  finished_macro_expansion ();
	  goto retry;
	}
      else if (saw_name_at_eof)
	{
	  saw_name_at_eof = 0;
	  return COMPLETE;
	}
      else if (par_state->parse_completion && saw_structop)
	return COMPLETE;
      else
	return 0;

    case ' ':
    case '\t':
    case '\n':
      pstate->lexptr++;
      goto retry;

    case '[':
    case '(':
      paren_depth++;
      pstate->lexptr++;
      if (par_state->language ()->la_language == language_objc
	  && c == '[')
	return OBJC_LBRAC;
      return c;

    case ']':
    case ')':
      if (paren_depth == 0)
	return 0;
      paren_depth--;
      pstate->lexptr++;
      return c;

    case ',':
      if (pstate->comma_terminates
	  && paren_depth == 0
	  && ! scanning_macro_expansion ())
	return 0;
      pstate->lexptr++;
      return c;

    case '.':
      /* Might be a floating point number.  */
      if (pstate->lexptr[1] < '0' || pstate->lexptr[1] > '9')
	{
	  last_was_structop = true;
	  goto symbol;		/* Nope, must be a symbol. */
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
	int got_dot = 0, got_e = 0, got_p = 0, toktype;
	const char *p = tokstart;
	int hex = input_radix > 10;

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
	    if (!hex && !got_e && !got_p && (*p == 'e' || *p == 'E'))
	      got_dot = got_e = 1;
	    else if (!got_e && !got_p && (*p == 'p' || *p == 'P'))
	      got_dot = got_p = 1;
	    /* This test does not include !hex, because a '.' always indicates
	       a decimal floating point number regardless of the radix.  */
	    else if (!got_dot && *p == '.')
	      got_dot = 1;
	    else if (((got_e && (p[-1] == 'e' || p[-1] == 'E'))
		      || (got_p && (p[-1] == 'p' || p[-1] == 'P')))
		     && (*p == '-' || *p == '+'))
	      {
		/* This is the sign of the exponent, not the end of
		   the number.  */
	      }
	    else if (*p == '\'')
	      {
		if (!no_tick.has_value ())
		  no_tick.emplace (tokstart, p);
		continue;
	      }
	    /* We will take any letters or digits.  parse_number will
	       complain if past the radix, or if L or U are not final.  */
	    else if ((*p < '0' || *p > '9')
		     && ((*p < 'a' || *p > 'z')
				  && (*p < 'A' || *p > 'Z')))
	      break;
	    if (no_tick.has_value ())
	      no_tick->push_back (*p);
	  }
	if (no_tick.has_value ())
	  toktype = parse_number (par_state, no_tick->c_str (),
				  no_tick->length (),
				  got_dot | got_e | got_p, &c_yylval);
	else
	  toktype = parse_number (par_state, tokstart, p - tokstart,
				  got_dot | got_e | got_p, &c_yylval);
	if (toktype == ERROR)
	  error (_("Invalid number \"%.*s\"."), (int) (p - tokstart),
		 tokstart);
	pstate->lexptr = p;
	return toktype;
      }

    case '@':
      {
	const char *p = &tokstart[1];

	if (par_state->language ()->la_language == language_objc)
	  {
	    struct stoken sel_token;
	    if (lex_selector (&p, &sel_token))
	      {
		pstate->lexptr = p;
		c_yylval.sval = sel_token;
		return SELECTOR;
	      }
	    else if (*p == '"')
	      goto parse_string;
	  }

	while (c_isspace (*p))
	  p++;
	size_t len = strlen ("entry");
	if (strncmp (p, "entry", len) == 0 && !c_ident_is_alnum (p[len])
	    && p[len] != '_')
	  {
	    pstate->lexptr = &p[len];
	    return ENTRY;
	  }
      }
      [[fallthrough]];
    case '+':
    case '-':
    case '*':
    case '/':
    case '%':
    case '|':
    case '&':
    case '^':
    case '~':
    case '!':
    case '<':
    case '>':
    case '?':
    case ':':
    case '=':
    case '{':
    case '}':
    symbol:
      pstate->lexptr++;
      return c;

    case 'L':
    case 'u':
    case 'U':
      if (tokstart[1] != '"' && tokstart[1] != '\'')
	break;
      [[fallthrough]];
    case '\'':
    case '"':

    parse_string:
      {
	int host_len;
	int result = parse_string_or_char (tokstart, &pstate->lexptr,
					   &c_yylval.tsval, &host_len);
	if (result == CHAR)
	  {
	    if (host_len == 0)
	      error (_("Empty character constant."));
	    else if (host_len > 2 && c == '\'')
	      {
		++tokstart;
		namelen = pstate->lexptr - tokstart - 1;
		*is_quoted_name = true;

		goto tryname;
	      }
	    else if (host_len > 1)
	      error (_("Invalid character constant."));
	  }
	return result;
      }
    }

  if (!(c == '_' || c == '$' || c_ident_is_alpha (c)))
    /* We must have come across a bad character (e.g. ';').  */
    error (_("Invalid character '%c' in expression."), c);

  /* It's a name.  See how long it is.  */
  namelen = 0;
  for (c = tokstart[namelen];
       (c == '_' || c == '$' || c_ident_is_alnum (c) || c == '<');)
    {
      /* Template parameter lists are part of the name.
	 FIXME: This mishandles `print $a<4&&$a>3'.  */

      if (c == '<')
	{
	  if (! is_cast_operator (tokstart, namelen))
	    {
	      /* Scan ahead to get rest of the template specification.  Note
		 that we look ahead only when the '<' adjoins non-whitespace
		 characters; for comparison expressions, e.g. "a < b > c",
		 there must be spaces before the '<', etc. */
	      const char *p = find_template_name_end (tokstart + namelen);

	      if (p)
		namelen = p - tokstart;
	    }
	  break;
	}
      c = tokstart[++namelen];
    }

  /* The token "if" terminates the expression and is NOT removed from
     the input stream.  It doesn't count if it appears in the
     expansion of a macro.  */
  if (namelen == 2
      && tokstart[0] == 'i'
      && tokstart[1] == 'f'
      && ! scanning_macro_expansion ())
    {
      return 0;
    }

  /* For the same reason (breakpoint conditions), "thread N"
     terminates the expression.  "thread" could be an identifier, but
     an identifier is never followed by a number without intervening
     punctuation.  "task" is similar.  Handle abbreviations of these,
     similarly to breakpoint.c:find_condition_and_thread.  */
  if (namelen >= 1
      && (strncmp (tokstart, "thread", namelen) == 0
	  || strncmp (tokstart, "task", namelen) == 0)
      && (tokstart[namelen] == ' ' || tokstart[namelen] == '\t')
      && ! scanning_macro_expansion ())
    {
      const char *p = skip_spaces (tokstart + namelen + 1);
      if (*p >= '0' && *p <= '9')
	return 0;
    }

  pstate->lexptr += namelen;

  tryname:

  c_yylval.sval.ptr = tokstart;
  c_yylval.sval.length = namelen;

  /* Catch specific keywords.  */
  std::string copy = copy_name (c_yylval.sval);
  for (const auto &token : ident_tokens)
    if (copy == token.oper)
      {
	if ((token.flags & FLAG_CXX) != 0
	    && par_state->language ()->la_language != language_cplus)
	  break;
	if ((token.flags & FLAG_C) != 0
	    && par_state->language ()->la_language != language_c
	    && par_state->language ()->la_language != language_objc)
	  break;

	if ((token.flags & FLAG_SHADOW) != 0)
	  {
	    struct field_of_this_result is_a_field_of_this;

	    if (lookup_symbol (copy.c_str (),
			       pstate->expression_context_block,
			       SEARCH_VFT, &is_a_field_of_this).symbol
		!= NULL)
	      {
		/* The keyword is shadowed.  */
		break;
	      }
	  }

	/* It is ok to always set this, even though we don't always
	   strictly need to.  */
	c_yylval.opcode = token.opcode;
	return token.token;
      }

  if (*tokstart == '$')
    return DOLLAR_VARIABLE;

  if (pstate->parse_completion && *pstate->lexptr == '\0')
    saw_name_at_eof = 1;

  c_yylval.ssym.stoken = c_yylval.sval;
  c_yylval.ssym.sym.symbol = NULL;
  c_yylval.ssym.sym.block = NULL;
  c_yylval.ssym.is_a_field_of_this = 0;
  return NAME;
}

/* An object of this type is pushed on a FIFO by the "outer" lexer.  */
struct c_token_and_value
{
  int token;
  c_exp_parser_YYSTYPE value;
};

/* A FIFO of tokens that have been read but not yet returned to the
   parser.  */
static std::vector<c_token_and_value> token_fifo;

/* Non-zero if the lexer should return tokens from the FIFO.  */
static int popping;

/* Temporary storage for c_lex; this holds symbol names as they are
   built up.  */
static auto_obstack name_obstack;

/* Classify a NAME token.  The contents of the token are in `yylval'.
   Updates yylval and returns the new token type.  BLOCK is the block
   in which lookups start; this can be NULL to mean the global scope.
   IS_QUOTED_NAME is non-zero if the name token was originally quoted
   in single quotes.  IS_AFTER_STRUCTOP is true if this name follows
   a structure operator -- either '.' or ARROW  */

static int
classify_name (struct parser_state *par_state, const struct block *block,
	       bool is_quoted_name, bool is_after_structop)
{
  struct block_symbol bsym;
  struct field_of_this_result is_a_field_of_this;

  std::string copy = copy_name (c_yylval.sval);

  bsym = lookup_symbol (copy.c_str (), block, SEARCH_VFT,
			&is_a_field_of_this);

  if (bsym.symbol && bsym.symbol->loc_class () == LOC_BLOCK)
    {
      c_yylval.ssym.sym = bsym;
      c_yylval.ssym.is_a_field_of_this = is_a_field_of_this.type != NULL;
      return BLOCKNAME;
    }
  else if (!bsym.symbol)
    {
      /* If we found a field of 'this', we might have erroneously
	 found a constructor where we wanted a type name.  Handle this
	 case by noticing that we found a constructor and then look up
	 the type tag instead.  */
      if (is_a_field_of_this.type != NULL
	  && is_a_field_of_this.fn_field != NULL
	  && TYPE_FN_FIELD_CONSTRUCTOR (is_a_field_of_this.fn_field->fn_fields,
					0))
	{
	  struct field_of_this_result inner_is_a_field_of_this;

	  bsym = lookup_symbol (copy.c_str (), block, SEARCH_STRUCT_DOMAIN,
				&inner_is_a_field_of_this);
	  if (bsym.symbol != NULL)
	    {
	      c_yylval.tsym.type = bsym.symbol->type ();
	      return TYPENAME;
	    }
	}

      /* If we found a field on the "this" object, or we are looking
	 up a field on a struct, then we want to prefer it over a
	 filename.  However, if the name was quoted, then it is better
	 to check for a filename or a block, since this is the only
	 way the user has of requiring the extension to be used.  */
      if ((is_a_field_of_this.type == NULL && !is_after_structop)
	  || is_quoted_name)
	{
	  /* See if it's a file name. */
	  if (auto symtab = lookup_symtab (current_program_space, copy.c_str ());
	      symtab != nullptr)
	    {
	      c_yylval.bval
		= symtab->compunit ().blockvector ()->static_block ();

	      return FILENAME;
	    }
	}
    }

  if (bsym.symbol && bsym.symbol->loc_class () == LOC_TYPEDEF)
    {
      c_yylval.tsym.type = bsym.symbol->type ();
      return TYPENAME;
    }

  /* See if it's an ObjC classname.  */
  if (par_state->language ()->la_language == language_objc && !bsym.symbol)
    {
      CORE_ADDR Class = lookup_objc_class (par_state->gdbarch (),
					   copy.c_str ());
      if (Class)
	{
	  struct symbol *sym;

	  c_yylval.theclass.theclass = Class;
	  sym = lookup_struct_noerr (copy.c_str (),
				     par_state->expression_context_block);
	  if (sym)
	    c_yylval.theclass.type = sym->type ();
	  return CLASSNAME;
	}
    }

  /* Input names that aren't symbols but ARE valid hex numbers, when
     the input radix permits them, can be names or numbers depending
     on the parse.  Note we support radixes > 16 here.  */
  if (!bsym.symbol
      && ((copy[0] >= 'a' && copy[0] < 'a' + input_radix - 10)
	  || (copy[0] >= 'A' && copy[0] < 'A' + input_radix - 10)))
    {
      c_exp_parser_YYSTYPE newlval;	/* Its value is ignored.  */
      int hextype = parse_number (par_state, copy.c_str (), c_yylval.sval.length,
				  0, &newlval);

      if (hextype == INT)
	{
	  c_yylval.ssym.sym = bsym;
	  c_yylval.ssym.is_a_field_of_this = is_a_field_of_this.type != NULL;
	  return NAME_OR_INT;
	}
    }

  /* Any other kind of symbol */
  c_yylval.ssym.sym = bsym;
  c_yylval.ssym.is_a_field_of_this = is_a_field_of_this.type != NULL;

  if (bsym.symbol == NULL
      && par_state->language ()->la_language == language_cplus
      && is_a_field_of_this.type == NULL
      && lookup_minimal_symbol (current_program_space, copy.c_str ()).minsym == nullptr)
    return UNKNOWN_CPP_NAME;

  return NAME;
}

/* Like classify_name, but used by the inner loop of the lexer, when a
   name might have already been seen.  CONTEXT is the context type, or
   NULL if this is the first component of a name.  */

static int
classify_inner_name (struct parser_state *par_state,
		     const struct block *block, struct type *context)
{
  struct type *type;

  if (context == NULL)
    return classify_name (par_state, block, false, false);

  type = check_typedef (context);
  if (!type_aggregate_p (type))
    return ERROR;

  std::string copy = copy_name (c_yylval.ssym.stoken);
  /* N.B. We assume the symbol can only be in VAR_DOMAIN.  */
  c_yylval.ssym.sym = cp_lookup_nested_symbol (type, copy.c_str (), block,
					     SEARCH_VFT);

  /* If no symbol was found, search for a matching base class named
     COPY.  This will allow users to enter qualified names of class members
     relative to the `this' pointer.  */
  if (c_yylval.ssym.sym.symbol == NULL)
    {
      struct type *base_type = cp_find_type_baseclass_by_name (type,
							       copy.c_str ());

      if (base_type != NULL)
	{
	  c_yylval.tsym.type = base_type;
	  return TYPENAME;
	}

      return ERROR;
    }

  switch (c_yylval.ssym.sym.symbol->loc_class ())
    {
    case LOC_BLOCK:
    case LOC_LABEL:
      /* cp_lookup_nested_symbol might have accidentally found a constructor
	 named COPY when we really wanted a base class of the same name.
	 Double-check this case by looking for a base class.  */
      {
	struct type *base_type
	  = cp_find_type_baseclass_by_name (type, copy.c_str ());

	if (base_type != NULL)
	  {
	    c_yylval.tsym.type = base_type;
	    return TYPENAME;
	  }
      }
      return ERROR;

    case LOC_TYPEDEF:
      c_yylval.tsym.type = c_yylval.ssym.sym.symbol->type ();
      return TYPENAME;

    default:
      return NAME;
    }
  internal_error (_("not reached"));
}

/* See c-exp-parser.h.  */

void
handle_qualified_field_name (qualified_name_token token)
{
  struct type *type = nullptr;
  std::string accum;
  for (const auto name : split_name (token.prefix, split_style::CXX))
    {
      std::string current (name);

      if (accum.empty ())
	accum = name;
      else
	accum = accum + "::" + current;

      c_yylval.ssym.stoken.ptr = current.c_str ();
      c_yylval.ssym.stoken.length = current.size ();
      c_yylval.ssym.sym = {};
      c_yylval.ssym.is_a_field_of_this = 0;

      int kind = classify_inner_name (pstate,
				      pstate->expression_context_block,
				      type);
      if (kind != TYPENAME)
	error (_("could not find type '%s'"), accum.c_str ());

      type = c_yylval.tsym.type;
    }

  type = check_typedef (type);
  if (!type_aggregate_p (type))
    error (_("`%s' is not defined as an aggregate type."),
	   type->safe_name ());
  if (token.name[0] == '~')
    destructor_name_p (token.name, type);
  pstate->push_new<expr::scope_operation> (type, token.name);
}

/* See c-exp-parser.h.  */

int
c_yylex ()
{
  c_token_and_value current;
  int first_was_coloncolon, last_was_coloncolon;
  struct type *context_type = NULL;
  int last_to_examine, next_to_examine, checkpoint;
  const struct block *search_block;
  bool is_quoted_name, last_lex_was_structop;

  if (popping && !token_fifo.empty ())
    goto do_pop;
  popping = 0;

  last_lex_was_structop = last_was_structop;

  /* Read the first token and decide what to do.  Most of the
     subsequent code is C++-only; but also depends on seeing a "::" or
     name-like token.  */
  current.token = lex_one_token (pstate, &is_quoted_name);
  if (cpstate->assume_classification == TYPE_CODE_UNDEF
      && current.token == NAME)
    current.token = classify_name (pstate, pstate->expression_context_block,
				   is_quoted_name, last_lex_was_structop);
  if (pstate->language ()->la_language != language_cplus
      || (current.token != TYPENAME && current.token != COLONCOLON
	  && current.token != FILENAME
	  && (cpstate->assume_classification == TYPE_CODE_UNDEF
	      || current.token != NAME))
      || cpstate->assume_classification == TYPE_CODE_VOID)
    return current.token;

  /* Read any sequence of alternating "::" and name-like tokens into
     the token FIFO.  */
  current.value = c_yylval;
  token_fifo.push_back (current);
  last_was_coloncolon = current.token == COLONCOLON;
  while (1)
    {
      bool ignore;

      /* We ignore quoted names other than the very first one.
	 Subsequent ones do not have any special meaning.  */
      current.token = lex_one_token (pstate, &ignore);
      current.value = c_yylval;
      token_fifo.push_back (current);

      if ((last_was_coloncolon && current.token != NAME)
	  || (!last_was_coloncolon && current.token != COLONCOLON))
	break;
      last_was_coloncolon = !last_was_coloncolon;
    }
  popping = 1;

  /* We always read one extra token, so compute the number of tokens
     to examine accordingly.  */
  last_to_examine = token_fifo.size () - 2;
  next_to_examine = 0;

  current = token_fifo[next_to_examine];
  ++next_to_examine;

  name_obstack.clear ();
  checkpoint = 0;
  if (current.token == FILENAME)
    search_block = current.value.bval;
  else if (current.token == COLONCOLON)
    search_block = NULL;
  else
    {
      gdb_assert (current.token == TYPENAME
		  || cpstate->assume_classification != TYPE_CODE_UNDEF);
      search_block = pstate->expression_context_block;
      obstack_grow (&name_obstack, current.value.sval.ptr,
		    current.value.sval.length);
      context_type = current.value.tsym.type;
      checkpoint = 1;
    }

  first_was_coloncolon = current.token == COLONCOLON;
  last_was_coloncolon = first_was_coloncolon;

  while (next_to_examine <= last_to_examine)
    {
      c_token_and_value next;

      next = token_fifo[next_to_examine];
      ++next_to_examine;

      if (next.token == NAME && last_was_coloncolon)
	{
	  int classification;

	  c_yylval = next.value;
	  if (cpstate->assume_classification != TYPE_CODE_UNDEF)
	    classification = NAME;
	  else
	    classification = classify_inner_name (pstate, search_block,
						  context_type);
	  /* We keep going until we either run out of names, or until
	     we have a qualified name which is not a type.  */
	  if (classification != TYPENAME && classification != NAME)
	    break;

	  /* Accept up to this token.  */
	  checkpoint = next_to_examine;

	  /* Update the partial name we are constructing.  */
	  if (next_to_examine > 1)
	    {
	      /* We don't want to put a leading "::" into the name.  */
	      obstack_grow_str (&name_obstack, "::");
	    }
	  obstack_grow (&name_obstack, next.value.sval.ptr,
			next.value.sval.length);

	  c_yylval.sval.ptr = (const char *) obstack_base (&name_obstack);
	  c_yylval.sval.length = obstack_object_size (&name_obstack);
	  current.value = c_yylval;
	  current.token = classification;

	  last_was_coloncolon = 0;

	  if (cpstate->assume_classification == TYPE_CODE_UNDEF
	      && classification == NAME)
	    break;

	  context_type = c_yylval.tsym.type;
	}
      else if (next.token == COLONCOLON && !last_was_coloncolon)
	last_was_coloncolon = 1;
      else
	{
	  /* We've reached the end of the name.  */
	  break;
	}
    }

  /* If we have a replacement token, install it as the first token in
     the FIFO, and delete the other constituent tokens.  */
  if (checkpoint > 0)
    {
      current.value.sval.ptr
	= obstack_strndup (&cpstate->expansion_obstack,
			   current.value.sval.ptr,
			   current.value.sval.length);

      token_fifo[0] = current;
      if (checkpoint > 1)
	token_fifo.erase (token_fifo.begin () + 1,
			  token_fifo.begin () + checkpoint);
    }

 do_pop:
  current = token_fifo[0];
  token_fifo.erase (token_fifo.begin ());
  c_yylval = current.value;
  return current.token;
}

/* See c-exp-parser.h.  */

void
c_yyerror (const char *msg)
{
  pstate->parse_error (msg);
}


} /* namespace c_exp_parser */

/* See c-exp-parser.h.  */

int
c_parse (struct parser_state *par_state)
{
  using namespace c_exp_parser;

  /* Setting up the parser state.  */
  scoped_restore pstate_restore = make_scoped_restore (&pstate);
  gdb_assert (par_state != NULL);
  pstate = par_state;

  c_parse_state cstate;
  scoped_restore cstate_restore = make_scoped_restore (&cpstate, &cstate);

  macro_scope macro_scope;

  if (par_state->expression_context_block)
    macro_scope
      = sal_macro_scope (find_sal_for_pc (par_state->expression_context_pc, 0));
  else
    macro_scope = default_macro_scope ();
  if (!macro_scope.is_valid ())
    macro_scope = user_macro_scope ();

  scoped_restore restore_macro_scope
    = make_scoped_restore (&expression_macro_scope, &macro_scope);

  scoped_restore restore_yydebug = make_scoped_restore (&c_yydebug,
							par_state->debug);

  /* Initialize some state used by the lexer.  */
  last_was_structop = false;
  saw_name_at_eof = 0;
  paren_depth = 0;

  token_fifo.clear ();
  popping = 0;
  name_obstack.clear ();

  int result = c_yyparse ();
  if (!result)
    pstate->set_operation (pstate->pop ());
  return result;
}

/* See c-exp-parser.h.  */

int
c_parse_escape (const char **ptr, struct obstack *output)
{
  const char *tokptr = *ptr;
  int result = 1;

  /* Some escape sequences undergo character set conversion.  Those we
     translate here.  */
  switch (*tokptr)
    {
      /* Hex escapes do not undergo character set conversion, so keep
	 the escape sequence for later.  */
    case 'x':
      if (output)
	obstack_grow_str (output, "\\x");
      ++tokptr;
      if (!c_isxdigit (*tokptr))
	error (_("\\x escape without a following hex digit"));
      while (c_isxdigit (*tokptr))
	{
	  if (output)
	    obstack_1grow (output, *tokptr);
	  ++tokptr;
	}
      break;

      /* Octal escapes do not undergo character set conversion, so
	 keep the escape sequence for later.  */
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
      {
	int i;
	if (output)
	  obstack_grow_str (output, "\\");
	for (i = 0;
	     i < 3 && c_isdigit (*tokptr) && *tokptr != '8' && *tokptr != '9';
	     ++i)
	  {
	    if (output)
	      obstack_1grow (output, *tokptr);
	    ++tokptr;
	  }
      }
      break;

      /* We handle UCNs later.  We could handle them here, but that
	 would mean a spurious error in the case where the UCN could
	 be converted to the target charset but not the host
	 charset.  */
    case 'u':
    case 'U':
      {
	char c = *tokptr;
	int i, len = c == 'U' ? 8 : 4;
	if (output)
	  {
	    obstack_1grow (output, '\\');
	    obstack_1grow (output, *tokptr);
	  }
	++tokptr;
	if (!c_isxdigit (*tokptr))
	  error (_("\\%c escape without a following hex digit"), c);
	for (i = 0; i < len && c_isxdigit (*tokptr); ++i)
	  {
	    if (output)
	      obstack_1grow (output, *tokptr);
	    ++tokptr;
	  }
      }
      break;

      /* We must pass backslash through so that it does not
	 cause quoting during the second expansion.  */
    case '\\':
      if (output)
	obstack_grow_str (output, "\\\\");
      ++tokptr;
      break;

      /* Escapes which undergo conversion.  */
    case 'a':
      if (output)
	obstack_1grow (output, '\a');
      ++tokptr;
      break;
    case 'b':
      if (output)
	obstack_1grow (output, '\b');
      ++tokptr;
      break;
    case 'f':
      if (output)
	obstack_1grow (output, '\f');
      ++tokptr;
      break;
    case 'n':
      if (output)
	obstack_1grow (output, '\n');
      ++tokptr;
      break;
    case 'r':
      if (output)
	obstack_1grow (output, '\r');
      ++tokptr;
      break;
    case 't':
      if (output)
	obstack_1grow (output, '\t');
      ++tokptr;
      break;
    case 'v':
      if (output)
	obstack_1grow (output, '\v');
      ++tokptr;
      break;

      /* GCC extension.  */
    case 'e':
      if (output)
	obstack_1grow (output, HOST_ESCAPE_CHAR);
      ++tokptr;
      break;

      /* Backslash-newline expands to nothing at all.  */
    case '\n':
      ++tokptr;
      result = 0;
      break;

      /* A few escapes just expand to the character itself.  */
    case '\'':
    case '\"':
    case '?':
      /* GCC extensions.  */
    case '(':
    case '{':
    case '[':
    case '%':
      /* Unrecognized escapes turn into the character itself.  */
    default:
      if (output)
	obstack_1grow (output, *tokptr);
      ++tokptr;
      break;
    }
  *ptr = tokptr;
  return result;
}
