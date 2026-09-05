/* YACC parser support code for Go expressions, for GDB.

   Copyright (C) 2012-2026 Free Software Foundation, Inc.

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

#include "go-exp-parser.h"
#include "go-exp-parser-gen.h"
#include "block.h"
#include "c-exp-parser.h"
#include "c-lang.h"
#include "charset.h"
#include "go-lang.h"
#include "language.h"
#include "parser-defs.h"
#include "value.h"

/* The entry point of the bison/yacc-generated parser, defined in
   go-exp-parser-gen.c.  Bison produces a declaration for go_yyparse in
   go-exp-parser-gen.h, but byacc does not, hence this declaration.  */

int go_yyparse ();

/* Likewise, byacc does not produce a declaration for go_yydebug.  */

extern int go_yydebug;

namespace go_exp_parser
{

/* See go-exp-parser.h.  */

parser_state *pstate;

/* See go-exp-parser.h.  */

int
parse_number (struct parser_state *par_state,
	      const char *p, int len, int parsed_float,
	      go_exp_parser_YYSTYPE *putithere)
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
      const struct builtin_go_type *builtin_go_types
	= builtin_go_type (par_state->gdbarch ());

      /* Handle suffixes: 'f' for float32, 'l' for long double.
	 FIXME: This appears to be an extension -- do we want this?  */
      if (len >= 1 && c_tolower (p[len - 1]) == 'f')
	{
	  putithere->typed_val_float.type
	    = builtin_go_types->builtin_float32;
	  len--;
	}
      else if (len >= 1 && c_tolower (p[len - 1]) == 'l')
	{
	  putithere->typed_val_float.type
	    = parse_type (par_state)->builtin_long_double;
	  len--;
	}
      /* Default type for floating-point literals is float64.  */
      else
	{
	  putithere->typed_val_float.type
	    = builtin_go_types->builtin_float64;
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

      case 'b':
      case 'B':
	if (len >= 3)
	  {
	    p += 2;
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
	  if (n == 0 && prevn == 0)
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

  /* Skip the quote.  */
  quote = *tokptr;
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

  value->type = (int) C_STRING | (quote == '\'' ? C_CHAR : 0); /*FIXME*/
  value->ptr = (char *) obstack_base (&tempbuf);
  value->length = obstack_object_size (&tempbuf);

  *outptr = tokptr;

  return quote == '\'' ? CHAR : STRING;
}

struct go_token
{
  const char *oper;
  int token;
  enum exp_opcode opcode;
};

static const struct go_token tokentab3[] =
  {
    {">>=", ASSIGN_MODIFY, BINOP_RSH},
    {"<<=", ASSIGN_MODIFY, BINOP_LSH},
    /*{"&^=", ASSIGN_MODIFY, BINOP_BITWISE_ANDNOT}, TODO */
    {"...", DOTDOTDOT, OP_NULL},
  };

static const struct go_token tokentab2[] =
  {
    {"+=", ASSIGN_MODIFY, BINOP_ADD},
    {"-=", ASSIGN_MODIFY, BINOP_SUB},
    {"*=", ASSIGN_MODIFY, BINOP_MUL},
    {"/=", ASSIGN_MODIFY, BINOP_DIV},
    {"%=", ASSIGN_MODIFY, BINOP_REM},
    {"|=", ASSIGN_MODIFY, BINOP_BITWISE_IOR},
    {"&=", ASSIGN_MODIFY, BINOP_BITWISE_AND},
    {"^=", ASSIGN_MODIFY, BINOP_BITWISE_XOR},
    {"++", INCREMENT, OP_NULL},
    {"--", DECREMENT, OP_NULL},
    /*{"->", RIGHT_ARROW, OP_NULL}, Doesn't exist in Go.  */
    {"<-", LEFT_ARROW, OP_NULL},
    {"&&", ANDAND, OP_NULL},
    {"||", OROR, OP_NULL},
    {"<<", LSH, OP_NULL},
    {">>", RSH, OP_NULL},
    {"==", EQUAL, OP_NULL},
    {"!=", NOTEQUAL, OP_NULL},
    {"<=", LEQ, OP_NULL},
    {">=", GEQ, OP_NULL},
    /*{"&^", ANDNOT, OP_NULL}, TODO */
  };

/* Identifier-like tokens.  */
static const struct go_token ident_tokens[] =
  {
    {"true", TRUE_KEYWORD, OP_NULL},
    {"false", FALSE_KEYWORD, OP_NULL},
    {"nil", NIL_KEYWORD, OP_NULL},
    {"const", CONST_KEYWORD, OP_NULL},
    {"struct", STRUCT_KEYWORD, OP_NULL},
    {"type", TYPE_KEYWORD, OP_NULL},
    {"interface", INTERFACE_KEYWORD, OP_NULL},
    {"chan", CHAN_KEYWORD, OP_NULL},
    {"byte", BYTE_KEYWORD, OP_NULL}, /* An alias of uint8.  */
    {"len", LEN_KEYWORD, OP_NULL},
    {"cap", CAP_KEYWORD, OP_NULL},
    {"new", NEW_KEYWORD, OP_NULL},
    {"iota", IOTA_KEYWORD, OP_NULL},
  };

/* This is set if a NAME token appeared at the very end of the input
   string, with no whitespace separating the name from the EOF.  This
   is used only when parsing to do field name completion.  */
static int saw_name_at_eof;

/* This is set if the previously-returned token was a structure
   operator -- either '.' or ARROW.  This is used only when parsing to
   do field name completion.  */
static int last_was_structop;

/* Depth of parentheses.  */
static int paren_depth;

/* Read one token, getting characters through lexptr.  */

static int
lex_one_token (struct parser_state *par_state)
{
  int c;
  int namelen;
  const char *tokstart;
  int saw_structop = last_was_structop;

  last_was_structop = 0;

 retry:

  par_state->prev_lexptr = par_state->lexptr;

  tokstart = par_state->lexptr;
  /* See if it is a special token of length 3.  */
  for (const auto &token : tokentab3)
    if (strncmp (tokstart, token.oper, 3) == 0)
      {
	par_state->lexptr += 3;
	go_yylval.opcode = token.opcode;
	return token.token;
      }

  /* See if it is a special token of length 2.  */
  for (const auto &token : tokentab2)
    if (strncmp (tokstart, token.oper, 2) == 0)
      {
	par_state->lexptr += 2;
	go_yylval.opcode = token.opcode;
	/* NOTE: -> doesn't exist in Go, so we don't need to watch for
	   setting last_was_structop here.  */
	return token.token;
      }

  switch (c = *tokstart)
    {
    case 0:
      if (saw_name_at_eof)
	{
	  saw_name_at_eof = 0;
	  return COMPLETE;
	}
      else if (saw_structop)
	return COMPLETE;
      else
	return 0;

    case ' ':
    case '\t':
    case '\n':
      par_state->lexptr++;
      goto retry;

    case '[':
    case '(':
      paren_depth++;
      par_state->lexptr++;
      return c;

    case ']':
    case ')':
      if (paren_depth == 0)
	return 0;
      paren_depth--;
      par_state->lexptr++;
      return c;

    case ',':
      if (pstate->comma_terminates
	  && paren_depth == 0)
	return 0;
      par_state->lexptr++;
      return c;

    case '.':
      /* Might be a floating point number.  */
      if (par_state->lexptr[1] < '0' || par_state->lexptr[1] > '9')
	{
	  if (pstate->parse_completion)
	    last_was_structop = 1;
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
	int got_dot = 0, got_e = 0, toktype;
	const char *p = tokstart;
	int hex = input_radix > 10;

	if (c == '0' && (p[1] == 'x' || p[1] == 'X'))
	  {
	    p += 2;
	    hex = 1;
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
	toktype = parse_number (par_state, tokstart, p - tokstart,
				got_dot|got_e, &go_yylval);
	if (toktype == ERROR)
	  error (_("Invalid number \"%.*s\"."), (int) (p - tokstart),
		 tokstart);
	par_state->lexptr = p;
	return toktype;
      }

    case '@':
      {
	const char *p = &tokstart[1];
	size_t len = strlen ("entry");

	while (c_isspace (*p))
	  p++;
	if (strncmp (p, "entry", len) == 0 && !c_isalnum (p[len])
	    && p[len] != '_')
	  {
	    par_state->lexptr = &p[len];
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
      par_state->lexptr++;
      return c;

    case '\'':
    case '"':
    case '`':
      {
	int host_len;
	int result = parse_string_or_char (tokstart, &par_state->lexptr,
					   &go_yylval.tsval, &host_len);
	if (result == CHAR)
	  {
	    if (host_len == 0)
	      error (_("Empty character constant."));
	    else if (host_len > 2 && c == '\'')
	      {
		++tokstart;
		namelen = par_state->lexptr - tokstart - 1;
		goto tryname;
	      }
	    else if (host_len > 1)
	      error (_("Invalid character constant."));
	  }
	return result;
      }
    }

  if (!(c == '_' || c == '$'
	|| (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')))
    /* We must have come across a bad character (e.g. ';').  */
    error (_("Invalid character '%c' in expression."), c);

  /* It's a name.  See how long it is.  */
  namelen = 0;
  for (c = tokstart[namelen];
       (c == '_' || c == '$' || (c >= '0' && c <= '9')
	|| (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));)
    {
      c = tokstart[++namelen];
    }

  /* The token "if" terminates the expression and is NOT removed from
     the input stream.  It doesn't count if it appears in the
     expansion of a macro.  */
  if (namelen == 2
      && tokstart[0] == 'i'
      && tokstart[1] == 'f')
    {
      return 0;
    }

  /* For the same reason (breakpoint conditions), "thread N"
     terminates the expression.  "thread" could be an identifier, but
     an identifier is never followed by a number without intervening
     punctuation.
     Handle abbreviations of these, similarly to
     breakpoint.c:find_condition_and_thread.
     TODO: Watch for "goroutine" here?  */
  if (namelen >= 1
      && strncmp (tokstart, "thread", namelen) == 0
      && (tokstart[namelen] == ' ' || tokstart[namelen] == '\t'))
    {
      const char *p = skip_spaces (tokstart + namelen + 1);
      if (*p >= '0' && *p <= '9')
	return 0;
    }

  par_state->lexptr += namelen;

  tryname:

  go_yylval.sval.ptr = tokstart;
  go_yylval.sval.length = namelen;

  /* Catch specific keywords.  */
  std::string copy = copy_name (go_yylval.sval);
  for (const auto &token : ident_tokens)
    if (copy == token.oper)
      {
	/* It is ok to always set this, even though we don't always
	   strictly need to.  */
	go_yylval.opcode = token.opcode;
	return token.token;
      }

  if (*tokstart == '$')
    return DOLLAR_VARIABLE;

  if (pstate->parse_completion && *par_state->lexptr == '\0')
    saw_name_at_eof = 1;
  return NAME;
}

/* An object of this type is pushed on a FIFO by the "outer" lexer.  */
struct go_token_and_value
{
  int token;
  go_exp_parser_YYSTYPE value;
};

/* A FIFO of tokens that have been read but not yet returned to the
   parser.  */
static std::vector<go_token_and_value> token_fifo;

/* Non-zero if the lexer should return tokens from the FIFO.  */
static int popping;

/* Temporary storage for yylex; this holds symbol names as they are
   built up.  */
static auto_obstack name_obstack;

/* Build "package.name" in name_obstack.
   For convenience of the caller, the name is NUL-terminated,
   but the NUL is not included in the recorded length.  */

static struct stoken
build_packaged_name (const char *package, int package_len,
		     const char *name, int name_len)
{
  struct stoken result;

  name_obstack.clear ();
  obstack_grow (&name_obstack, package, package_len);
  obstack_grow_str (&name_obstack, ".");
  obstack_grow (&name_obstack, name, name_len);
  obstack_grow (&name_obstack, "", 1);
  result.ptr = (char *) obstack_base (&name_obstack);
  result.length = obstack_object_size (&name_obstack) - 1;

  return result;
}

/* Return non-zero if NAME is a package name.
   BLOCK is the scope in which to interpret NAME; this can be NULL
   to mean the global scope.  */

static int
package_name_p (const char *name, const struct block *block)
{
  struct symbol *sym;
  struct field_of_this_result is_a_field_of_this;

  sym = lookup_symbol (name, block, SEARCH_TYPE_DOMAIN,
		       &is_a_field_of_this).symbol;

  if (sym
      && sym->loc_class () == LOC_TYPEDEF
      && sym->type ()->code () == TYPE_CODE_MODULE)
    return 1;

  return 0;
}

/* Classify a (potential) function in the "unsafe" package.
   We fold these into "keywords" to keep things simple, at least until
   something more complex is warranted.  */

static int
classify_unsafe_function (struct stoken function_name)
{
  std::string copy = copy_name (function_name);

  if (copy == "Sizeof")
    {
      go_yylval.sval = function_name;
      return SIZEOF_KEYWORD;
    }

  error (_("Unknown function in `unsafe' package: %s"), copy.c_str ());
}

/* Classify token(s) "name1.name2" where name1 is known to be a package.
   The contents of the token are in `yylval'.
   Updates yylval and returns the new token type.

   The result is one of NAME, NAME_OR_INT, or TYPENAME.  */

static int
classify_packaged_name (const struct block *block)
{
  struct block_symbol sym;
  struct field_of_this_result is_a_field_of_this;

  std::string copy = copy_name (go_yylval.sval);

  sym = lookup_symbol (copy.c_str (), block, SEARCH_VFT, &is_a_field_of_this);

  if (sym.symbol)
    {
      go_yylval.ssym.sym = sym;
      go_yylval.ssym.is_a_field_of_this = is_a_field_of_this.type != NULL;
    }

  return NAME;
}

/* Classify a NAME token.
   The contents of the token are in `yylval'.
   Updates yylval and returns the new token type.
   BLOCK is the block in which lookups start; this can be NULL
   to mean the global scope.

   The result is one of NAME, NAME_OR_INT, or TYPENAME.  */

static int
classify_name (struct parser_state *par_state, const struct block *block)
{
  struct type *type;
  struct block_symbol sym;
  struct field_of_this_result is_a_field_of_this;

  std::string copy = copy_name (go_yylval.sval);

  /* Try primitive types first so they win over bad/weird debug info.  */
  type = language_lookup_primitive_type (par_state->language (),
					 par_state->gdbarch (),
					 copy.c_str ());
  if (type != NULL)
    {
      /* NOTE: We take advantage of the fact that yylval coming in was a
	 NAME, and that struct ttype is a compatible extension of struct
	 stoken, so yylval.tsym.stoken is already filled in.  */
      go_yylval.tsym.type = type;
      return TYPENAME;
    }

  /* TODO: What about other types?  */

  sym = lookup_symbol (copy.c_str (), block, SEARCH_VFT, &is_a_field_of_this);

  if (sym.symbol)
    {
      go_yylval.ssym.sym = sym;
      go_yylval.ssym.is_a_field_of_this = is_a_field_of_this.type != NULL;
      return NAME;
    }

  /* If we didn't find a symbol, look again in the current package.
     This is to, e.g., make "p global_var" work without having to specify
     the package name.  We intentionally only looks for objects in the
     current package.  */

  {
    gdb::unique_xmalloc_ptr<char> current_package_name
      = go_block_package_name (block);

    if (current_package_name != NULL)
      {
	struct stoken sval =
	  build_packaged_name (current_package_name.get (),
			       strlen (current_package_name.get ()),
			       copy.c_str (), copy.size ());

	sym = lookup_symbol (sval.ptr, block, SEARCH_VFT,
			     &is_a_field_of_this);
	if (sym.symbol)
	  {
	    go_yylval.ssym.stoken = sval;
	    go_yylval.ssym.sym = sym;
	    go_yylval.ssym.is_a_field_of_this = is_a_field_of_this.type != NULL;
	    return NAME;
	  }
      }
  }

  /* Input names that aren't symbols but ARE valid hex numbers, when
     the input radix permits them, can be names or numbers depending
     on the parse.  Note we support radixes > 16 here.  */
  if ((copy[0] >= 'a' && copy[0] < 'a' + input_radix - 10)
      || (copy[0] >= 'A' && copy[0] < 'A' + input_radix - 10))
    {
      go_exp_parser_YYSTYPE newlval;	/* Its value is ignored.  */
      int hextype = parse_number (par_state, copy.c_str (),
				  go_yylval.sval.length, 0, &newlval);
      if (hextype == INT)
	{
	  go_yylval.ssym.sym.symbol = NULL;
	  go_yylval.ssym.sym.block = NULL;
	  go_yylval.ssym.is_a_field_of_this = 0;
	  return NAME_OR_INT;
	}
    }

  go_yylval.ssym.sym.symbol = NULL;
  go_yylval.ssym.sym.block = NULL;
  go_yylval.ssym.is_a_field_of_this = 0;
  return NAME;
}

/* See go-exp-parser.h.  */

int
go_yylex (void)
{
  go_token_and_value current, next;

  if (popping && !token_fifo.empty ())
    {
      go_token_and_value tv = token_fifo[0];
      token_fifo.erase (token_fifo.begin ());
      go_yylval = tv.value;
      /* There's no need to fall through to handle package.name
	 as that can never happen here.  In theory.  */
      return tv.token;
    }
  popping = 0;

  current.token = lex_one_token (pstate);

  /* TODO: Need a way to force specifying name1 as a package.
     .name1.name2 ?  */

  if (current.token != NAME)
    return current.token;

  /* See if we have "name1 . name2".  */

  current.value = go_yylval;
  next.token = lex_one_token (pstate);
  next.value = go_yylval;

  if (next.token == '.')
    {
      go_token_and_value name2;

      name2.token = lex_one_token (pstate);
      name2.value = go_yylval;

      if (name2.token == NAME)
	{
	  /* Ok, we have "name1 . name2".  */
	  std::string copy = copy_name (current.value.sval);

	  if (copy == "unsafe")
	    {
	      popping = 1;
	      return classify_unsafe_function (name2.value.sval);
	    }

	  if (package_name_p (copy.c_str (), pstate->expression_context_block))
	    {
	      popping = 1;
	      go_yylval.sval = build_packaged_name (current.value.sval.ptr,
						 current.value.sval.length,
						 name2.value.sval.ptr,
						 name2.value.sval.length);
	      return classify_packaged_name (pstate->expression_context_block);
	    }
	}

      token_fifo.push_back (next);
      token_fifo.push_back (name2);
    }
  else
    token_fifo.push_back (next);

  /* If we arrive here we don't have a package-qualified name.  */

  popping = 1;
  go_yylval = current.value;
  return classify_name (pstate, pstate->expression_context_block);
}

/* See go-exp-parser.h.  */

void
go_yyerror (const char *msg)
{
  pstate->parse_error (msg);
}

} /* namespace go_exp_parser */

/* See go-exp-parser.h.  */

int
go_parse (struct parser_state *par_state)
{
  using namespace go_exp_parser;

  /* Setting up the parser state.  */
  scoped_restore pstate_restore = make_scoped_restore (&pstate);
  gdb_assert (par_state != NULL);
  pstate = par_state;

  scoped_restore restore_yydebug = make_scoped_restore (&go_yydebug,
							par_state->debug);

  /* Initialize some state used by the lexer.  */
  last_was_structop = 0;
  saw_name_at_eof = 0;
  paren_depth = 0;

  token_fifo.clear ();
  popping = 0;
  name_obstack.clear ();

  int result = go_yyparse ();
  if (!result)
    pstate->set_operation (pstate->pop ());
  return result;
}
