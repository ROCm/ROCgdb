/* YACC parser support code for Fortran expressions, for GDB.

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

#include "f-exp-parser.h"
#include "f-exp-parser-gen.h"
#include "block.h"
#include "expression.h"
#include "f-exp.h"
#include "language.h"
#include "parser-defs.h"
#include "value.h"
#include <algorithm>

/* The entry point of the bison/yacc-generated parser, defined in
   f-exp-parser-gen.c.  Bison produces a declaration for f_yyparse in
   f-exp-parser-gen.h, but byacc does not, hence this declaration.  */

int f_yyparse ();

/* Likewise, byacc does not produce a declaration for f_yydebug.  */

extern int f_yydebug;

using namespace expr;

namespace f_exp_parser
{

/* See f-exp-parser.h.  */

parser_state *pstate;

/* Depth of parentheses.  */

static int paren_depth;

/* See f-exp-parser.h.  */

struct type_stack *type_stack;

/* A helper that pops two operations (similar to wrap2), evaluates the last one
   assuming it is a kind parameter, and wraps them in some other operation
   pushing it to the stack.  */

template<typename T>
static void
fortran_wrap2_kind (type *base_type)
{
  operation_up kind_arg = pstate->pop ();
  operation_up arg = pstate->pop ();

  value *val = kind_arg->evaluate (nullptr, pstate->expout.get (),
				   EVAL_AVOID_SIDE_EFFECTS);
  gdb_assert (val != nullptr);

  type *follow_type = convert_to_kind_type (base_type, value_as_long (val));

  pstate->push_new<T> (std::move (arg), follow_type);
}

/* A helper that pops three operations, evaluates the last one assuming it is a
   kind parameter, and wraps them in some other operation pushing it to the
   stack.  */

template<typename T>
static void
fortran_wrap3_kind (type *base_type)
{
  operation_up kind_arg = pstate->pop ();
  operation_up arg2 = pstate->pop ();
  operation_up arg1 = pstate->pop ();

  value *val = kind_arg->evaluate (nullptr, pstate->expout.get (),
				   EVAL_AVOID_SIDE_EFFECTS);
  gdb_assert (val != nullptr);

  type *follow_type = convert_to_kind_type (base_type, value_as_long (val));

  pstate->push_new<T> (std::move (arg1), std::move (arg2), follow_type);
}

/* See f-exp-parser.h.  */

void
wrap_unop_intrinsic (exp_opcode code)
{
  switch (code)
    {
    case UNOP_ABS:
      pstate->wrap<fortran_abs_operation> ();
      break;
    case FORTRAN_FLOOR:
      pstate->wrap<fortran_floor_operation_1arg> ();
      break;
    case FORTRAN_CEILING:
      pstate->wrap<fortran_ceil_operation_1arg> ();
      break;
    case UNOP_FORTRAN_ALLOCATED:
      pstate->wrap<fortran_allocated_operation> ();
      break;
    case UNOP_FORTRAN_RANK:
      pstate->wrap<fortran_rank_operation> ();
      break;
    case UNOP_FORTRAN_SHAPE:
      pstate->wrap<fortran_array_shape_operation> ();
      break;
    case UNOP_FORTRAN_LOC:
      pstate->wrap<fortran_loc_operation> ();
      break;
    case FORTRAN_ASSOCIATED:
      pstate->wrap<fortran_associated_1arg> ();
      break;
    case FORTRAN_ARRAY_SIZE:
      pstate->wrap<fortran_array_size_1arg> ();
      break;
    case FORTRAN_CMPLX:
      pstate->wrap<fortran_cmplx_operation_1arg> ();
      break;
    case FORTRAN_LBOUND:
    case FORTRAN_UBOUND:
      pstate->push_new<fortran_bound_1arg> (code, pstate->pop ());
      break;
    default:
      gdb_assert_not_reached ("unhandled intrinsic");
    }
}

/* See f-exp-parser.h.  */

void
wrap_binop_intrinsic (exp_opcode code)
{
  switch (code)
    {
    case FORTRAN_FLOOR:
      fortran_wrap2_kind<fortran_floor_operation_2arg>
	(parse_f_type (pstate)->builtin_integer);
      break;
    case FORTRAN_CEILING:
      fortran_wrap2_kind<fortran_ceil_operation_2arg>
	(parse_f_type (pstate)->builtin_integer);
      break;
    case BINOP_MOD:
      pstate->wrap2<fortran_mod_operation> ();
      break;
    case BINOP_FORTRAN_MODULO:
      pstate->wrap2<fortran_modulo_operation> ();
      break;
    case FORTRAN_CMPLX:
      pstate->wrap2<fortran_cmplx_operation_2arg> ();
      break;
    case FORTRAN_ASSOCIATED:
      pstate->wrap2<fortran_associated_2arg> ();
      break;
    case FORTRAN_ARRAY_SIZE:
      pstate->wrap2<fortran_array_size_2arg> ();
      break;
    case FORTRAN_LBOUND:
    case FORTRAN_UBOUND:
      {
	operation_up arg2 = pstate->pop ();
	operation_up arg1 = pstate->pop ();
	pstate->push_new<fortran_bound_2arg> (code, std::move (arg1),
					      std::move (arg2));
      }
      break;
    default:
      gdb_assert_not_reached ("unhandled intrinsic");
    }
}

/* See f-exp-parser.h.  */

void
wrap_ternop_intrinsic (exp_opcode code)
{
  switch (code)
    {
    case FORTRAN_LBOUND:
    case FORTRAN_UBOUND:
      {
	operation_up kind_arg = pstate->pop ();
	operation_up arg2 = pstate->pop ();
	operation_up arg1 = pstate->pop ();

	value *val = kind_arg->evaluate (nullptr, pstate->expout.get (),
					 EVAL_AVOID_SIDE_EFFECTS);
	gdb_assert (val != nullptr);

	type *follow_type
	  = convert_to_kind_type (parse_f_type (pstate)->builtin_integer,
				  value_as_long (val));

	pstate->push_new<fortran_bound_3arg> (code, std::move (arg1),
					      std::move (arg2), follow_type);
      }
      break;
    case FORTRAN_ARRAY_SIZE:
      fortran_wrap3_kind<fortran_array_size_3arg>
	(parse_f_type (pstate)->builtin_integer);
      break;
    case FORTRAN_CMPLX:
      fortran_wrap3_kind<fortran_cmplx_operation_3arg>
	(parse_f_type (pstate)->builtin_complex);
      break;
    default:
      gdb_assert_not_reached ("unhandled intrinsic");
    }
}

/* See f-exp-parser.h.  */

int
parse_number (struct parser_state *par_state,
	      const char *p, int len, int parsed_float,
	      f_exp_parser_YYSTYPE *putithere)
{
  ULONGEST n = 0;
  ULONGEST prevn = 0;
  int c;
  int base = input_radix;
  int unsigned_p = 0;
  int long_p = 0;
  ULONGEST high_bit;
  struct type *signed_type;
  struct type *unsigned_type;

  if (parsed_float)
    {
      /* It's a float since it contains a point or an exponent.  */
      /* [dD] is not understood as an exponent by parse_float,
	 change it to 'e'.  */
      char *tmp, *tmp2;

      tmp = xstrdup (p);
      for (tmp2 = tmp; *tmp2; ++tmp2)
	if (*tmp2 == 'd' || *tmp2 == 'D')
	  *tmp2 = 'e';

      /* FIXME: Should this use different types?  */
      putithere->typed_val_float.type = parse_f_type (pstate)->builtin_real_s8;
      bool parsed = parse_float (tmp, len,
				 putithere->typed_val_float.type,
				 putithere->typed_val_float.val);
      xfree (tmp);
      return parsed? FLOAT : ERROR;
    }

  /* Handle base-switching prefixes 0x, 0t, 0d, 0 */
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
      if (c_isupper (c))
	c = c_tolower (c);
      if (len == 0 && c == 'l')
	long_p = 1;
      else if (len == 0 && c == 'u')
	unsigned_p = 1;
      else
	{
	  int i;
	  if (c >= '0' && c <= '9')
	    i = c - '0';
	  else if (c >= 'a' && c <= 'f')
	    i = c - 'a' + 10;
	  else
	    return ERROR;	/* Char not a digit */
	  if (i >= base)
	    return ERROR;		/* Invalid digit in this base */
	  n *= base;
	  n += i;
	}
      /* Test for overflow.  */
      if (prevn == 0 && n == 0)
	;
      else if (RANGE_CHECK && prevn >= n)
	range_error (_("Overflow on numeric constant."));
      prevn = n;
    }

  /* If the number is too big to be an int, or it's got an l suffix
     then it's a long.  Work out if this has to be a long by
     shifting right and seeing if anything remains, and the
     target int size is different to the target long size.

     In the expression below, we could have tested
     (n >> gdbarch_int_bit (parse_gdbarch))
     to see if it was zero,
     but too many compilers warn about that, when ints and longs
     are the same size.  So we shift it twice, with fewer bits
     each time, for the same result.  */

  int bits_available;
  if ((gdbarch_int_bit (par_state->gdbarch ())
       != gdbarch_long_bit (par_state->gdbarch ())
       && ((n >> 2)
	   >> (gdbarch_int_bit (par_state->gdbarch ())-2))) /* Avoid
							    shift warning */
      || long_p)
    {
      bits_available = gdbarch_long_bit (par_state->gdbarch ());
      unsigned_type = parse_type (par_state)->builtin_unsigned_long;
      signed_type = parse_type (par_state)->builtin_long;
  }
  else
    {
      bits_available = gdbarch_int_bit (par_state->gdbarch ());
      unsigned_type = parse_type (par_state)->builtin_unsigned_int;
      signed_type = parse_type (par_state)->builtin_int;
    }
  high_bit = ((ULONGEST)1) << (bits_available - 1);

  if (RANGE_CHECK
      && ((n >> 2) >> (bits_available - 2)))
    range_error (_("Overflow on numeric constant."));

  putithere->typed_val.val = n;

  /* If the high bit of the worked out type is set then this number
     has to be unsigned.  */

  if (unsigned_p || (n & high_bit))
    putithere->typed_val.type = unsigned_type;
  else
    putithere->typed_val.type = signed_type;

  return INT;
}

/* See f-exp-parser.h.  */

void
push_kind_type (LONGEST val, struct type *type)
{
  int ival;

  if (type->is_unsigned ())
    {
      ULONGEST uval = static_cast <ULONGEST> (val);
      if (uval > INT_MAX)
	error (_("kind value out of range"));
      ival = static_cast <int> (uval);
    }
  else
    {
      if (val > INT_MAX || val < 0)
	error (_("kind value out of range"));
      ival = static_cast <int> (val);
    }

  type_stack->push (tp_kind, ival);
}

/* Helper function for convert_to_kind_type.  */
static struct type *
convert_to_kind_type_1 (struct type *basetype, int kind)
{
  if (basetype == parse_f_type (pstate)->builtin_character)
    {
      /* Character of kind 1 is a special case, this is the same as the
	 base character type.  */
      if (kind == 1)
	return parse_f_type (pstate)->builtin_character;
    }
  else if (basetype == parse_f_type (pstate)->builtin_complex)
    {
      if (kind == 4)
	return parse_f_type (pstate)->builtin_complex;
      else if (kind == 8)
	return parse_f_type (pstate)->builtin_complex_s8;
      else if (kind == 16)
	return parse_f_type (pstate)->builtin_complex_s16;
    }
  else if (basetype == parse_f_type (pstate)->builtin_real)
    {
      if (kind == 4)
	return parse_f_type (pstate)->builtin_real;
      else if (kind == 8)
	return parse_f_type (pstate)->builtin_real_s8;
      else if (kind == 16)
	return parse_f_type (pstate)->builtin_real_s16;
    }
  else if (basetype == parse_f_type (pstate)->builtin_logical)
    {
      if (kind == 1)
	return parse_f_type (pstate)->builtin_logical_s1;
      else if (kind == 2)
	return parse_f_type (pstate)->builtin_logical_s2;
      else if (kind == 4)
	return parse_f_type (pstate)->builtin_logical;
      else if (kind == 8)
	return parse_f_type (pstate)->builtin_logical_s8;
    }
  else if (basetype == parse_f_type (pstate)->builtin_integer)
    {
      if (kind == 1)
	return parse_f_type (pstate)->builtin_integer_s1;
      else if (kind == 2)
	return parse_f_type (pstate)->builtin_integer_s2;
      else if (kind == 4)
	return parse_f_type (pstate)->builtin_integer;
      else if (kind == 8)
	return parse_f_type (pstate)->builtin_integer_s8;
    }

  return nullptr;
}

/* See f-exp-parser.h.  */

struct type *
convert_to_kind_type (struct type *basetype, int kind)
{
  struct type *res = convert_to_kind_type_1 (basetype, kind);

  if (res == nullptr || res->code () == TYPE_CODE_ERROR)
    error (_("unsupported kind %d for type %s"),
	   kind, basetype->safe_name ());

  return res;
}

struct f_token
{
  /* The string to match against.  */
  const char *oper;

  /* The lexer token to return.  */
  int token;

  /* The expression opcode to embed within the token.  */
  enum exp_opcode opcode;

  /* When this is true the string in OPER is matched exactly including
     case, when this is false OPER is matched case insensitively.  */
  bool case_sensitive;
};

/* List of Fortran operators.  */

static const struct f_token fortran_operators[] =
{
  { ".and.", BOOL_AND, OP_NULL, false },
  { ".or.", BOOL_OR, OP_NULL, false },
  { ".not.", BOOL_NOT, OP_NULL, false },
  { ".eq.", EQUAL, OP_NULL, false },
  { ".eqv.", EQUAL, OP_NULL, false },
  { ".neqv.", NOTEQUAL, OP_NULL, false },
  { ".xor.", NOTEQUAL, OP_NULL, false },
  { "==", EQUAL, OP_NULL, false },
  { ".ne.", NOTEQUAL, OP_NULL, false },
  { "/=", NOTEQUAL, OP_NULL, false },
  { ".le.", LEQ, OP_NULL, false },
  { "<=", LEQ, OP_NULL, false },
  { ".ge.", GEQ, OP_NULL, false },
  { ">=", GEQ, OP_NULL, false },
  { ".gt.", GREATERTHAN, OP_NULL, false },
  { ">", GREATERTHAN, OP_NULL, false },
  { ".lt.", LESSTHAN, OP_NULL, false },
  { "<", LESSTHAN, OP_NULL, false },
  { "**", STARSTAR, BINOP_EXP, false },
};

/* Holds the Fortran representation of a boolean, and the integer value we
   substitute in when one of the matching strings is parsed.  */
struct f77_boolean_val
{
  /* The string representing a Fortran boolean.  */
  const char *name;

  /* The integer value to replace it with.  */
  int value;
};

/* The set of Fortran booleans.  These are matched case insensitively.  */
static const struct f77_boolean_val boolean_values[]  =
{
  { ".true.", 1 },
  { ".false.", 0 }
};

static const struct f_token f_intrinsics[] =
{
  /* The following correspond to actual functions in Fortran and are case
     insensitive.  */
  { "kind", KIND, OP_NULL, false },
  { "abs", UNOP_INTRINSIC, UNOP_ABS, false },
  { "mod", BINOP_INTRINSIC, BINOP_MOD, false },
  { "floor", UNOP_OR_BINOP_INTRINSIC, FORTRAN_FLOOR, false },
  { "ceiling", UNOP_OR_BINOP_INTRINSIC, FORTRAN_CEILING, false },
  { "modulo", BINOP_INTRINSIC, BINOP_FORTRAN_MODULO, false },
  { "cmplx", UNOP_OR_BINOP_OR_TERNOP_INTRINSIC, FORTRAN_CMPLX, false },
  { "lbound", UNOP_OR_BINOP_OR_TERNOP_INTRINSIC, FORTRAN_LBOUND, false },
  { "ubound", UNOP_OR_BINOP_OR_TERNOP_INTRINSIC, FORTRAN_UBOUND, false },
  { "allocated", UNOP_INTRINSIC, UNOP_FORTRAN_ALLOCATED, false },
  { "associated", UNOP_OR_BINOP_INTRINSIC, FORTRAN_ASSOCIATED, false },
  { "rank", UNOP_INTRINSIC, UNOP_FORTRAN_RANK, false },
  { "size", UNOP_OR_BINOP_OR_TERNOP_INTRINSIC, FORTRAN_ARRAY_SIZE, false },
  { "shape", UNOP_INTRINSIC, UNOP_FORTRAN_SHAPE, false },
  { "loc", UNOP_INTRINSIC, UNOP_FORTRAN_LOC, false },
  { "sizeof", SIZEOF, OP_NULL, false },
};

static const f_token f_keywords[] =
{
  /* Historically these have always been lowercase only in GDB.  */
  { "character", CHARACTER, OP_NULL, true },
  { "complex", COMPLEX_KEYWORD, OP_NULL, true },
  { "complex_4", COMPLEX_S4_KEYWORD, OP_NULL, true },
  { "complex_8", COMPLEX_S8_KEYWORD, OP_NULL, true },
  { "complex_16", COMPLEX_S16_KEYWORD, OP_NULL, true },
  { "integer_1", INT_S1_KEYWORD, OP_NULL, true },
  { "integer_2", INT_S2_KEYWORD, OP_NULL, true },
  { "integer_4", INT_S4_KEYWORD, OP_NULL, true },
  { "integer", INT_KEYWORD, OP_NULL, true },
  { "integer_8", INT_S8_KEYWORD, OP_NULL, true },
  { "logical_1", LOGICAL_S1_KEYWORD, OP_NULL, true },
  { "logical_2", LOGICAL_S2_KEYWORD, OP_NULL, true },
  { "logical", LOGICAL_KEYWORD, OP_NULL, true },
  { "logical_4", LOGICAL_S4_KEYWORD, OP_NULL, true },
  { "logical_8", LOGICAL_S8_KEYWORD, OP_NULL, true },
  { "real", REAL_KEYWORD, OP_NULL, true },
  { "real_4", REAL_S4_KEYWORD, OP_NULL, true },
  { "real_8", REAL_S8_KEYWORD, OP_NULL, true },
  { "real_16", REAL_S16_KEYWORD, OP_NULL, true },
  { "single", SINGLE, OP_NULL, true },
  { "double", DOUBLE, OP_NULL, true },
  { "precision", PRECISION, OP_NULL, true },
};

/* Implementation of a dynamically expandable buffer for processing input
   characters acquired through lexptr and building a value to return in
   yylval.  Ripped off from ch-exp.y */

static char *tempbuf;		/* Current buffer contents */
static int tempbufsize;		/* Size of allocated buffer */
static int tempbufindex;	/* Current index into buffer */

#define GROWBY_MIN_SIZE 64	/* Minimum amount to grow buffer by */

#define CHECKBUF(size) \
  do { \
    if (tempbufindex + (size) >= tempbufsize) \
      { \
	growbuf_by_size (size); \
      } \
  } while (0);

/* Grow the static temp buffer if necessary, including allocating the
   first one on demand.  */

static void
growbuf_by_size (int count)
{
  int growby;

  growby = std::max (count, GROWBY_MIN_SIZE);
  tempbufsize += growby;
  if (tempbuf == NULL)
    tempbuf = (char *) xmalloc (tempbufsize);
  else
    tempbuf = (char *) xrealloc (tempbuf, tempbufsize);
}

/* Blatantly ripped off from ch-exp.y. This routine recognizes F77
   string-literals.

   Recognize a string literal.  A string literal is a nonzero sequence
   of characters enclosed in matching single quotes, except that
   a single character inside single quotes is a character literal, which
   we reject as a string literal.  To embed the terminator character inside
   a string, it is simply doubled (I.E. 'this''is''one''string') */

static int
match_string_literal (void)
{
  const char *tokptr = pstate->lexptr;

  for (tempbufindex = 0, tokptr++; *tokptr != '\0'; tokptr++)
    {
      CHECKBUF (1);
      if (*tokptr == *pstate->lexptr)
	{
	  if (*(tokptr + 1) == *pstate->lexptr)
	    tokptr++;
	  else
	    break;
	}
      tempbuf[tempbufindex++] = *tokptr;
    }
  if (*tokptr == '\0'					/* no terminator */
      || tempbufindex == 0)				/* no string */
    return 0;
  else
    {
      tempbuf[tempbufindex] = '\0';
      f_yylval.sval.ptr = tempbuf;
      f_yylval.sval.length = tempbufindex;
      pstate->lexptr = ++tokptr;
      return STRING_LITERAL;
    }
}

/* This is set if a NAME token appeared at the very end of the input
   string, with no whitespace separating the name from the EOF.  This
   is used only when parsing to do field name completion.  */
static bool saw_name_at_eof;

/* This is set if the previously-returned token was a structure
   operator '%'.  */
static bool last_was_structop;

/* See f-exp-parser.h.  */

int
f_yylex (void)
{
  int c;
  int namelen;
  unsigned int token;
  const char *tokstart;
  bool saw_structop = last_was_structop;

  last_was_structop = false;

 retry:

  pstate->prev_lexptr = pstate->lexptr;

  tokstart = pstate->lexptr;

  /* First of all, let us make sure we are not dealing with the
     special tokens .true. and .false. which evaluate to 1 and 0.  */

  if (*pstate->lexptr == '.')
    {
      for (const auto &candidate : boolean_values)
	{
	  if (strncasecmp (tokstart, candidate.name,
			   strlen (candidate.name)) == 0)
	    {
	      pstate->lexptr += strlen (candidate.name);
	      f_yylval.lval = candidate.value;
	      return BOOLEAN_LITERAL;
	    }
	}
    }

  /* See if it is a Fortran operator.  */
  for (const auto &candidate : fortran_operators)
    if (strncasecmp (tokstart, candidate.oper,
		     strlen (candidate.oper)) == 0)
      {
	gdb_assert (!candidate.case_sensitive);
	pstate->lexptr += strlen (candidate.oper);
	f_yylval.opcode = candidate.opcode;
	return candidate.token;
      }

  switch (c = *tokstart)
    {
    case 0:
      if (saw_name_at_eof)
	{
	  saw_name_at_eof = false;
	  return COMPLETE;
	}
      else if (pstate->parse_completion && saw_structop)
	return COMPLETE;
      return 0;

    case ' ':
    case '\t':
    case '\n':
      pstate->lexptr++;
      goto retry;

    case '\'':
      token = match_string_literal ();
      if (token != 0)
	return (token);
      break;

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
	goto symbol;		/* Nope, must be a symbol.  */
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
	int got_dot = 0, got_e = 0, got_d = 0, toktype;
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
	    if (!hex && !got_e && (*p == 'e' || *p == 'E'))
	      got_dot = got_e = 1;
	    else if (!hex && !got_d && (*p == 'd' || *p == 'D'))
	      got_dot = got_d = 1;
	    else if (!hex && !got_dot && *p == '.')
	      got_dot = 1;
	    else if (((got_e && (p[-1] == 'e' || p[-1] == 'E'))
		     || (got_d && (p[-1] == 'd' || p[-1] == 'D')))
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
	toktype = parse_number (pstate, tokstart, p - tokstart,
				got_dot|got_e|got_d,
				&f_yylval);
	if (toktype == ERROR)
	  error (_("Invalid number \"%.*s\"."), (int) (p - tokstart),
		 tokstart);
	pstate->lexptr = p;
	return toktype;
      }

    case '%':
      last_was_structop = true;
      [[fallthrough]];
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
    }

  if (!(c == '_' || c == '$' || c ==':'
	|| (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')))
    /* We must have come across a bad character (e.g. ';').  */
    error (_("Invalid character '%c' in expression."), c);

  namelen = 0;
  for (c = tokstart[namelen];
       (c == '_' || c == '$' || c == ':' || (c >= '0' && c <= '9')
	|| (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
       c = tokstart[++namelen]);

  /* The token "if" terminates the expression and is NOT
     removed from the input stream.  */

  if (namelen == 2 && tokstart[0] == 'i' && tokstart[1] == 'f')
    return 0;

  pstate->lexptr += namelen;

  /* Catch specific keywords.  */

  for (const auto &keyword : f_keywords)
    if (strlen (keyword.oper) == namelen
	&& ((!keyword.case_sensitive
	     && strncasecmp (tokstart, keyword.oper, namelen) == 0)
	    || (keyword.case_sensitive
		&& strncmp (tokstart, keyword.oper, namelen) == 0)))
      {
	f_yylval.opcode = keyword.opcode;
	return keyword.token;
      }

  f_yylval.sval.ptr = tokstart;
  f_yylval.sval.length = namelen;

  if (*tokstart == '$')
    return DOLLAR_VARIABLE;

  /* Use token-type TYPENAME for symbols that happen to be defined
     currently as names of types; NAME for other symbols.
     The caller is not constrained to care about the distinction.  */
  {
    std::string tmp = copy_name (f_yylval.sval);
    struct block_symbol result;
    const domain_search_flags lookup_domains[] =
    {
      SEARCH_VFT,
      SEARCH_STRUCT_DOMAIN,
      SEARCH_MODULE_DOMAIN
    };
    int hextype;

    for (const auto &domain : lookup_domains)
      {
	result = lookup_symbol (tmp.c_str (), pstate->expression_context_block,
				domain, NULL);
	if (result.symbol && result.symbol->loc_class () == LOC_TYPEDEF)
	  {
	    f_yylval.tsym.type = result.symbol->type ();
	    return TYPENAME;
	  }

	if (result.symbol)
	  break;
      }

    f_yylval.tsym.type
      = language_lookup_primitive_type (pstate->language (),
					pstate->gdbarch (), tmp.c_str ());
    if (f_yylval.tsym.type != NULL)
      return TYPENAME;

    /* This is post the symbol search as symbols can hide intrinsics.  Also,
       give Fortran intrinsics priority over C symbols.  This prevents
       non-Fortran symbols from hiding intrinsics, for example abs.  */
    if (!result.symbol || result.symbol->language () != language_fortran)
      for (const auto &intrinsic : f_intrinsics)
	{
	  gdb_assert (!intrinsic.case_sensitive);
	  if (strlen (intrinsic.oper) == namelen
	      && strncasecmp (tokstart, intrinsic.oper, namelen) == 0)
	    {
	      f_yylval.opcode = intrinsic.opcode;
	      return intrinsic.token;
	    }
	}

    /* Input names that aren't symbols but ARE valid hex numbers,
       when the input radix permits them, can be names or numbers
       depending on the parse.  Note we support radixes > 16 here.  */
    if (!result.symbol
	&& ((tokstart[0] >= 'a' && tokstart[0] < 'a' + input_radix - 10)
	    || (tokstart[0] >= 'A' && tokstart[0] < 'A' + input_radix - 10)))
      {
	f_exp_parser_YYSTYPE newlval;	/* Its value is ignored.  */
	hextype = parse_number (pstate, tokstart, namelen, 0, &newlval);
	if (hextype == INT)
	  {
	    f_yylval.ssym.sym = result;
	    f_yylval.ssym.is_a_field_of_this = false;
	    return NAME_OR_INT;
	  }
      }

    if (pstate->parse_completion && *pstate->lexptr == '\0')
      saw_name_at_eof = true;

    /* Any other kind of symbol */
    f_yylval.ssym.sym = result;
    f_yylval.ssym.is_a_field_of_this = false;
    return NAME;
  }
}

/* See f-exp-parser.h.  */

void
f_yyerror (const char *msg)
{
  pstate->parse_error (msg);
}

} /* namespace f_exp_parser */

/* See f-exp-parser.h.  */

int
f_parse (struct parser_state *par_state)
{
  using namespace f_exp_parser;

  /* Setting up the parser state.  */
  scoped_restore pstate_restore = make_scoped_restore (&pstate);
  scoped_restore restore_yydebug = make_scoped_restore (&f_yydebug,
							par_state->debug);
  gdb_assert (par_state != NULL);
  pstate = par_state;
  last_was_structop = false;
  saw_name_at_eof = false;
  paren_depth = 0;

  struct type_stack stack;
  scoped_restore restore_type_stack
    = make_scoped_restore (&f_exp_parser::type_stack, &stack);

  int result = f_yyparse ();
  if (!result)
    pstate->set_operation (pstate->pop ());
  return result;
}
