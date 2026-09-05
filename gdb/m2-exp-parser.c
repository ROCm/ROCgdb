/* YACC parser support code for Modula-2 expressions, for GDB.

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

#include "m2-exp-parser.h"
#include "m2-exp-parser-gen.h"
#include "block.h"
#include "expression.h"
#include "language.h"
#include "m2-exp.h"
#include "parser-defs.h"
#include "value.h"

/* The entry point of the bison/yacc-generated parser, defined in
   m2-exp-parser-gen.c.  Bison produces a declaration for m2_yyparse in
   m2-exp-parser-gen.h, but byacc does not, hence this declaration.  */

int m2_yyparse ();

/* Likewise, byacc does not produce a declaration for m2_yydebug.  */

extern int m2_yydebug;

namespace m2_exp_parser
{

/* See m2-exp-parser.h.  */

parser_state *pstate;

/* See m2-exp-parser.h.  */

int number_sign = 1;

/* Take care of parsing a number (anything that starts with a digit).
   Set yylval and return the token type; update lexptr.
   LEN is the number of characters in it.  */

/*** Needs some error checking for the float case ***/

static int
parse_number (int olen)
{
  const char *p = pstate->lexptr;
  ULONGEST n = 0;
  ULONGEST prevn = 0;
  int c,i,ischar=0;
  int base = input_radix;
  int len = olen;

  if(p[len-1] == 'H')
  {
     base = 16;
     len--;
  }
  else if(p[len-1] == 'C' || p[len-1] == 'B')
  {
     base = 8;
     ischar = p[len-1] == 'C';
     len--;
  }

  /* Scan the number */
  for (c = 0; c < len; c++)
  {
    if (p[c] == '.' && base == 10)
      {
	/* It's a float since it contains a point.  */
	if (!parse_float (p, len,
			  parse_m2_type (pstate)->builtin_real,
			  m2_yylval.val))
	  return ERROR;

	pstate->lexptr += len;
	return FLOAT;
      }
    if (p[c] == '.' && base != 10)
       error (_("Floating point numbers must be base 10."));
    if (base == 10 && (p[c] < '0' || p[c] > '9'))
       error (_("Invalid digit \'%c\' in number."),p[c]);
 }

  while (len-- > 0)
    {
      c = *p++;
      n *= base;
      if( base == 8 && (c == '8' || c == '9'))
	 error (_("Invalid digit \'%c\' in octal number."),c);
      if (c >= '0' && c <= '9')
	i = c - '0';
      else
	{
	  if (base == 16 && c >= 'A' && c <= 'F')
	    i = c - 'A' + 10;
	  else
	     return ERROR;
	}
      n+=i;
      if(i >= base)
	 return ERROR;
      if (n == 0 && prevn == 0)
	;
      else if (RANGE_CHECK && prevn >= n)
	range_error (_("Overflow on numeric constant."));

	 prevn=n;
    }

  pstate->lexptr = p;
  if(*p == 'B' || *p == 'C' || *p == 'H')
     pstate->lexptr++;			/* Advance past B,C or H */

  if (ischar)
  {
     m2_yylval.ulval = n;
     return CHAR;
  }

  int int_bits = gdbarch_int_bit (pstate->gdbarch ());
  bool have_signed = number_sign == -1;
  bool have_unsigned = number_sign == 1;
  if (have_signed && fits_in_type (number_sign, n, int_bits, true))
    {
      m2_yylval.lval = n;
      return INT;
    }
  else if (have_unsigned && fits_in_type (number_sign, n, int_bits, false))
    {
      m2_yylval.ulval = n;
      return UINT;
    }
  else
    error (_("Overflow on numeric constant."));
}

/* Some tokens */

static struct
{
   char name[2];
   int token;
} tokentab2[] =
{
    { {'<', '>'},    NOTEQUAL 	},
    { {':', '='},    ASSIGN	},
    { {'<', '='},    LEQ	},
    { {'>', '='},    GEQ	},
    { {':', ':'},    COLONCOLON },

};

/* Some specific keywords */

struct keyword {
   char keyw[10];
   int token;
};

static struct keyword keytab[] =
{
    {"OR" ,   OROR	 },
    {"IN",    IN         },/* Note space after IN */
    {"AND",   LOGICAL_AND},
    {"ABS",   ABS	 },
    {"ADR",   ADR	 },
    {"CHR",   CHR	 },
    {"DEC",   DEC	 },
    {"NOT",   NOT	 },
    {"DIV",   DIV    	 },
    {"INC",   INC	 },
    {"MAX",   MAX_FUNC	 },
    {"MIN",   MIN_FUNC	 },
    {"MOD",   MOD	 },
    {"ODD",   ODD	 },
    {"CAP",   CAP	 },
    {"ORD",   ORD	 },
    {"VAL",   VAL	 },
    {"EXCL",  EXCL	 },
    {"HIGH",  HIGH       },
    {"INCL",  INCL	 },
    {"SIZE",  SIZE       },
    {"FLOAT", FLOAT_FUNC },
    {"TRUNC", TRUNC	 },
    {"TSIZE", SIZE       },
};

/* Depth of parentheses.  */
static int paren_depth;

/* See m2-exp-parser.h.  */

int
m2_yylex (void)
{
  int c;
  int namelen;
  int i;
  const char *tokstart;
  char quote;

 retry:

  pstate->prev_lexptr = pstate->lexptr;

  tokstart = pstate->lexptr;


  /* See if it is a special token of length 2 */
  for( i = 0 ; i < (int) (sizeof tokentab2 / sizeof tokentab2[0]) ; i++)
     if (strncmp (tokentab2[i].name, tokstart, 2) == 0)
     {
	pstate->lexptr += 2;
	return tokentab2[i].token;
     }

  switch (c = *tokstart)
    {
    case 0:
      return 0;

    case ' ':
    case '\t':
    case '\n':
      pstate->lexptr++;
      goto retry;

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
      if (pstate->lexptr[1] >= '0' && pstate->lexptr[1] <= '9')
	break;			/* Falls into number code.  */
      else
      {
	 pstate->lexptr++;
	 return DOT;
      }

/* These are character tokens that appear as-is in the YACC grammar */
    case '+':
    case '-':
    case '*':
    case '/':
    case '^':
    case '<':
    case '>':
    case '[':
    case ']':
    case '=':
    case '{':
    case '}':
    case '#':
    case '@':
    case '~':
    case '&':
      pstate->lexptr++;
      return c;

    case '\'' :
    case '"':
      quote = c;
      for (namelen = 1; (c = tokstart[namelen]) != quote && c != '\0'; namelen++)
	if (c == '\\')
	  {
	    c = tokstart[++namelen];
	    if (c >= '0' && c <= '9')
	      {
		c = tokstart[++namelen];
		if (c >= '0' && c <= '9')
		  c = tokstart[++namelen];
	      }
	  }
      if(c != quote)
	 error (_("Unterminated string or character constant."));
      m2_yylval.sval.ptr = tokstart + 1;
      m2_yylval.sval.length = namelen - 1;
      pstate->lexptr += namelen + 1;

      if(namelen == 2)  	/* Single character */
      {
	   m2_yylval.ulval = tokstart[1];
	   return CHAR;
      }
      else
	 return STRING;
    }

  /* Is it a number?  */
  /* Note:  We have already dealt with the case of the token '.'.
     See case '.' above.  */
  if ((c >= '0' && c <= '9'))
    {
      /* It's a number.  */
      int got_dot = 0, got_e = 0;
      const char *p = tokstart;
      int toktype;

      for (++p ;; ++p)
	{
	  if (!got_e && (*p == 'e' || *p == 'E'))
	    got_dot = got_e = 1;
	  else if (!got_dot && *p == '.')
	    got_dot = 1;
	  else if (got_e && (p[-1] == 'e' || p[-1] == 'E')
		   && (*p == '-' || *p == '+'))
	    /* This is the sign of the exponent, not the end of the
	       number.  */
	    continue;
	  else if ((*p < '0' || *p > '9') &&
		   (*p < 'A' || *p > 'F') &&
		   (*p != 'H'))  /* Modula-2 hexadecimal number */
	    break;
	}
	toktype = parse_number (p - tokstart);
	if (toktype == ERROR)
	  error (_("Invalid number \"%.*s\"."), (int) (p - tokstart),
		 tokstart);
	pstate->lexptr = p;
	return toktype;
    }

  if (!(c == '_' || c == '$'
	|| (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')))
    /* We must have come across a bad character (e.g. ';').  */
    error (_("Invalid character '%c' in expression."), c);

  /* It's a name.  See how long it is.  */
  namelen = 0;
  for (c = tokstart[namelen];
       (c == '_' || c == '$' || (c >= '0' && c <= '9')
	|| (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
       c = tokstart[++namelen])
    ;

  /* The token "if" terminates the expression and is NOT
     removed from the input stream.  */
  if (namelen == 2 && tokstart[0] == 'i' && tokstart[1] == 'f')
    {
      return 0;
    }

  pstate->lexptr += namelen;

  /*  Lookup special keywords */
  for(i = 0 ; i < (int) (sizeof(keytab) / sizeof(keytab[0])) ; i++)
     if (namelen == strlen (keytab[i].keyw)
	 && strncmp (tokstart, keytab[i].keyw, namelen) == 0)
	   return keytab[i].token;

  m2_yylval.sval.ptr = tokstart;
  m2_yylval.sval.length = namelen;

  if (*tokstart == '$')
    return DOLLAR_VARIABLE;

  /* Use token-type BLOCKNAME for symbols that happen to be defined as
     functions.  If this is not so, then ...
     Use token-type TYPENAME for symbols that happen to be defined
     currently as names of types; NAME for other symbols.
     The caller is not constrained to care about the distinction.  */
 {
    std::string tmp = copy_name (m2_yylval.sval);
    struct symbol *sym;

    if (lookup_symtab (current_program_space, tmp.c_str ()) != nullptr)
      return BLOCKNAME;

    sym = lookup_symbol (tmp.c_str (), pstate->expression_context_block,
			 SEARCH_VFT, 0).symbol;
    if (sym && sym->loc_class () == LOC_BLOCK)
      return BLOCKNAME;
    if (lookup_typename (pstate->language (),
			 tmp.c_str (), pstate->expression_context_block, 1))
      return TYPENAME;

    if(sym)
    {
      switch(sym->loc_class ())
       {
       case LOC_STATIC:
       case LOC_REGISTER:
       case LOC_ARG:
       case LOC_REF_ARG:
       case LOC_REGPARM_ADDR:
       case LOC_LOCAL:
       case LOC_CONST:
       case LOC_CONST_BYTES:
       case LOC_OPTIMIZED_OUT:
       case LOC_COMPUTED:
	  return NAME;

       case LOC_TYPEDEF:
	  return TYPENAME;

       case LOC_BLOCK:
	  return BLOCKNAME;

       case LOC_UNDEF:
	  error (_("internal:  Undefined class in m2lex()"));

       case LOC_LABEL:
       case LOC_UNRESOLVED:
	  error (_("internal:  Unforeseen case in m2lex()"));

       default:
	  error (_("unhandled token in m2lex()"));
	  break;
       }
    }
    else
    {
       /* Built-in BOOLEAN type.  This is sort of a hack.  */
       if (startswith (tokstart, "TRUE"))
       {
	  m2_yylval.ulval = 1;
	  return M2_TRUE;
       }
       else if (startswith (tokstart, "FALSE"))
       {
	  m2_yylval.ulval = 0;
	  return M2_FALSE;
       }
    }

    /* Must be another type of name...  */
    return NAME;
 }
}

/* See m2-exp-parser.h.  */

void
m2_yyerror (const char *msg)
{
  pstate->parse_error (msg);
}

} /* namespace m2_exp_parser */

/* See m2-exp-parser.h.  */

int
m2_parse (struct parser_state *par_state)
{
  using namespace m2_exp_parser;

  /* Setting up the parser state.  */
  scoped_restore pstate_restore = make_scoped_restore (&pstate);
  gdb_assert (par_state != NULL);
  pstate = par_state;
  paren_depth = 0;

  int result = m2_yyparse ();
  if (!result)
    pstate->set_operation (pstate->pop ());
  return result;
}
