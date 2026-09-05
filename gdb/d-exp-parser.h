/* Support code for the D expression parser, for GDB.

   Copyright (C) 2014-2026 Free Software Foundation, Inc.

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

#ifndef GDB_D_EXP_PARSER_H
#define GDB_D_EXP_PARSER_H

#include "parser-defs.h"
#include "type-stack.h"
#include "d-lang.h"

union d_exp_parser_YYSTYPE;

namespace d_exp_parser {

/* The state of the parser, used internally when we are parsing the
   expression.  */

extern parser_state *pstate;

/* The current type stack.  */

extern struct type_stack *type_stack;

/* Return the D type table for the architecture associated to PS.  */

static inline const struct builtin_d_type *
parse_d_type (parser_state *ps)
{
  return builtin_d_type (ps->gdbarch ());
}

/* Return true if the type is aggregate-like.  */

int type_aggregate_p (struct type *type);

/* Take care of parsing a number (anything that starts with a digit).
   Set yylval and return the token type; update lexptr.
   LEN is the number of characters in it.  */

/*** Needs some error checking for the float case ***/

int parse_number (struct parser_state *ps, const char *p, int len,
		  int parsed_float, d_exp_parser_YYSTYPE *putithere);

/* The outer level of a two-level lexer.  This calls the inner lexer
   to return tokens.  It then either returns these tokens, or
   aggregates them into a larger token.  This lets us work around a
   problem in our parsing approach, where the parser could not
   distinguish between qualified names and qualified types at the
   right point.  */

int d_yylex ();

/* The error handler invoked by the generated parser.  Report MSG as a
   parse error on the current parser state.  */

void d_yyerror (const char *msg);

} /* namespace d_exp_parser */

/* Parse a D expression using the lexer input and context held in
   PAR_STATE.  On success, return 0 and leave the resulting operation
   set on PAR_STATE.  On failure, return non-zero.  */

int d_parse (struct parser_state *par_state);

#endif /* GDB_D_EXP_PARSER_H */
