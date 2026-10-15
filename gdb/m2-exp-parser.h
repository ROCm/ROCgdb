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

#ifndef GDB_M2_EXP_PARSER_H
#define GDB_M2_EXP_PARSER_H

#include "parser-defs.h"
#include "m2-lang.h"

union m2_exp_parser_YYSTYPE;

namespace m2_exp_parser {

/* The state of the parser, used internally when we are parsing the
   expression.  */

extern parser_state *pstate;

/* The sign of the number being parsed.  */

extern int number_sign;

/* Return the Modula-2 type table for the architecture associated to PS.  */

static inline const struct builtin_m2_type *
parse_m2_type (parser_state *ps)
{
  return builtin_m2_type (ps->gdbarch ());
}

/* Read one token, getting characters through lexptr.  */

/* This is where we will check to make sure that the language and the
   operators used are compatible  */

int m2_yylex ();

/* The error handler invoked by the generated parser.  Report MSG as a
   parse error on the current parser state.  */

void m2_yyerror (const char *msg);

} /* namespace m2_exp_parser */

/* Parse a Modula-2 expression using the lexer input and context held in
   PAR_STATE.  On success, return 0 and leave the resulting operation set
   on PAR_STATE.  On failure, return non-zero.  */

int m2_parse (struct parser_state *par_state);

#endif /* GDB_M2_EXP_PARSER_H */
