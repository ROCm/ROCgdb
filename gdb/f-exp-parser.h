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

#ifndef GDB_F_EXP_PARSER_H
#define GDB_F_EXP_PARSER_H

#include "parser-defs.h"
#include "type-stack.h"
#include "f-lang.h"

union f_exp_parser_YYSTYPE;

namespace f_exp_parser {

/* The state of the parser, used internally when we are parsing the
   expression.  */

extern parser_state *pstate;

/* The current type stack.  */

extern struct type_stack *type_stack;

/* Return the Fortran type table for the architecture associated to PS.  */

static inline const struct builtin_f_type *
parse_f_type (parser_state *ps)
{
  return builtin_f_type (ps->gdbarch ());
}

/* Called to match intrinsic function calls with one argument to their
   respective implementation and push the operation.  */

void wrap_unop_intrinsic (exp_opcode opcode);

/* Called to match intrinsic function calls with two arguments to their
   respective implementation and push the operation.  */

void wrap_binop_intrinsic (exp_opcode opcode);

/* Called to match intrinsic function calls with three arguments to their
   respective implementation and push the operation.  */

void wrap_ternop_intrinsic (exp_opcode opcode);

/* Take care of parsing a number (anything that starts with a digit).
   Set yylval and return the token type; update lexptr.
   LEN is the number of characters in it.  */

/*** Needs some error checking for the float case ***/

int parse_number (struct parser_state *par_state, const char *p, int len,
		  int parsed_float, f_exp_parser_YYSTYPE *putithere);

/* Called to setup the type stack when we encounter a '(kind=N)' type
   modifier, performs some bounds checking on 'N' and then pushes this to
   the type stack followed by the 'tp_kind' marker.  */

void push_kind_type (LONGEST val, struct type *type);

/* Called when a type has a '(kind=N)' modifier after it, for example
   'character(kind=1)'.  The BASETYPE is the type described by 'character'
   in our example, and KIND is the integer '1'.  This function returns a
   new type that represents the basetype of a specific kind.  */

struct type *convert_to_kind_type (struct type *basetype, int kind);

/* Read one token, getting characters through lexptr.  */

int f_yylex ();

/* The error handler invoked by the generated parser.  Report MSG as a
   parse error on the current parser state.  */

void f_yyerror (const char *msg);

} /* namespace f_exp_parser */

/* Parse a Fortran expression using the lexer input and context held in
   PAR_STATE.  On success, return 0 and leave the resulting operation set
   on PAR_STATE.  On failure, return non-zero.  */

int f_parse (struct parser_state *par_state);

#endif /* GDB_F_EXP_PARSER_H */
