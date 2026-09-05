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

#ifndef GDB_P_EXP_PARSER_H
#define GDB_P_EXP_PARSER_H

#include "parser-defs.h"
#include "p-lang.h"

union p_exp_parser_YYSTYPE;

namespace p_exp_parser {

/* The state of the parser, used internally when we are parsing the
   expression.  */

extern parser_state *pstate;

/* The type of the sub-expression parsed most recently, or nullptr if it
   is not known.  */

extern struct type *current_type;

/* Non-zero if the left operand of the '/' operator being parsed has an
   integral type.  */

extern int leftdiv_is_integer;

/* Non-zero while the name being lexed should be looked up as a field of
   CURRENT_TYPE.  */

extern int search_field;

/* Save CURRENT_TYPE on an internal stack and reset it.  */

void push_current_type ();

/* Restore CURRENT_TYPE from the internal stack.  */

void pop_current_type ();

/* Take care of parsing a number (anything that starts with a digit).
   Set yylval and return the token type; update lexptr.
   LEN is the number of characters in it.  */

/*** Needs some error checking for the float case ***/

int parse_number (struct parser_state *par_state, const char *p, int len,
		  int parsed_float, p_exp_parser_YYSTYPE *putithere);

/* Read one token, getting characters through lexptr.  */

int pascal_yylex ();

/* The error handler invoked by the generated parser.  Report MSG as a
   parse error on the current parser state.  */

void pascal_yyerror (const char *msg);

} /* namespace p_exp_parser */

/* Parse a Pascal expression using the lexer input and context held in
   PAR_STATE.  On success, return 0 and leave the resulting operation set
   on PAR_STATE.  On failure, return non-zero.  */

int pascal_parse (struct parser_state *par_state);

#endif /* GDB_P_EXP_PARSER_H */
