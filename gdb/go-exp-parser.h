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

#ifndef GDB_GO_EXP_PARSER_H
#define GDB_GO_EXP_PARSER_H

#include "parser-defs.h"

union go_exp_parser_YYSTYPE;

namespace go_exp_parser {

/* The state of the parser, used internally when we are parsing the
   expression.  */

extern parser_state *pstate;

/* Take care of parsing a number (anything that starts with a digit).
   Set yylval and return the token type; update lexptr.
   LEN is the number of characters in it.  */

int parse_number (struct parser_state *par_state, const char *p, int len,
		  int parsed_float, go_exp_parser_YYSTYPE *putithere);

/* This is taken from c-exp-parser.y mostly to get something working.
   The basic structure has been kept because we may yet need some of it.  */

int go_yylex ();

/* The error handler invoked by the generated parser.  Report MSG as a
   parse error on the current parser state.  */

void go_yyerror (const char *msg);

} /* namespace go_exp_parser */

/* Parse a Go expression using the lexer input and context held in
   PAR_STATE.  On success, return 0 and leave the resulting operation set
   on PAR_STATE.  On failure, return non-zero.  */

int go_parse (struct parser_state *par_state);

#endif /* GDB_GO_EXP_PARSER_H */
