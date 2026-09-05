/* State of the C++ name parser, for GDB.

   Copyright (C) 2003-2026 Free Software Foundation, Inc.

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

#ifndef GDB_CP_NAME_PARSER_H
#define GDB_CP_NAME_PARSER_H

#include "demangle.h"

union cp_name_parser_YYSTYPE;
struct demangle_parse_info;

/* Flags passed to cpname_state::d_qualify.  */

#define QUAL_CONST 1
#define QUAL_RESTRICT 2
#define QUAL_VOLATILE 4

/* Flags passed to cpname_state::d_int_type.  */

#define INT_CHAR	(1 << 0)
#define INT_SHORT	(1 << 1)
#define INT_LONG	(1 << 2)
#define INT_LLONG	(1 << 3)

#define INT_SIGNED	(1 << 4)
#define INT_UNSIGNED	(1 << 5)

#define d_left(dc) (dc)->u.s_binary.left
#define d_right(dc) (dc)->u.s_binary.right

namespace cp_name_parser {

/* State of an ongoing parse.  */

struct cpname_state
{
  cpname_state (const char *input, demangle_parse_info *info)
    : lexptr (input),
      prev_lexptr (input),
      demangle_info (info)
  { }

  /* Un-push a character into the lexer.  This can only un-push the
     previous character in the input string.  */
  void unpush (char c)
  {
    gdb_assert (lexptr[-1] == c);
    --lexptr;
  }

  /* LEXPTR is the current pointer into our lex buffer.  PREV_LEXPTR
     is the start of the last token lexed, only used for diagnostics.
     ERROR_LEXPTR is the first place an error occurred.  GLOBAL_ERRMSG
     is the first error message encountered.  */

  const char *lexptr, *prev_lexptr;
  const char *error_lexptr = nullptr;
  const char *global_errmsg = nullptr;

  demangle_parse_info *demangle_info;

  /* The parse tree created by the parser is stored here after a
     successful parse.  */

  struct demangle_component *global_result = nullptr;

  struct demangle_component *d_grab ();

  /* Helper functions.  These wrap the demangler tree interface,
     handle allocation from our global store, and return the allocated
     component.  */

  struct demangle_component *fill_comp (enum demangle_component_type d_type,
					struct demangle_component *lhs,
					struct demangle_component *rhs);

  struct demangle_component *make_operator (const char *name, int args);

  struct demangle_component *make_dtor (enum gnu_v3_dtor_kinds kind,
					struct demangle_component *name);

  struct demangle_component *make_builtin_type (const char *name);

  struct demangle_component *make_name (const char *name, int len);

  struct demangle_component *d_qualify (struct demangle_component *lhs,
					int qualifiers, int is_method);

  struct demangle_component *d_int_type (int flags);

  struct demangle_component *d_unary (const char *name,
				      struct demangle_component *lhs);

  struct demangle_component *d_binary (const char *name,
				       struct demangle_component *lhs,
				       struct demangle_component *rhs);

  int parse_number (const char *p, int len, int parsed_float,
		    cp_name_parser_YYSTYPE *lvalp);
};

} /* namespace cp_name_parser */

/* The lexer used by the generated parser.  */

int cpname_yylex (cp_name_parser_YYSTYPE *lvalp,
		  cp_name_parser::cpname_state *state);

/* The error handler invoked by the generated parser.  Report MSG as a
   parse error on the current parser state.  */

void cpname_yyerror (cp_name_parser::cpname_state *state, const char *msg);

#endif /* GDB_CP_NAME_PARSER_H */
