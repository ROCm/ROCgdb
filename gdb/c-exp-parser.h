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

#ifndef GDB_C_EXP_PARSER_H
#define GDB_C_EXP_PARSER_H

#include "gdbsupport/gdb_obstack.h"
#include "parser-defs.h"
#include "type-stack.h"

union c_exp_parser_YYSTYPE;

namespace c_exp_parser {

/* Data that must be held for the duration of a parse.  */

struct c_parse_state
{
  /* These are used to hold type lists and type stacks that are
     allocated during the parse.  */
  std::vector<std::unique_ptr<std::vector<struct type *>>> type_lists;
  std::vector<std::unique_ptr<struct type_stack>> type_stacks;

  /* Storage for some strings allocated during the parse.  */
  std::vector<gdb::unique_xmalloc_ptr<char>> strings;

  /* When we find that lexptr (the global var defined in parse.c) is
     pointing at a macro invocation, we expand the invocation, and call
     scan_macro_expansion to save the old lexptr here and point lexptr
     into the expanded text.  When we reach the end of that, we call
     end_macro_expansion to pop back to the value we saved here.  The
     macro expansion code promises to return only fully-expanded text,
     so we don't need to "push" more than one level.

     This is disgusting, of course.  It would be cleaner to do all macro
     expansion beforehand, and then hand that to lexptr.  But we don't
     really know where the expression ends.  Remember, in a command like

     (gdb) break *ADDRESS if CONDITION

     we evaluate ADDRESS in the scope of the current frame, but we
     evaluate CONDITION in the scope of the breakpoint's location.  So
     it's simply wrong to try to macro-expand the whole thing at once.  */
  const char *macro_original_text = nullptr;

  /* We save all intermediate macro expansions on this obstack for the
     duration of a single parse.  The expansion text may sometimes have
     to live past the end of the expansion, due to yacc lookahead.
     Rather than try to be clever about saving the data for a single
     token, we simply keep it all and delete it after parsing has
     completed.  */
  auto_obstack expansion_obstack;

  /* The type stack.  */
  struct type_stack type_stack;

  /* When set, a name token is not looked up.  This can be useful when
     the search domain is known by context.  TYPE_CODE_UNDEF is used
     to mean "unset" here -- typically only types with tags (enum,
     struct, class, union) use this feature, but TYPE_CODE_VOID is
     also used to avoid the lookup for field names.  */
  type_code assume_classification = TYPE_CODE_UNDEF;
};

/* Used for field names, which skip name lookup.  */
struct qualified_name_token
{
  /* The prefix, if any.  This can be nullptr.  */
  const char *prefix;
  /* The field name itself.  */
  const char *name;
  /* True if the COMPLETE token was seen.  */
  bool complete;
};

/* This is set and cleared in c_parse.  */

extern c_parse_state *cpstate;

/* The state of the parser, used internally when we are parsing the
   expression.  */

extern parser_state *pstate;

/* The outer level of a two-level lexer.  This calls the inner lexer
   to return tokens.  It then either returns these tokens, or
   aggregates them into a larger token.  This lets us work around a
   problem in our parsing approach, where the parser could not
   distinguish between qualified names and qualified types at the
   right point.

   This approach is still not ideal, because it mishandles template
   types.  See the comment in lex_one_token for an example.  However,
   this is still an improvement over the earlier approach, and will
   suffice until we move to better parsing technology.  */

int c_yylex ();

/* The error handler invoked by the generated parser.  Report MSG as a
   parse error on the current parser state.  */

void c_yyerror (const char *msg);

/* A helper function for the specific case of a qualified field name,
   like "obj->type1::type2::field".  This takes the type prefix
   ("type1::type2" in the example) and finds the corresponding type.
   It will either throw an exception, or push a scope_operation on the
   operation stack.  */

void handle_qualified_field_name (qualified_name_token token);

/* Return true if the type is aggregate-like.  */

int type_aggregate_p (struct type *type);

/* Take care of parsing a number (anything that starts with a digit).
   Set yylval and return the token type; update lexptr.
   LEN is the number of characters in it.  */

/*** Needs some error checking for the float case ***/

int parse_number (struct parser_state *par_state, const char *buf, int len,
		  int parsed_float, c_exp_parser_YYSTYPE *putithere);

/* Validate a parameter typelist.  */

void check_parameter_typelist (std::vector<struct type *> *params);

/* Returns a stoken of the operator name given by OP (which does not
   include the string "operator").  */

struct stoken operator_stoken (const char *op);

/* Returns a stoken of the type named TYPE.  */

qualified_name_token typename_stoken (const char *type);

/* A convenient overload of copy_name.  */
static inline std::string
copy_name (qualified_name_token token)
{
  if (token.prefix == nullptr)
    return token.name;
  return std::string (token.prefix) + "::" + token.name;
}

} /* namespace c_exp_parser */

/* Parse a C expression using the lexer input and context held in
   PAR_STATE.  On success, return 0 and leave the resulting operation
   set on PAR_STATE.  On failure, return non-zero.  */

int c_parse (struct parser_state *par_state);

/* Parse a C escape sequence.  The initial backslash of the sequence
   is at (*PTR)[-1].  *PTR will be updated to point to just after the
   last character of the sequence.  If OUTPUT is not NULL, the
   translated form of the escape sequence will be written there.  If
   OUTPUT is NULL, no output is written and the call will only affect
   *PTR.  If an escape sequence is expressed in target bytes, then the
   entire sequence will simply be copied to OUTPUT.  Return 1 if any
   character was emitted, 0 otherwise.  */

int c_parse_escape (const char **ptr, struct obstack *output);

#endif /* GDB_C_EXP_PARSER_H */
