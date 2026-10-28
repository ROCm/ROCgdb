/* Support code for the Ada expression parser, for GDB.

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

#ifndef GDB_ADA_EXP_PARSER_H
#define GDB_ADA_EXP_PARSER_H

#include "ada-exp.h"
#include "parser-defs.h"

namespace ada_exp_parser
{

using ada_assign_up = std::unique_ptr<expr::ada_assign_operation>;

/* Data that must be held for the duration of a parse.  */

struct ada_parse_state
{
  explicit ada_parse_state (const char *expr)
    : m_original_expr (expr)
  {
  }

  std::string find_completion_bounds ();

  const gdb_mpz *push_integer (gdb_mpz &&val)
  {
    auto &result = m_int_storage.emplace_back (new gdb_mpz (std::move (val)));
    return result.get ();
  }

  /* The components being constructed during this parse.  */
  std::vector<expr::ada_component_up> components;

  /* The associations being constructed during this parse.  */
  std::vector<expr::ada_association_up> associations;

  /* The stack of currently active assignment expressions.  This is used
     to implement '@', the target name symbol.  */
  std::vector<ada_assign_up> assignments;

  /* Track currently active iterated assignment names.  */
  gdb::unordered_string_map<std::vector<expr::ada_index_var_operation *>>
       iterated_associations;

  auto_obstack temp_space;

  /* Lex a single token.  ada_yylval is updated if needed.  The token
     value is returned.  The lexer state in this object or in the
     parser_state object will be updated as appropriate.  */
  int lex ();

private:

  /* A lexer helper function to lex a number.  */
  int lex_number ();

  /* A lexer helper function to lex a dollar variable like $x or
     $$.  */
  int lex_dollar ();

  /* A lexer helper that is called when a '.' is seen.  */
  int lex_dot ();

  /* A lexer helper that is called when a single quote is seen.  */
  int lex_single_quote ();

  /* A lexer helper for lexing a string.  */
  int lex_double_quote ();

  /* A helper function that examines TEXT.  If TEXT looks like a
     simple identifier, a pointer to the character past the end of the
     identifier is returned.  Otherwise, nullptr is returned.  */
  const char *lex_simple_identifier (const char *text);

  /* A helper that does some extra checking for the "thread"
     pseudo-keyword.  It returns true if this instance of "thread"
     should be reported as an EOF rather than an identifier.  TEXT
     points to just past the end of the "thread" keyword.  */
  bool check_thread_keyword (const char *text);

  /* Check whether the current token, held at pstate->lexptr with
     length LEN, is a keyword.  Returns the token value if so; returns
     nullopt if not.  */
  std::optional<int> check_keyword (size_t len);

  /* Check whether BRACKET holds text matching a hex character as
     would be seen in an Ada string or character literal.  If so,
     returns a pointer to the character just past the end of the form;
     otherwise returns nullptr.  */
  const char *recognize_encoded_char (const char *bracket);

  /* A lexer helper for lexing an identifier.  */
  int lex_identifier (const char *id_end);

  /* A helper that examines START.  If this is a quoted operator name
     (like "mod" or "+"), returns a pointer to just after the final
     quote.  Otherwise, returns nullptr.  */
  const char *lex_operator (const char *start);

  /* Return the syntactic code corresponding to the attribute name or
     abbreviation STR.  COMPLETING should be true if the name was
     lexed at the end of the input and completion is enabled for this
     parse.  */
  int process_attribute (const std::string &str, bool completing);

  /* Return a canonicalized version of NAME0.  The
     resulting string is valid until the next call to ada_parse.  If
     NAME0 contains the substring "___", it is assumed to be already
     encoded and the resulting name is equal to it.  Similarly, if the name
     starts with '<', it is copied verbatim.  Otherwise, it differs
     from NAME0 in that:
     + Characters between '...' are transferred verbatim to yylval.ssym.
     + Trailing "'" characters in quoted sequences are removed (a leading quote is
     preserved to indicate that the name is not to be GNAT-encoded).
     + Unquoted whitespace is removed.
     + Unquoted alphabetic characters are mapped to lower case.
     Result is returned as a struct stoken, but for convenience, the string
     is also null-terminated.  Result string valid until the next call of
     ada_parse.
  */
  stoken process_id (std::string_view name0);

  /* Throw an exception when lexing, indicating that an invalid
     character was seen.  */
  [[noreturn]] void invalid_char ();

  /* We don't have a good way to manage non-POD data in Yacc, so store
     values here.  The storage here is only valid for the duration of
     the parse.  */
  std::vector<std::unique_ptr<gdb_mpz>> m_int_storage;

  /* The original expression string.  */
  const char *m_original_expr;

  /* Depth of parentheses, used by the lexer.  */
  int m_paren_depth = 0;

  /* This holds some state needed when lexing a single quote.
     Normally, a single quote will introduce a character literal.
     However, after an identifier, it will not, to support cases like
     Char_Enum_Type'('B'); see char_enum.exp for examples.  */
  bool m_before_qual_quote = false;
};

/* The current state of the parser, used internally when parsing an
   expression.  */

extern struct parser_state *pstate;

/* The current Ada parser object.  */

extern struct ada_parse_state *ada_parser;

/* Return TEXT[0..LEN-1], a string literal without surrounding quotes,
   with special hex character notations replaced with characters.
   Result valid until the next call to ada_parse.  */

stoken processString (const char *text, int len);

/* Interprets the prefix of NUM that consists of digits of the given BASE
   as an integer of that BASE, with the string EXP as an exponent.
   Puts value in yylval, and returns INT, if the string is valid.  Causes
   an error if the number is improperly formatted.   BASE, if NULL, defaults
   to "10", and EXP to "1".  The EXP does not contain a leading 'e' or 'E'.
 */

int processInt (parser_state *par_state, const char *base0, const char *num0,
		const char *exp0);

/* Parse NUM0 as a floating-point literal, store the result in yylval,
   and return the FLOAT token.  */

int processReal (struct parser_state *par_state, const char *num0);

/* Like parser_state::pop, but handles Ada type resolution.
   DEPROCEDURE_P and CONTEXT_TYPE are passed to the resolve method, if
   called.  */

expr::operation_up ada_pop (bool deprocedure_p = true,
			    struct type *context_type = nullptr);

/* Handle operator overloading.  Either returns a function all
   operation wrapping the arguments, or it returns null, leaving the
   caller to construct the appropriate operation.  If RHS is null, a
   unary operator is assumed.  */

expr::operation_up maybe_overload (enum exp_opcode op, expr::operation_up &lhs,
				   expr::operation_up &rhs);

/* Handle Ada type resolution for OP.  DEPROCEDURE_P and CONTEXT_TYPE
   are passed to the resolve method, if called.  */

expr::operation_up resolve (expr::operation_up &&op, bool deprocedure_p,
			    struct type *context_type);

/* Pop NARGS operands, then a callee operand, and use these to
   construct and push a new Ada function call operation.  */

void ada_funcall (int nargs);

/* Pop the most recent component from the global stack, and return
   it.  */

expr::ada_component_up pop_component ();

/* Create and push an address-of operation, as appropriate for Ada.
   If TYPE is not NULL, the resulting operation will be wrapped in a
   cast to TYPE.  */

void ada_addrof (type *type = nullptr);

/* Make a new ada_tick_completer and wrap it in a unique pointer.  */

std::unique_ptr<expr_completion_base> make_tick_completer (struct stoken tok);

/* Pop the N most recent components from the global stack, and return
   them in a vector.  */

std::vector<expr::ada_component_up> pop_components (int n);

/* Examine the final element of the 'components' vector, and return it
   as a pointer to an ada_choices_component.  The caller is
   responsible for ensuring that the final element is in fact an
   ada_choices_component.  */

expr::ada_choices_component *choice_component ();

/* Pop the N most recent associations from the global stack, and
   return them in a vector.  */

std::vector<expr::ada_association_up> pop_associations (int n);

/* Return the type of System.Address for PAR_STATE, or the builtin data
   pointer type if that type is not defined.  */

type *type_system_address (parser_state *par_state);

/* Write integer or boolean constant ARG of type TYPE.  */

void write_int (parser_state *par_state, LONGEST arg, type *type);

/* Look up NAME0 (an unencoded identifier or dotted name) in BLOCK (or
   expression_block_context if NULL).  If it denotes a type, return
   that type.  Otherwise, write expression code to evaluate it as an
   object and return NULL. In this second case, NAME0 will, in general,
   have the form <name>(.<selector_name>)*, where <name> is an object
   or renaming encoded in the debugging data.  Calls error if no
   prefix <name> matches a name in the debugging data (i.e., matches
   either a complete name or, as a wild-card match, the final
   identifier).  */

type *write_var_or_type (parser_state *par_state,
			 const block *block, stoken name0);

/* A wrapper for write_var_or_type that is used specifically when
   completion is requested for the last of a sequence of
   identifiers.  */

type *write_var_or_type_completion (struct parser_state *par_state,
				    const block *block,
				    struct stoken name0);

/* Look up the block for the function or file named RAW_NAME, in the
   context of CONTEXT (or the global context if NULL).  Calls error if
   no matching block is found.  */

const block *block_lookup (const block *context, const char *raw_name);

/* Write a left side of a component association (e.g., NAME in NAME =>
   exp).  If NAME has the form of a selected component, write it as an
   ordinary expression.  If it is a simple variable that unambiguously
   corresponds to exactly one symbol that does not denote a type or an
   object renaming, also write it normally as an OP_VAR_VALUE.
   Otherwise, write it as an OP_NAME.

   Unfortunately, we don't know at this point whether NAME is supposed
   to denote a record component name or the value of an array index.
   Therefore, it is not appropriate to disambiguate an ambiguous name
   as we normally would, nor to replace a renaming with its referent.
   As a result, in the (one hopes) rare case that one writes an
   aggregate such as (R => 42) where R renames an object or is an
   ambiguous name, one must write instead ((R) => 42). */

void write_name_assoc (parser_state *par_state, stoken name);

/* Return the character type appropriate for the character constant
   VALUE: a normal, wide, or wide-wide character type depending on the
   magnitude of VALUE.  */

type *type_for_char (parser_state *par_state, ULONGEST value);

/* The error handler invoked by the generated parser.  Report MSG as a
   parse error on the current parser state.  */

void ada_yyerror (const char *msg);

/* Like parser_state::wrap, but use ada_pop to pop the value.  */

template<typename T, typename... Args>
void
ada_wrap (Args... args)
{
  expr::operation_up arg = ada_pop ();
  pstate->push_new<T> (std::move (arg), std::forward<Args> (args)...);
}

/* Like parser_state::wrap, but use ada_pop to pop the value, and
   handle unary overloading.  */

template<typename T>
void
ada_wrap_overload (enum exp_opcode op)
{
  expr::operation_up arg = ada_pop ();
  expr::operation_up empty;

  expr::operation_up call = maybe_overload (op, arg, empty);
  if (call == nullptr)
    call = expr::make_operation<T> (std::move (arg));
  pstate->push (std::move (call));
}

/* A variant of parser_state::wrap2 that uses ada_pop to pop both
   operands, and then pushes a new Ada-wrapped operation of the
   template type T.  */

template<typename T>
void
ada_un_wrap2 (enum exp_opcode op)
{
  expr::operation_up rhs = ada_pop ();
  expr::operation_up lhs = ada_pop ();

  expr::operation_up wrapped = maybe_overload (op, lhs, rhs);
  if (wrapped == nullptr)
    {
      wrapped = expr::make_operation<T> (std::move (lhs), std::move (rhs));
      wrapped = expr::make_operation<expr::ada_wrapped_operation> (
	std::move (wrapped));
    }
  pstate->push (std::move (wrapped));
}

/* A variant of parser_state::wrap2 that uses ada_pop to pop both
   operands.  Unlike ada_un_wrap2, ada_wrapped_operation is not
   used.  */

template<typename T>
void
ada_wrap2 (enum exp_opcode op)
{
  expr::operation_up rhs = ada_pop ();
  expr::operation_up lhs = ada_pop ();
  expr::operation_up call = maybe_overload (op, lhs, rhs);
  if (call == nullptr)
    call = expr::make_operation<T> (std::move (lhs), std::move (rhs));
  pstate->push (std::move (call));
}

/* A variant of parser_state::wrap2 that uses ada_pop to pop both
   operands.  OP is also passed to the constructor of the new binary
   operation.  */

template<typename T>
void
ada_wrap_op (enum exp_opcode op)
{
  expr::operation_up rhs = ada_pop ();
  expr::operation_up lhs = ada_pop ();
  expr::operation_up call = maybe_overload (op, lhs, rhs);
  if (call == nullptr)
    call = expr::make_operation<T> (op, std::move (lhs), std::move (rhs));
  pstate->push (std::move (call));
}

/* Pop three operands using ada_pop, then construct a new ternary
   operation of type T and push it.  */

template<typename T>
void
ada_wrap3 ()
{
  expr::operation_up rhs = ada_pop ();
  expr::operation_up mid = ada_pop ();
  expr::operation_up lhs = ada_pop ();
  pstate->push_new<T> (std::move (lhs), std::move (mid), std::move (rhs));
}

/* Create a new ada_component_up of the indicated type and arguments,
   and push it on the global 'components' vector.  */

template<typename T, typename... Arg>
void
push_component (Arg... args)
{
  ada_parser->components.emplace_back (new T (std::forward<Arg> (args)...));
}

/* Create a new ada_association_up of the indicated type and
   arguments, and push it on the global 'associations' vector.  */

template<typename T, typename... Arg>
void
push_association (Arg... args)
{
  ada_parser->associations.emplace_back (new T (std::forward<Arg> (args)...));
}

} /* namespace ada_exp_parser */

/* The Ada expression parser entry point.  */

int ada_parse (struct parser_state *par_state);

#endif /* GDB_ADA_EXP_PARSER_H */
