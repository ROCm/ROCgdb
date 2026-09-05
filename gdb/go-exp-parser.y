/* YACC parser for Go expressions, for GDB.

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

/* This file is derived from c-exp-parser.y, p-exp-parser.y.  */

/* Parse a Go expression from text in a string,
   and return the result as a struct expression pointer.
   That structure contains arithmetic operations in reverse polish,
   with constants represented by operations that are followed by special data.
   See expression.h for the details of the format.
   What is important here is that it can be built up sequentially
   during the process of parsing; the lower levels of the tree always
   come first in the result.

   Note that malloc's and realloc's in this file are transformed to
   xmalloc and xrealloc respectively by the same sed command in the
   makefile that remaps any other malloc/realloc inserted by the parser
   generator.  Doing this with #defines and trying to control the interaction
   with include files (<malloc.h> and <stdlib.h> for example) just became
   too messy, particularly when such includes can be inserted at random
   times by the parser generator.  */

/* Known bugs or limitations:

   - Unicode
   - &^
   - '_' (blank identifier)
   - automatic deref of pointers
   - method expressions
   - interfaces, channels, etc.

   And lots of other things.
   I'm sure there's some cleanup to do.
*/

%{

#include "expression.h"
#include "value.h"
#include "parser-defs.h"
#include "language.h"
#include "go-lang.h"
#include "go-exp-parser.h"
#include "block.h"
#include "expop.h"

using namespace go_exp_parser;
using namespace expr;

%}

/* Although the yacc "value" of an expression is not used,
   since the result is stored in the structure being created,
   other node types do have values.  */

%union
  {
    LONGEST lval;
    struct {
      LONGEST val;
      struct type *type;
    } typed_val_int;
    struct {
      gdb_byte val[16];
      struct type *type;
    } typed_val_float;
    struct stoken sval;
    struct symtoken ssym;
    struct type *tval;
    struct typed_stoken tsval;
    struct ttype tsym;
    int voidval;
    enum exp_opcode opcode;
    struct internalvar *ivar;
    struct stoken_vector svec;
  }

%type <voidval> exp exp1 type_exp start variable lcurly
%type <lval> rcurly
%type <tval> type

%token <typed_val_int> INT
%token <typed_val_float> FLOAT

/* Both NAME and TYPENAME tokens represent symbols in the input,
   and both convey their data as strings.
   But a TYPENAME is a string that happens to be defined as a type
   or builtin type name (such as int or char)
   and a NAME is any other symbol.
   Contexts where this distinction is not important can use the
   nonterminal "name", which matches either NAME or TYPENAME.  */

%token <tsval> RAW_STRING
%token <tsval> STRING
%token <tsval> CHAR
%token <ssym> NAME
%token <tsym> TYPENAME /* Not TYPE_NAME cus already taken.  */
%token <voidval> COMPLETE
/*%type <sval> name*/
%type <svec> string_exp
%type <ssym> name_not_typename

/* A NAME_OR_INT is a symbol which is not known in the symbol table,
   but which would parse as a valid number in the current input radix.
   E.g. "c" when input_radix==16.  Depending on the parse, it will be
   turned into a name or into a number.  */
%token <ssym> NAME_OR_INT

%token <lval> TRUE_KEYWORD FALSE_KEYWORD
%token STRUCT_KEYWORD INTERFACE_KEYWORD TYPE_KEYWORD CHAN_KEYWORD
%token SIZEOF_KEYWORD
%token LEN_KEYWORD CAP_KEYWORD
%token NEW_KEYWORD
%token IOTA_KEYWORD NIL_KEYWORD
%token CONST_KEYWORD
%token DOTDOTDOT
%token ENTRY
%token ERROR

/* Special type cases.  */
%token BYTE_KEYWORD /* An alias of uint8.  */

%token <sval> DOLLAR_VARIABLE

%token <opcode> ASSIGN_MODIFY

%left ','
%left ABOVE_COMMA
%right '=' ASSIGN_MODIFY
%right '?'
%left OROR
%left ANDAND
%left '|'
%left '^'
%left '&'
%left ANDNOT
%left EQUAL NOTEQUAL
%left '<' '>' LEQ GEQ
%left LSH RSH
%left '@'
%left '+' '-'
%left '*' '/' '%'
%right UNARY INCREMENT DECREMENT
%right LEFT_ARROW '.' '[' '('


%%

start   :	exp1
	|	type_exp
	;

type_exp:	type
			{ pstate->push_new<type_operation> ($1); }
	;

/* Expressions, including the comma operator.  */
exp1	:	exp
	|	exp1 ',' exp
			{ pstate->wrap2<comma_operation> (); }
	;

/* Expressions, not including the comma operator.  */
exp	:	'*' exp    %prec UNARY
			{ pstate->wrap<unop_ind_operation> (); }
	;

exp	:	'&' exp    %prec UNARY
			{ pstate->wrap<unop_addr_operation> (); }
	;

exp	:	'-' exp    %prec UNARY
			{ pstate->wrap<unary_neg_operation> (); }
	;

exp	:	'+' exp    %prec UNARY
			{ pstate->wrap<unary_plus_operation> (); }
	;

exp	:	'!' exp    %prec UNARY
			{ pstate->wrap<unary_logical_not_operation> (); }
	;

exp	:	'^' exp    %prec UNARY
			{ pstate->wrap<unary_complement_operation> (); }
	;

exp	:	exp INCREMENT    %prec UNARY
			{ pstate->wrap<postinc_operation> (); }
	;

exp	:	exp DECREMENT    %prec UNARY
			{ pstate->wrap<postdec_operation> (); }
	;

/* foo->bar is not in Go.  May want as a gdb extension.  Later.  */

exp	:	exp '.' name_not_typename
			{
			  pstate->push_new<structop_operation>
			    (pstate->pop (), copy_name ($3.stoken));
			}
	;

exp	:	exp '.' name_not_typename COMPLETE
			{
			  structop_base_operation *op
			    = new structop_operation (pstate->pop (),
						      copy_name ($3.stoken));
			  pstate->mark_struct_expression (op);
			  pstate->push (operation_up (op));
			}
	;

exp	:	exp '.' COMPLETE
			{
			  structop_base_operation *op
			    = new structop_operation (pstate->pop (), "");
			  pstate->mark_struct_expression (op);
			  pstate->push (operation_up (op));
			}
	;

exp	:	exp '[' exp1 ']'
			{ pstate->wrap2<subscript_operation> (); }
	;

exp	:	exp '('
			/* This is to save the value of arglist_len
			   being accumulated by an outer function call.  */
			{ pstate->start_arglist (); }
		arglist ')'	%prec LEFT_ARROW
			{
			  std::vector<operation_up> args
			    = pstate->pop_vector (pstate->end_arglist ());
			  pstate->push_new<funcall_operation>
			    (pstate->pop (), std::move (args));
			}
	;

lcurly	:	'{'
			{ pstate->start_arglist (); }
	;

arglist	:
	;

arglist	:	exp
			{ pstate->arglist_len = 1; }
	;

arglist	:	arglist ',' exp   %prec ABOVE_COMMA
			{ pstate->arglist_len++; }
	;

rcurly	:	'}'
			{ $$ = pstate->end_arglist () - 1; }
	;

exp	:	lcurly type rcurly exp  %prec UNARY
			{
			  pstate->push_new<unop_memval_operation>
			    (pstate->pop (), $2);
			}
	;

exp	:	type '(' exp ')'  %prec UNARY
			{
			  pstate->push_new<unop_cast_operation>
			    (pstate->pop (), $1);
			}
	;

exp	:	'(' exp1 ')'
			{ }
	;

/* Binary operators in order of decreasing precedence.  */

exp	:	exp '@' exp
			{ pstate->wrap2<repeat_operation> (); }
	;

exp	:	exp '*' exp
			{ pstate->wrap2<mul_operation> (); }
	;

exp	:	exp '/' exp
			{ pstate->wrap2<div_operation> (); }
	;

exp	:	exp '%' exp
			{ pstate->wrap2<rem_operation> (); }
	;

exp	:	exp '+' exp
			{ pstate->wrap2<add_operation> (); }
	;

exp	:	exp '-' exp
			{ pstate->wrap2<sub_operation> (); }
	;

exp	:	exp LSH exp
			{ pstate->wrap2<lsh_operation> (); }
	;

exp	:	exp RSH exp
			{ pstate->wrap2<rsh_operation> (); }
	;

exp	:	exp EQUAL exp
			{ pstate->wrap2<equal_operation> (); }
	;

exp	:	exp NOTEQUAL exp
			{ pstate->wrap2<notequal_operation> (); }
	;

exp	:	exp LEQ exp
			{ pstate->wrap2<leq_operation> (); }
	;

exp	:	exp GEQ exp
			{ pstate->wrap2<geq_operation> (); }
	;

exp	:	exp '<' exp
			{ pstate->wrap2<less_operation> (); }
	;

exp	:	exp '>' exp
			{ pstate->wrap2<gtr_operation> (); }
	;

exp	:	exp '&' exp
			{ pstate->wrap2<bitwise_and_operation> (); }
	;

exp	:	exp '^' exp
			{ pstate->wrap2<bitwise_xor_operation> (); }
	;

exp	:	exp '|' exp
			{ pstate->wrap2<bitwise_ior_operation> (); }
	;

exp	:	exp ANDAND exp
			{ pstate->wrap2<logical_and_operation> (); }
	;

exp	:	exp OROR exp
			{ pstate->wrap2<logical_or_operation> (); }
	;

exp	:	exp '?' exp ':' exp	%prec '?'
			{
			  operation_up last = pstate->pop ();
			  operation_up mid = pstate->pop ();
			  operation_up first = pstate->pop ();
			  pstate->push_new<ternop_cond_operation>
			    (std::move (first), std::move (mid),
			     std::move (last));
			}
	;

exp	:	exp '=' exp
			{ pstate->wrap2<assign_operation> (); }
	;

exp	:	exp ASSIGN_MODIFY exp
			{
			  operation_up rhs = pstate->pop ();
			  operation_up lhs = pstate->pop ();
			  pstate->push_new<assign_modify_operation>
			    ($2, std::move (lhs), std::move (rhs));
			}
	;

exp	:	INT
			{
			  pstate->push_new<long_const_operation>
			    ($1.type, $1.val);
			}
	;

exp	:	CHAR
			{
			  struct stoken_vector vec;
			  vec.len = 1;
			  vec.tokens = &$1;
			  pstate->push_c_string ($1.type, &vec);
			}
	;

exp	:	NAME_OR_INT
			{ YYSTYPE val;
			  parse_number (pstate, $1.stoken.ptr,
					$1.stoken.length, 0, &val);
			  pstate->push_new<long_const_operation>
			    (val.typed_val_int.type,
			     val.typed_val_int.val);
			}
	;


exp	:	FLOAT
			{
			  float_data data;
			  std::copy (std::begin ($1.val), std::end ($1.val),
				     std::begin (data));
			  pstate->push_new<float_const_operation> ($1.type, data);
			}
	;

exp	:	variable
	;

exp	:	DOLLAR_VARIABLE
			{
			  pstate->push_dollar ($1);
			}
	;

exp	:	SIZEOF_KEYWORD '(' type ')'  %prec UNARY
			{
			  /* TODO(dje): Go objects in structs.  */
			  /* TODO(dje): What's the right type here?  */
			  struct type *size_type
			    = parse_type (pstate)->builtin_unsigned_int;
			  $3 = check_typedef ($3);
			  pstate->push_new<long_const_operation>
			    (size_type, (LONGEST) $3->length ());
			}
	;

exp	:	SIZEOF_KEYWORD  '(' exp ')'  %prec UNARY
			{
			  /* TODO(dje): Go objects in structs.  */
			  pstate->wrap<unop_sizeof_operation> ();
			}

string_exp:
		STRING
			{
			  /* We copy the string here, and not in the
			     lexer, to guarantee that we do not leak a
			     string.  */
			  /* Note that we NUL-terminate here, but just
			     for convenience.  */
			  struct typed_stoken *vec = XNEW (struct typed_stoken);
			  $$.len = 1;
			  $$.tokens = vec;

			  vec->type = $1.type;
			  vec->length = $1.length;
			  vec->ptr = (char *) malloc ($1.length + 1);
			  memcpy (vec->ptr, $1.ptr, $1.length + 1);
			}

	|	string_exp '+' STRING
			{
			  /* Note that we NUL-terminate here, but just
			     for convenience.  */
			  char *p;
			  ++$$.len;
			  $$.tokens = XRESIZEVEC (struct typed_stoken,
						  $$.tokens, $$.len);

			  p = (char *) malloc ($3.length + 1);
			  memcpy (p, $3.ptr, $3.length + 1);

			  $$.tokens[$$.len - 1].type = $3.type;
			  $$.tokens[$$.len - 1].length = $3.length;
			  $$.tokens[$$.len - 1].ptr = p;
			}
	;

exp	:	string_exp  %prec ABOVE_COMMA
			{
			  int i;

			  /* Always utf8.  */
			  pstate->push_c_string (0, &$1);
			  for (i = 0; i < $1.len; ++i)
			    free ($1.tokens[i].ptr);
			  free ($1.tokens);
			}
	;

exp	:	TRUE_KEYWORD
			{ pstate->push_new<bool_operation> ($1); }
	;

exp	:	FALSE_KEYWORD
			{ pstate->push_new<bool_operation> ($1); }
	;

variable:	name_not_typename ENTRY
			{ struct symbol *sym = $1.sym.symbol;

			  if (sym == NULL
			      || !sym->is_argument ()
			      || !symbol_read_needs_frame (sym))
			    error (_("@entry can be used only for function "
				     "parameters, not for \"%s\""),
				   copy_name ($1.stoken).c_str ());

			  pstate->push_new<var_entry_value_operation> (sym);
			}
	;

variable:	name_not_typename
			{ struct block_symbol sym = $1.sym;

			  if (sym.symbol)
			    {
			      if (symbol_read_needs_frame (sym.symbol))
				pstate->block_tracker->update (sym);

			      pstate->push_new<var_value_operation> (sym);
			    }
			  else if ($1.is_a_field_of_this)
			    {
			      /* TODO(dje): Can we get here?
				 E.g., via a mix of c++ and go?  */
			      gdb_assert_not_reached ("go with `this' field");
			    }
			  else
			    {
			      std::string arg = copy_name ($1.stoken);

			      bound_minimal_symbol msymbol =
				lookup_minimal_symbol (current_program_space, arg.c_str ());
			      if (msymbol.minsym != NULL)
				pstate->push_new<var_msym_value_operation>
				  (msymbol);
			      else if (!current_program_space->has_full_symbols ()
				       && !current_program_space->has_partial_symbols ())
				error (_("No symbol table is loaded.  "
				       "Use the \"file\" command."));
			      else
				error (_("No symbol \"%s\" in current context."),
				       arg.c_str ());
			    }
			}
	;

/* TODO
method_exp: PACKAGENAME '.' name '.' name
			{
			}
	;
*/

type  /* Implements (approximately): [*] type-specifier */
	:	'*' type
			{ $$ = lookup_pointer_type ($2); }
	|	TYPENAME
			{ $$ = $1.type; }
/*
	|	STRUCT_KEYWORD name
			{ $$ = lookup_struct (copy_name ($2),
					      expression_context_block); }
*/
	|	BYTE_KEYWORD
			{ $$ = builtin_go_type (pstate->gdbarch ())
			    ->builtin_uint8; }
	;

/* TODO
name	:	NAME { $$ = $1.stoken; }
	|	TYPENAME { $$ = $1.stoken; }
	|	NAME_OR_INT  { $$ = $1.stoken; }
	;
*/

name_not_typename
	:	NAME
/* These would be useful if name_not_typename was useful, but it is just
   a fake for "variable", so these cause reduce/reduce conflicts because
   the parser can't tell whether NAME_OR_INT is a name_not_typename (=variable,
   =exp) or just an exp.  If name_not_typename was ever used in an lvalue
   context where only a name could occur, this might be useful.
	|	NAME_OR_INT
*/
	;
