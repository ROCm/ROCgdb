/* YACC parser for D expressions, for GDB.

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

/* This file is derived from c-exp-parser.y, jv-exp.y.  */

/* Parse a D expression from text in a string,
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

%{

#include "expression.h"
#include "value.h"
#include "parser-defs.h"
#include "language.h"
#include "d-lang.h"
#include "d-exp-parser.h"
#include "block.h"
#include "type-stack.h"
#include "expop.h"
#include "cli/cli-style.h"

using namespace d_exp_parser;
using namespace expr;

/* Bring the d_exp_parser::type_stack global into this scope, so that it hides
   the struct type_stack type name.  */
using d_exp_parser::type_stack;

%}

/* Although the yacc "value" of an expression is not used,
   since the result is stored in the structure being created,
   other node types do have values.  */

%union
  {
    struct {
      LONGEST val;
      struct type *type;
    } typed_val_int;
    struct {
      gdb_byte val[16];
      struct type *type;
    } typed_val_float;
    struct symbol *sym;
    struct type *tval;
    struct typed_stoken tsval;
    struct stoken sval;
    struct ttype tsym;
    struct symtoken ssym;
    int ival;
    int voidval;
    enum exp_opcode opcode;
    struct stoken_vector svec;
  }

%token <sval> IDENTIFIER UNKNOWN_NAME
%token <tsym> TYPENAME
%token <voidval> COMPLETE

/* A NAME_OR_INT is a symbol which is not known in the symbol table,
   but which would parse as a valid number in the current input radix.
   E.g. "c" when input_radix==16.  Depending on the parse, it will be
   turned into a name or into a number.  */

%token <sval> NAME_OR_INT

%token <typed_val_int> INTEGER_LITERAL
%token <typed_val_float> FLOAT_LITERAL
%token <tsval> CHARACTER_LITERAL
%token <tsval> STRING_LITERAL

%type <svec> StringExp
%type <tval> BasicType TypeExp
%type <sval> IdentifierExp
%type <ival> ArrayLiteral

%token ENTRY
%token ERROR

/* Keywords that have a constant value.  */
%token TRUE_KEYWORD FALSE_KEYWORD NULL_KEYWORD
/* Class 'super' accessor.  */
%token SUPER_KEYWORD
/* Properties.  */
%token CAST_KEYWORD SIZEOF_KEYWORD
%token TYPEOF_KEYWORD TYPEID_KEYWORD
%token INIT_KEYWORD
/* Comparison keywords.  */
/* Type storage classes.  */
%token IMMUTABLE_KEYWORD CONST_KEYWORD SHARED_KEYWORD
/* Non-scalar type keywords.  */
%token STRUCT_KEYWORD UNION_KEYWORD
%token CLASS_KEYWORD INTERFACE_KEYWORD
%token ENUM_KEYWORD TEMPLATE_KEYWORD
%token DELEGATE_KEYWORD FUNCTION_KEYWORD

%token <sval> DOLLAR_VARIABLE

%token <opcode> ASSIGN_MODIFY

%left ','
%right '=' ASSIGN_MODIFY
%right '?'
%left OROR
%left ANDAND
%left '|'
%left '^'
%left '&'
%left EQUAL NOTEQUAL '<' '>' LEQ GEQ
%right LSH RSH
%left '+' '-'
%left '*' '/' '%'
%right HATHAT
%left IDENTITY NOTIDENTITY
%right INCREMENT DECREMENT
%right '.' '[' '('
%token DOTDOT


%%

start   :
	Expression
|	TypeExp
;

/* Expressions, including the comma operator.  */

Expression:
	CommaExpression
;

CommaExpression:
	AssignExpression
|	AssignExpression ',' CommaExpression
		{ pstate->wrap2<comma_operation> (); }
;

AssignExpression:
	ConditionalExpression
|	ConditionalExpression '=' AssignExpression
		{ pstate->wrap2<assign_operation> (); }
|	ConditionalExpression ASSIGN_MODIFY AssignExpression
		{
		  operation_up rhs = pstate->pop ();
		  operation_up lhs = pstate->pop ();
		  pstate->push_new<assign_modify_operation>
		    ($2, std::move (lhs), std::move (rhs));
		}
;

ConditionalExpression:
	OrOrExpression
|	OrOrExpression '?' Expression ':' ConditionalExpression
		{
		  operation_up last = pstate->pop ();
		  operation_up mid = pstate->pop ();
		  operation_up first = pstate->pop ();
		  pstate->push_new<ternop_cond_operation>
		    (std::move (first), std::move (mid),
		     std::move (last));
		}
;

OrOrExpression:
	AndAndExpression
|	OrOrExpression OROR AndAndExpression
		{ pstate->wrap2<logical_or_operation> (); }
;

AndAndExpression:
	OrExpression
|	AndAndExpression ANDAND OrExpression
		{ pstate->wrap2<logical_and_operation> (); }
;

OrExpression:
	XorExpression
|	OrExpression '|' XorExpression
		{ pstate->wrap2<bitwise_ior_operation> (); }
;

XorExpression:
	AndExpression
|	XorExpression '^' AndExpression
		{ pstate->wrap2<bitwise_xor_operation> (); }
;

AndExpression:
	CmpExpression
|	AndExpression '&' CmpExpression
		{ pstate->wrap2<bitwise_and_operation> (); }
;

CmpExpression:
	ShiftExpression
|	EqualExpression
|	IdentityExpression
|	RelExpression
;

EqualExpression:
	ShiftExpression EQUAL ShiftExpression
		{ pstate->wrap2<equal_operation> (); }
|	ShiftExpression NOTEQUAL ShiftExpression
		{ pstate->wrap2<notequal_operation> (); }
;

IdentityExpression:
	ShiftExpression IDENTITY ShiftExpression
		{ pstate->wrap2<equal_operation> (); }
|	ShiftExpression NOTIDENTITY ShiftExpression
		{ pstate->wrap2<notequal_operation> (); }
;

RelExpression:
	ShiftExpression '<' ShiftExpression
		{ pstate->wrap2<less_operation> (); }
|	ShiftExpression LEQ ShiftExpression
		{ pstate->wrap2<leq_operation> (); }
|	ShiftExpression '>' ShiftExpression
		{ pstate->wrap2<gtr_operation> (); }
|	ShiftExpression GEQ ShiftExpression
		{ pstate->wrap2<geq_operation> (); }
;

ShiftExpression:
	AddExpression
|	ShiftExpression LSH AddExpression
		{ pstate->wrap2<lsh_operation> (); }
|	ShiftExpression RSH AddExpression
		{ pstate->wrap2<rsh_operation> (); }
;

AddExpression:
	MulExpression
|	AddExpression '+' MulExpression
		{ pstate->wrap2<add_operation> (); }
|	AddExpression '-' MulExpression
		{ pstate->wrap2<sub_operation> (); }
|	AddExpression '~' MulExpression
		{ pstate->wrap2<concat_operation> (); }
;

MulExpression:
	UnaryExpression
|	MulExpression '*' UnaryExpression
		{ pstate->wrap2<mul_operation> (); }
|	MulExpression '/' UnaryExpression
		{ pstate->wrap2<div_operation> (); }
|	MulExpression '%' UnaryExpression
		{ pstate->wrap2<rem_operation> (); }

UnaryExpression:
	'&' UnaryExpression
		{ pstate->wrap<unop_addr_operation> (); }
|	INCREMENT UnaryExpression
		{ pstate->wrap<preinc_operation> (); }
|	DECREMENT UnaryExpression
		{ pstate->wrap<predec_operation> (); }
|	'*' UnaryExpression
		{ pstate->wrap<unop_ind_operation> (); }
|	'-' UnaryExpression
		{ pstate->wrap<unary_neg_operation> (); }
|	'+' UnaryExpression
		{ pstate->wrap<unary_plus_operation> (); }
|	'!' UnaryExpression
		{ pstate->wrap<unary_logical_not_operation> (); }
|	'~' UnaryExpression
		{ pstate->wrap<unary_complement_operation> (); }
|	TypeExp '.' SIZEOF_KEYWORD
		{ pstate->wrap<unop_sizeof_operation> (); }
|	CastExpression
|	PowExpression
;

CastExpression:
	CAST_KEYWORD '(' TypeExp ')' UnaryExpression
		{ pstate->wrap2<unop_cast_type_operation> (); }
	/* C style cast is illegal D, but is still recognised in
	   the grammar, so we keep this around for convenience.  */
|	'(' TypeExp ')' UnaryExpression
		{ pstate->wrap2<unop_cast_type_operation> (); }
;

PowExpression:
	PostfixExpression
|	PostfixExpression HATHAT UnaryExpression
		{ pstate->wrap2<exp_operation> (); }
;

PostfixExpression:
	PrimaryExpression
|	PostfixExpression '.' COMPLETE
		{
		  structop_base_operation *op
		    = new structop_ptr_operation (pstate->pop (), "");
		  pstate->mark_struct_expression (op);
		  pstate->push (operation_up (op));
		}
|	PostfixExpression '.' IDENTIFIER
		{
		  pstate->push_new<structop_operation>
		    (pstate->pop (), copy_name ($3));
		}
|	PostfixExpression '.' IDENTIFIER COMPLETE
		{
		  structop_base_operation *op
		    = new structop_operation (pstate->pop (), copy_name ($3));
		  pstate->mark_struct_expression (op);
		  pstate->push (operation_up (op));
		}
|	PostfixExpression '.' SIZEOF_KEYWORD
		{ pstate->wrap<unop_sizeof_operation> (); }
|	PostfixExpression INCREMENT
		{ pstate->wrap<postinc_operation> (); }
|	PostfixExpression DECREMENT
		{ pstate->wrap<postdec_operation> (); }
|	CallExpression
|	IndexExpression
|	SliceExpression
;

ArgumentList:
	AssignExpression
		{ pstate->arglist_len = 1; }
|	ArgumentList ',' AssignExpression
		{ pstate->arglist_len++; }
;

ArgumentList_opt:
	/* EMPTY */
		{ pstate->arglist_len = 0; }
|	ArgumentList
;

CallExpression:
	PostfixExpression '('
		{ pstate->start_arglist (); }
	ArgumentList_opt ')'
		{
		  std::vector<operation_up> args
		    = pstate->pop_vector (pstate->end_arglist ());
		  pstate->push_new<funcall_operation>
		    (pstate->pop (), std::move (args));
		}
;

IndexExpression:
	PostfixExpression '[' ArgumentList ']'
		{ if (pstate->arglist_len > 0)
		    {
		      std::vector<operation_up> args
			= pstate->pop_vector (pstate->arglist_len);
		      pstate->push_new<multi_subscript_operation>
			(pstate->pop (), std::move (args));
		    }
		  else
		    pstate->wrap2<subscript_operation> ();
		}
;

SliceExpression:
	PostfixExpression '[' ']'
		{ /* Do nothing.  */ }
|	PostfixExpression '[' AssignExpression DOTDOT AssignExpression ']'
		{
		  operation_up last = pstate->pop ();
		  operation_up mid = pstate->pop ();
		  operation_up first = pstate->pop ();
		  pstate->push_new<ternop_slice_operation>
		    (std::move (first), std::move (mid),
		     std::move (last));
		}
;

PrimaryExpression:
	'(' Expression ')'
		{ /* Do nothing.  */ }
|	IdentifierExp
		{
		  std::string copy = copy_name ($1);
		  struct field_of_this_result is_a_field_of_this;
		  struct block_symbol sym;

		  /* Handle VAR, which could be local or global.  */
		  sym = lookup_symbol (copy.c_str (),
				       pstate->expression_context_block,
				       SEARCH_VFT, &is_a_field_of_this);
		  if (sym.symbol && sym.symbol->loc_class () != LOC_TYPEDEF)
		    {
		      if (symbol_read_needs_frame (sym.symbol))
			pstate->block_tracker->update (sym);
		      pstate->push_new<var_value_operation> (sym);
		    }
		  else if (is_a_field_of_this.type != NULL)
		     {
		      /* It hangs off of `this'.  Must not inadvertently convert from a
			 method call to data ref.  */
		      pstate->block_tracker->update (sym);
		      operation_up thisop
			= make_operation<op_this_operation> ();
		      pstate->push_new<structop_ptr_operation>
			(std::move (thisop), std::move (copy));
		    }
		  else
		    {
		      /* Lookup foreign name in global static symbols.  */
		      bound_minimal_symbol msymbol
			= lookup_minimal_symbol (current_program_space, copy.c_str ());
		      if (msymbol.minsym != NULL)
			pstate->push_new<var_msym_value_operation> (msymbol);
		      else if (!current_program_space->has_full_symbols ()
			       && !current_program_space->has_partial_symbols ())
			error (_("No symbol table is loaded.  Use the \"%ps\" command"),
			       styled_string (command_style.style (), "file"));
		      else
			error (_("No symbol \"%s\" in current context."),
			       copy.c_str ());
		    }
		  }
|	TypeExp '.' IdentifierExp
			{ struct type *type = check_typedef ($1);

			  /* Check if the qualified name is in the global
			     context.  However if the symbol has not already
			     been resolved, it's not likely to be found.  */
			  if (type->code () == TYPE_CODE_MODULE)
			    {
			      struct block_symbol sym;
			      const char *type_name = type->safe_name ();
			      int type_name_len = strlen (type_name);
			      std::string name
				= string_printf ("%.*s.%.*s",
						 type_name_len, type_name,
						 $3.length, $3.ptr);

			      sym =
				lookup_symbol (name.c_str (),
					       (const struct block *) NULL,
					       SEARCH_VFT, NULL);
			      pstate->push_symbol (name.c_str (), sym);
			    }
			  else
			    {
			      /* Check if the qualified name resolves as a member
				 of an aggregate or an enum type.  */
			      if (!type_aggregate_p (type))
				error (_("`%s' is not defined as an aggregate type."),
				       type->safe_name ());

			      pstate->push_new<scope_operation>
				(type, copy_name ($3));
			    }
			}
|	DOLLAR_VARIABLE
		{ pstate->push_dollar ($1); }
|	NAME_OR_INT
		{ YYSTYPE val;
		  parse_number (pstate, $1.ptr, $1.length, 0, &val);
		  pstate->push_new<long_const_operation>
		    (val.typed_val_int.type, val.typed_val_int.val); }
|	NULL_KEYWORD
		{ struct type *type = parse_d_type (pstate)->builtin_void;
		  type = lookup_pointer_type (type);
		  pstate->push_new<long_const_operation> (type, 0); }
|	TRUE_KEYWORD
		{ pstate->push_new<bool_operation> (true); }
|	FALSE_KEYWORD
		{ pstate->push_new<bool_operation> (false); }
|	INTEGER_LITERAL
		{ pstate->push_new<long_const_operation> ($1.type, $1.val); }
|	FLOAT_LITERAL
		{
		  float_data data;
		  std::copy (std::begin ($1.val), std::end ($1.val),
			     std::begin (data));
		  pstate->push_new<float_const_operation> ($1.type, data);
		}
|	CHARACTER_LITERAL
		{ struct stoken_vector vec;
		  vec.len = 1;
		  vec.tokens = &$1;
		  pstate->push_c_string (0, &vec); }
|	StringExp
		{ int i;
		  pstate->push_c_string (0, &$1);
		  for (i = 0; i < $1.len; ++i)
		    free ($1.tokens[i].ptr);
		  free ($1.tokens); }
|	ArrayLiteral
		{
		  std::vector<operation_up> args
		    = pstate->pop_vector ($1);
		  pstate->push_new<array_operation>
		    (0, $1 - 1, std::move (args));
		}
|	TYPEOF_KEYWORD '(' Expression ')'
		{ pstate->wrap<typeof_operation> (); }
;

ArrayLiteral:
	'[' ArgumentList_opt ']'
		{ $$ = pstate->arglist_len; }
;

IdentifierExp:
	IDENTIFIER
;

StringExp:
	STRING_LITERAL
		{ /* We copy the string here, and not in the
		     lexer, to guarantee that we do not leak a
		     string.  Note that we follow the
		     NUL-termination convention of the
		     lexer.  */
		  struct typed_stoken *vec = XNEW (struct typed_stoken);
		  $$.len = 1;
		  $$.tokens = vec;

		  vec->type = $1.type;
		  vec->length = $1.length;
		  vec->ptr = (char *) malloc ($1.length + 1);
		  memcpy (vec->ptr, $1.ptr, $1.length + 1);
		}
|	StringExp STRING_LITERAL
		{ /* Note that we NUL-terminate here, but just
		     for convenience.  */
		  char *p;
		  ++$$.len;
		  $$.tokens
		    = XRESIZEVEC (struct typed_stoken, $$.tokens, $$.len);

		  p = (char *) malloc ($2.length + 1);
		  memcpy (p, $2.ptr, $2.length + 1);

		  $$.tokens[$$.len - 1].type = $2.type;
		  $$.tokens[$$.len - 1].length = $2.length;
		  $$.tokens[$$.len - 1].ptr = p;
		}
;

TypeExp:
	'(' TypeExp ')'
		{ /* Do nothing.  */ }
|	BasicType
		{ pstate->push_new<type_operation> ($1); }
|	BasicType BasicType2
		{ $$ = type_stack->follow_types ($1);
		  pstate->push_new<type_operation> ($$);
		}
;

BasicType2:
	'*'
		{ type_stack->push (tp_pointer); }
|	'*' BasicType2
		{ type_stack->push (tp_pointer); }
|	'[' INTEGER_LITERAL ']'
		{ type_stack->push (tp_array, $2.val); }
|	'[' INTEGER_LITERAL ']' BasicType2
		{ type_stack->push (tp_array, $2.val); }
;

BasicType:
	TYPENAME
		{ $$ = $1.type; }
;
