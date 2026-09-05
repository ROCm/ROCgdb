
/* YACC parser for Fortran expressions, for GDB.
   Copyright (C) 1986-2026 Free Software Foundation, Inc.

   Contributed by Motorola.  Adapted from the C parser by Farooq Butt
   (fmbutt@engage.sps.mot.com).

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

/* This was blantantly ripped off the C expression parser, please
   be aware of that as you look at its basic structure -FMB */

/* Parse a F77 expression from text in a string,
   and return the result as a  struct expression  pointer.
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
#include "f-lang.h"
#include "f-exp-parser.h"
#include "block.h"
#include <algorithm>
#include "type-stack.h"
#include "f-exp.h"

using namespace f_exp_parser;
using namespace expr;

/* Bring the f_exp_parser::type_stack global into this scope, so that it hides
   the struct type_stack type name.  */
using f_exp_parser::type_stack;
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
    } typed_val;
    struct {
      gdb_byte val[16];
      struct type *type;
    } typed_val_float;
    struct symbol *sym;
    struct type *tval;
    struct stoken sval;
    struct ttype tsym;
    struct symtoken ssym;
    int voidval;
    enum exp_opcode opcode;
    struct internalvar *ivar;

    struct type **tvec;
    int *ivec;
  }

%type <voidval> exp  type_exp start variable
%type <tval> type typebase
%type <tvec> nonempty_typelist
/* %type <bval> block */

/* Fancy type parsing.  */
%type <voidval> func_mod direct_abs_decl abs_decl
%type <tval> ptype

%token <typed_val> INT
%token <typed_val_float> FLOAT

/* Both NAME and TYPENAME tokens represent symbols in the input,
   and both convey their data as strings.
   But a TYPENAME is a string that happens to be defined as a typedef
   or builtin type name (such as int or char)
   and a NAME is any other symbol.
   Contexts where this distinction is not important can use the
   nonterminal "name", which matches either NAME or TYPENAME.  */

%token <sval> STRING_LITERAL
%token <lval> BOOLEAN_LITERAL
%token <ssym> NAME
%token <tsym> TYPENAME
%token <voidval> COMPLETE
%type <sval> name
%type <ssym> name_not_typename

/* A NAME_OR_INT is a symbol which is not known in the symbol table,
   but which would parse as a valid number in the current input radix.
   E.g. "c" when input_radix==16.  Depending on the parse, it will be
   turned into a name or into a number.  */

%token <ssym> NAME_OR_INT

%token SIZEOF KIND
%token ERROR

/* Special type cases, put in to allow the parser to distinguish different
   legal basetypes.  */
%token INT_S1_KEYWORD INT_S2_KEYWORD INT_KEYWORD INT_S4_KEYWORD INT_S8_KEYWORD
%token LOGICAL_S1_KEYWORD LOGICAL_S2_KEYWORD LOGICAL_KEYWORD LOGICAL_S4_KEYWORD
%token LOGICAL_S8_KEYWORD
%token REAL_KEYWORD REAL_S4_KEYWORD REAL_S8_KEYWORD REAL_S16_KEYWORD
%token COMPLEX_KEYWORD COMPLEX_S4_KEYWORD COMPLEX_S8_KEYWORD
%token COMPLEX_S16_KEYWORD
%token BOOL_AND BOOL_OR BOOL_NOT
%token SINGLE DOUBLE PRECISION
%token <lval> CHARACTER

%token <sval> DOLLAR_VARIABLE

%token <opcode> ASSIGN_MODIFY
%token <opcode> UNOP_INTRINSIC BINOP_INTRINSIC
%token <opcode> UNOP_OR_BINOP_INTRINSIC UNOP_OR_BINOP_OR_TERNOP_INTRINSIC

%left ','
%left ABOVE_COMMA
%right '=' ASSIGN_MODIFY
%right '?'
%left BOOL_OR
%right BOOL_NOT
%left BOOL_AND
%left '|'
%left '^'
%left '&'
%left EQUAL NOTEQUAL
%left LESSTHAN GREATERTHAN LEQ GEQ
%left LSH RSH
%left '@'
%left '+' '-'
%left '*' '/'
%right STARSTAR
%right '%'
%right UNARY
%right '('


%%

start   :	exp
	|	type_exp
	;

type_exp:	type
			{ pstate->push_new<type_operation> ($1); }
	;

exp     :       '(' exp ')'
			{ }
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

exp	:	BOOL_NOT exp    %prec UNARY
			{ pstate->wrap<unary_logical_not_operation> (); }
	;

exp	:	'~' exp    %prec UNARY
			{ pstate->wrap<unary_complement_operation> (); }
	;

exp	:	SIZEOF exp       %prec UNARY
			{ pstate->wrap<unop_sizeof_operation> (); }
	;

exp	:	KIND '(' exp ')'       %prec UNARY
			{ pstate->wrap<fortran_kind_operation> (); }
	;

/* No more explicit array operators, we treat everything in F77 as
   a function call.  The disambiguation as to whether we are
   doing a subscript operation or a function call is done
   later in eval.c.  */

exp	:	exp '('
			{ pstate->start_arglist (); }
		arglist ')'
			{
			  std::vector<operation_up> args
			    = pstate->pop_vector (pstate->end_arglist ());
			  pstate->push_new<fortran_undetermined>
			    (pstate->pop (), std::move (args));
			}
	;

exp	:	UNOP_INTRINSIC '(' exp ')'
			{
			  wrap_unop_intrinsic ($1);
			}
	;

exp	:	BINOP_INTRINSIC '(' exp ',' exp ')'
			{
			  wrap_binop_intrinsic ($1);
			}
	;

exp	:	UNOP_OR_BINOP_INTRINSIC '('
			{ pstate->start_arglist (); }
		arglist ')'
			{
			  const int n = pstate->end_arglist ();

			  switch (n)
			    {
			    case 1:
			      wrap_unop_intrinsic ($1);
			      break;
			    case 2:
			      wrap_binop_intrinsic ($1);
			      break;
			    default:
			      gdb_assert_not_reached
				("wrong number of arguments for intrinsics");
			    }
			}

exp	:	UNOP_OR_BINOP_OR_TERNOP_INTRINSIC '('
			{ pstate->start_arglist (); }
		arglist ')'
			{
			  const int n = pstate->end_arglist ();

			  switch (n)
			    {
			    case 1:
			      wrap_unop_intrinsic ($1);
			      break;
			    case 2:
			      wrap_binop_intrinsic ($1);
			      break;
			    case 3:
			      wrap_ternop_intrinsic ($1);
			      break;
			    default:
			      gdb_assert_not_reached
				("wrong number of arguments for intrinsics");
			    }
			}
	;

arglist	:
	;

arglist	:	exp
			{ pstate->arglist_len = 1; }
	;

arglist :	subrange
			{ pstate->arglist_len = 1; }
	;

arglist	:	arglist ',' exp   %prec ABOVE_COMMA
			{ pstate->arglist_len++; }
	;

arglist	:	arglist ',' subrange   %prec ABOVE_COMMA
			{ pstate->arglist_len++; }
	;

/* There are four sorts of subrange types in F90.  */

subrange:	exp ':' exp	%prec ABOVE_COMMA
			{
			  operation_up high = pstate->pop ();
			  operation_up low = pstate->pop ();
			  pstate->push_new<fortran_range_operation>
			    (RANGE_STANDARD, std::move (low),
			     std::move (high), operation_up ());
			}
	;

subrange:	exp ':'	%prec ABOVE_COMMA
			{
			  operation_up low = pstate->pop ();
			  pstate->push_new<fortran_range_operation>
			    (RANGE_HIGH_BOUND_DEFAULT, std::move (low),
			     operation_up (), operation_up ());
			}
	;

subrange:	':' exp	%prec ABOVE_COMMA
			{
			  operation_up high = pstate->pop ();
			  pstate->push_new<fortran_range_operation>
			    (RANGE_LOW_BOUND_DEFAULT, operation_up (),
			     std::move (high), operation_up ());
			}
	;

subrange:	':'	%prec ABOVE_COMMA
			{
			  pstate->push_new<fortran_range_operation>
			    (RANGE_LOW_BOUND_DEFAULT
			     | RANGE_HIGH_BOUND_DEFAULT,
			     operation_up (), operation_up (),
			     operation_up ());
			}
	;

/* And each of the four subrange types can also have a stride.  */
subrange:	exp ':' exp ':' exp	%prec ABOVE_COMMA
			{
			  operation_up stride = pstate->pop ();
			  operation_up high = pstate->pop ();
			  operation_up low = pstate->pop ();
			  pstate->push_new<fortran_range_operation>
			    (RANGE_STANDARD | RANGE_HAS_STRIDE,
			     std::move (low), std::move (high),
			     std::move (stride));
			}
	;

subrange:	exp ':' ':' exp	%prec ABOVE_COMMA
			{
			  operation_up stride = pstate->pop ();
			  operation_up low = pstate->pop ();
			  pstate->push_new<fortran_range_operation>
			    (RANGE_HIGH_BOUND_DEFAULT
			     | RANGE_HAS_STRIDE,
			     std::move (low), operation_up (),
			     std::move (stride));
			}
	;

subrange:	':' exp ':' exp	%prec ABOVE_COMMA
			{
			  operation_up stride = pstate->pop ();
			  operation_up high = pstate->pop ();
			  pstate->push_new<fortran_range_operation>
			    (RANGE_LOW_BOUND_DEFAULT
			     | RANGE_HAS_STRIDE,
			     operation_up (), std::move (high),
			     std::move (stride));
			}
	;

subrange:	':' ':' exp	%prec ABOVE_COMMA
			{
			  operation_up stride = pstate->pop ();
			  pstate->push_new<fortran_range_operation>
			    (RANGE_LOW_BOUND_DEFAULT
			     | RANGE_HIGH_BOUND_DEFAULT
			     | RANGE_HAS_STRIDE,
			     operation_up (), operation_up (),
			     std::move (stride));
			}
	;

complexnum:     exp ',' exp
			{ }
	;

exp	:	'(' complexnum ')'
			{
			  operation_up rhs = pstate->pop ();
			  operation_up lhs = pstate->pop ();
			  pstate->push_new<complex_operation>
			    (std::move (lhs), std::move (rhs),
			     parse_f_type (pstate)->builtin_complex_s16);
			}
	;

exp	:	'(' type ')' exp  %prec UNARY
			{
			  pstate->push_new<unop_cast_operation>
			    (pstate->pop (), $2);
			}
	;

exp     :       exp '%' name
			{
			  pstate->push_new<fortran_structop_operation>
			    (pstate->pop (), copy_name ($3));
			}
	;

exp     :       exp '%' name COMPLETE
			{
			  structop_base_operation *op
			    = new fortran_structop_operation (pstate->pop (),
							      copy_name ($3));
			  pstate->mark_struct_expression (op);
			  pstate->push (operation_up (op));
			}
	;

exp     :       exp '%' COMPLETE
			{
			  structop_base_operation *op
			    = new fortran_structop_operation (pstate->pop (),
							      "");
			  pstate->mark_struct_expression (op);
			  pstate->push (operation_up (op));
			}
	;

/* Binary operators in order of decreasing precedence.  */

exp	:	exp '@' exp
			{ pstate->wrap2<repeat_operation> (); }
	;

exp	:	exp STARSTAR exp
			{ pstate->wrap2<exp_operation> (); }
	;

exp	:	exp '*' exp
			{ pstate->wrap2<mul_operation> (); }
	;

exp	:	exp '/' exp
			{ pstate->wrap2<div_operation> (); }
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

exp	:	exp LESSTHAN exp
			{ pstate->wrap2<less_operation> (); }
	;

exp	:	exp GREATERTHAN exp
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

exp     :       exp BOOL_AND exp
			{ pstate->wrap2<logical_and_operation> (); }
	;


exp	:	exp BOOL_OR exp
			{ pstate->wrap2<logical_or_operation> (); }
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

exp	:	NAME_OR_INT
			{ YYSTYPE val;
			  parse_number (pstate, $1.stoken.ptr,
					$1.stoken.length, 0, &val);
			  pstate->push_new<long_const_operation>
			    (val.typed_val.type,
			     val.typed_val.val);
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
			{ pstate->push_dollar ($1); }
	;

exp	:	SIZEOF '(' type ')'	%prec UNARY
			{
			  $3 = check_typedef ($3);
			  pstate->push_new<long_const_operation>
			    (parse_f_type (pstate)->builtin_integer,
			     $3->length ());
			}
	;

exp     :       BOOLEAN_LITERAL
			{ pstate->push_new<bool_operation> ($1); }
	;

exp	:	STRING_LITERAL
			{
			  pstate->push_new<string_operation>
			    (copy_name ($1));
			}
	;

variable:	name_not_typename
			{ struct block_symbol sym = $1.sym;
			  std::string name = copy_name ($1.stoken);
			  pstate->push_symbol (name.c_str (), sym);
			}
	;


type    :       ptype
	;

ptype	:	typebase
	|	typebase abs_decl
		{
		  /* This is where the interesting stuff happens.  */
		  int done = 0;
		  int array_size;
		  struct type *follow_type = $1;
		  struct type *range_type;

		  while (!done)
		    switch (type_stack->pop ())
		      {
		      case tp_end:
			done = 1;
			break;
		      case tp_pointer:
			follow_type = lookup_pointer_type (follow_type);
			break;
		      case tp_reference:
			follow_type = lookup_lvalue_reference_type (follow_type);
			break;
		      case tp_array:
			array_size = type_stack->pop_int ();
			if (array_size != -1)
			  {
			    struct type *idx_type
			      = parse_f_type (pstate)->builtin_integer;
			    type_allocator alloc (idx_type);
			    range_type =
			      create_static_range_type (alloc, idx_type,
							0, array_size - 1);
			    follow_type = create_array_type (alloc,
							     follow_type,
							     range_type);
			  }
			else
			  follow_type = lookup_pointer_type (follow_type);
			break;
		      case tp_function:
			follow_type = lookup_function_type (follow_type);
			break;
		      case tp_kind:
			{
			  int kind_val = type_stack->pop_int ();
			  follow_type
			    = convert_to_kind_type (follow_type, kind_val);
			}
			break;
		      }
		  $$ = follow_type;
		}
	;

abs_decl:	'*'
			{ type_stack->push (tp_pointer); $$ = 0; }
	|	'*' abs_decl
			{ type_stack->push (tp_pointer); $$ = $2; }
	|	'&'
			{ type_stack->push (tp_reference); $$ = 0; }
	|	'&' abs_decl
			{ type_stack->push (tp_reference); $$ = $2; }
	|	direct_abs_decl
	;

direct_abs_decl: '(' abs_decl ')'
			{ $$ = $2; }
	| 	'(' KIND '=' INT ')'
			{ push_kind_type ($4.val, $4.type); }
	|	'*' INT
			{ push_kind_type ($2.val, $2.type); }
	| 	direct_abs_decl func_mod
			{ type_stack->push (tp_function); }
	|	func_mod
			{ type_stack->push (tp_function); }
	;

func_mod:	'(' ')'
			{ $$ = 0; }
	|	'(' nonempty_typelist ')'
			{ free ($2); $$ = 0; }
	;

typebase  /* Implements (approximately): (type-qualifier)* type-specifier */
	:	TYPENAME
			{ $$ = $1.type; }
	|	INT_S1_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_integer_s1; }
	|	INT_S2_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_integer_s2; }
	|	INT_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_integer; }
	|	INT_S4_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_integer; }
	|	INT_S8_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_integer_s8; }
	|	CHARACTER
			{ $$ = parse_f_type (pstate)->builtin_character; }
	|	LOGICAL_S1_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_logical_s1; }
	|	LOGICAL_S2_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_logical_s2; }
	|	LOGICAL_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_logical; }
	|	LOGICAL_S4_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_logical; }
	|	LOGICAL_S8_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_logical_s8; }
	|	REAL_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_real; }
	|	REAL_S4_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_real; }
	|       REAL_S8_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_real_s8; }
	|	REAL_S16_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_real_s16;
			  if ($$->code () == TYPE_CODE_ERROR)
			    error (_("unsupported type %s"),
				   $$->safe_name ());
			}
	|	COMPLEX_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_complex; }
	|	COMPLEX_S4_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_complex; }
	|	COMPLEX_S8_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_complex_s8; }
	|	COMPLEX_S16_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_complex_s16;
			  if ($$->code () == TYPE_CODE_ERROR)
			    error (_("unsupported type %s"),
				   $$->safe_name ());
			}
	|	SINGLE PRECISION
			{ $$ = parse_f_type (pstate)->builtin_real;}
	|	DOUBLE PRECISION
			{ $$ = parse_f_type (pstate)->builtin_real_s8;}
	|	SINGLE COMPLEX_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_complex;}
	|	DOUBLE COMPLEX_KEYWORD
			{ $$ = parse_f_type (pstate)->builtin_complex_s8;}
	;

nonempty_typelist
	:	type
		{ $$ = (struct type **) malloc (sizeof (struct type *) * 2);
		  $<ivec>$[0] = 1;	/* Number of types in vector */
		  $$[1] = $1;
		}
	|	nonempty_typelist ',' type
		{ int len = sizeof (struct type *) * (++($<ivec>1[0]) + 1);
		  $$ = (struct type **) realloc ((char *) $1, len);
		  $$[$<ivec>$[0]] = $3;
		}
	;

name
	:	NAME
		{ $$ = $1.stoken; }
	|	TYPENAME
		{ $$ = $1.stoken; }
	;

name_not_typename :	NAME
/* These would be useful if name_not_typename was useful, but it is just
   a fake for "variable", so these cause reduce/reduce conflicts because
   the parser can't tell whether NAME_OR_INT is a name_not_typename (=variable,
   =exp) or just an exp.  If name_not_typename was ever used in an lvalue
   context where only a name could occur, this might be useful.
	|	NAME_OR_INT
   */
	;
