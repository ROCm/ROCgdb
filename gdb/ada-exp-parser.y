/* YACC parser for Ada expressions, for GDB.
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

/* Parse an Ada expression from text in a string,
   and return the result as a  struct expression  pointer.
   That structure contains arithmetic operations in reverse polish,
   with constants represented by operations that are followed by special data.
   See expression.h for the details of the format.
   What is important here is that it can be built up sequentially
   during the process of parsing; the lower levels of the tree always
   come first in the result.

   malloc's and realloc's in this file are transformed to
   xmalloc and xrealloc respectively by the same sed command in the
   makefile that remaps any other malloc/realloc inserted by the parser
   generator.  Doing this with #defines and trying to control the interaction
   with include files (<malloc.h> and <stdlib.h> for example) just became
   too messy, particularly when such includes can be inserted at random
   times by the parser generator.  */

%{

#include "gdbsupport/unordered_map.h"
#include "expression.h"
#include "value.h"
#include "parser-defs.h"
#include "language.h"
#include "ada-lang.h"
#include "frame.h"
#include "block.h"
#include "ada-exp.h"
#include "ada-exp-parser.h"
#include "cli/cli-style.h"

using namespace expr;
using namespace ada_exp_parser;

static int yylex (void);

%}

%union
  {
    LONGEST lval;
    struct {
      const gdb_mpz *val;
      struct type *type;
    } typed_val;
    struct {
      LONGEST val;
      struct type *type;
    } typed_char;
    struct {
      gdb_byte val[16];
      struct type *type;
    } typed_val_float;
    struct type *tval;
    struct stoken sval;
    const struct block *bval;
    struct internalvar *ivar;
  }

%type <lval> positional_list component_groups component_associations
%type <lval> aggregate_component_list
%type <tval> var_or_type type_prefix opt_type_prefix

%token <typed_val> INT NULL_PTR
%token <typed_char> CHARLIT
%token <typed_val_float> FLOAT
%token TRUEKEYWORD FALSEKEYWORD
%token WITH DELTA
%token COLONCOLON
%token <sval> STRING NAME DOT_ID TICK_COMPLETE DOT_COMPLETE NAME_COMPLETE
%type <bval> block
%type <lval> arglist tick_arglist

/* Special type cases, put in to allow the parser to distinguish different
   legal basetypes.  */
%token <sval> DOLLAR_VARIABLE

%nonassoc ASSIGN
%left _AND_ OR XOR THEN ELSE
%left '=' NOTEQUAL '<' '>' LEQ GEQ IN DOTDOT
%left '@'
%left '+' '-' '&'
%left UNARY
%left '*' '/' MOD REM
%right STARSTAR ABS NOT

/* Artificial token to give NAME => ... and NAME | priority over reducing
   NAME to <primary> and to give <primary>' priority over reducing <primary>
   to <simple_exp>. */
%nonassoc VAR

%nonassoc ARROW '|'

%right TICK_ACCESS TICK_ADDRESS TICK_FIRST TICK_LAST TICK_LENGTH
%right TICK_MAX TICK_MIN TICK_MODULUS
%right TICK_POS TICK_RANGE TICK_SIZE TICK_OBJECT_SIZE TICK_TAG TICK_VAL
%right TICK_COMPLETE TICK_ENUM_REP TICK_ENUM_VAL
 /* The following are right-associative only so that reductions at this
    precedence have lower precedence than '.' and '('.  The syntax still
    forces a.b.c, e.g., to be LEFT-associated.  */
%right '.' '(' '[' DOT_ID DOT_COMPLETE

%token NEW OTHERS FOR


%%

start   :	exp1
	;

/* Expressions, including the sequencing operator.  */
exp1	:	exp
	|	exp1 ';' exp
			{ ada_wrap2<comma_operation> (BINOP_COMMA); }
	| 	primary ASSIGN
			{
			  ada_parser->assignments.emplace_back
			    (new ada_assign_operation (ada_pop (), nullptr));
			}
		exp   /* Extension for convenience */
			{
			  ada_assign_up assign
			    = std::move (ada_parser->assignments.back ());
			  ada_parser->assignments.pop_back ();
			  value *lhs_val = (assign->eval_for_resolution
					    (pstate->expout.get ()));

			  operation_up rhs = pstate->pop ();
			  rhs = resolve (std::move (rhs), true,
					 lhs_val->type ());

			  assign->set_rhs (std::move (rhs));
			  pstate->push (std::move (assign));
			}
	;

/* Expressions, not including the sequencing operator.  */

primary :	primary DOT_ID
			{
			  if (streq ($2.ptr, "all"))
			    ada_wrap<ada_unop_ind_operation> ();
			  else
			    {
			      operation_up arg = ada_pop ();
			      pstate->push_new<ada_structop_operation>
				(std::move (arg), copy_name ($2));
			    }
			}
	;

primary :	primary DOT_COMPLETE
			{
			  /* This is done even for ".all", because
			     that might be a prefix.  */
			  operation_up arg = ada_pop ();
			  ada_structop_operation *str_op
			    = (new ada_structop_operation
			       (std::move (arg), copy_name ($2)));
			  str_op->set_prefix (ada_parser->find_completion_bounds ());
			  pstate->push (operation_up (str_op));
			  pstate->mark_struct_expression (str_op);
			}
	;

primary :	primary '(' arglist ')'
			{ ada_funcall ($3); }
	|	var_or_type '(' arglist ')'
			{
			  if ($1 != NULL)
			    {
			      if ($3 != 1)
				error (_("Invalid conversion"));
			      operation_up arg = ada_pop ();
			      pstate->push_new<unop_cast_operation>
				(std::move (arg), $1);
			    }
			  else
			    ada_funcall ($3);
			}
	;

primary :	var_or_type '\'' '(' exp ')'
			{
			  if ($1 == NULL)
			    error (_("Type required for qualification"));
			  operation_up arg = ada_pop (true,
						      check_typedef ($1));
			  pstate->push_new<ada_qual_operation>
			    (std::move (arg), $1);
			}
	;

primary :
		primary '(' simple_exp DOTDOT simple_exp ')'
			{ ada_wrap3<ada_ternop_slice_operation> (); }
	|	var_or_type '(' simple_exp DOTDOT simple_exp ')'
			{ if ($1 == NULL)
			    ada_wrap3<ada_ternop_slice_operation> ();
			  else
			    error (_("Cannot slice a type"));
			}
	;

primary :	'(' exp1 ')'	{ }
	;

/* The following rule causes a conflict with the type conversion
       var_or_type (exp)
   To get around it, we give '(' higher priority and add bridge rules for
       var_or_type (exp, exp, ...)
       var_or_type (exp .. exp)
   We also have the action for  var_or_type(exp) generate a function call
   when the first symbol does not denote a type. */

primary :	var_or_type	%prec VAR
			{ if ($1 != NULL)
			    pstate->push_new<type_operation> ($1);
			}
	;

primary :	DOLLAR_VARIABLE /* Various GDB extensions */
			{ pstate->push_dollar ($1); }
	;

primary :     	aggregate
			{
			  pstate->push_new<ada_aggregate_operation>
			    (pop_component ());
			}
	;

primary :	'@'
			{
			  if (ada_parser->assignments.empty ())
			    error (_("the target name symbol ('@') may only "
				     "appear in an assignment context"));
			  ada_assign_operation *current
			    = ada_parser->assignments.back ().get ();
			  pstate->push_new<ada_target_operation> (current);
			}
	;

simple_exp : 	primary
	;

simple_exp :	'-' simple_exp    %prec UNARY
			{ ada_wrap_overload<ada_neg_operation> (UNOP_NEG); }
	;

simple_exp :	'+' simple_exp    %prec UNARY
			{
			  operation_up arg = ada_pop ();
			  operation_up empty;

			  /* If an overloaded operator was found, use
			     it.  Otherwise, unary + has no effect and
			     the argument can be pushed instead.  */
			  operation_up call = maybe_overload (UNOP_PLUS, arg,
							      empty);
			  if (call != nullptr)
			    arg = std::move (call);
			  pstate->push (std::move (arg));
			}
	;

simple_exp :	NOT simple_exp    %prec UNARY
			{
			  ada_wrap_overload<unary_logical_not_operation>
			    (UNOP_LOGICAL_NOT);
			}
	;

simple_exp :    ABS simple_exp	   %prec UNARY
			{ ada_wrap_overload<ada_abs_operation> (UNOP_ABS); }
	;

arglist	:		{ $$ = 0; }
	;

arglist	:	exp
			{ $$ = 1; }
	|	NAME ARROW exp
			{ $$ = 1; }
	|	arglist ',' exp
			{ $$ = $1 + 1; }
	|	arglist ',' NAME ARROW exp
			{ $$ = $1 + 1; }
	;

primary :	'{' var_or_type '}' primary  %prec '.'
		/* GDB extension */
			{
			  if ($2 == NULL)
			    error (_("Type required within braces in coercion"));
			  operation_up arg = ada_pop ();
			  pstate->push_new<unop_memval_operation>
			    (std::move (arg), $2);
			}
	;

/* Binary operators in order of decreasing precedence.  */

simple_exp 	: 	simple_exp STARSTAR simple_exp
			{ ada_wrap2<ada_binop_exp_operation> (BINOP_EXP); }
	;

simple_exp	:	simple_exp '*' simple_exp
			{ ada_wrap2<ada_binop_mul_operation> (BINOP_MUL); }
	;

simple_exp	:	simple_exp '/' simple_exp
			{ ada_wrap2<ada_binop_div_operation> (BINOP_DIV); }
	;

simple_exp	:	simple_exp REM simple_exp /* May need to be fixed to give correct Ada REM */
			{ ada_wrap2<ada_binop_rem_operation> (BINOP_REM); }
	;

simple_exp	:	simple_exp MOD simple_exp
			{ ada_wrap2<ada_binop_mod_operation> (BINOP_MOD); }
	;

simple_exp	:	simple_exp '@' simple_exp	/* GDB extension */
			{ ada_wrap2<repeat_operation> (BINOP_REPEAT); }
	;

simple_exp	:	simple_exp '+' simple_exp
			{ ada_wrap_op<ada_binop_addsub_operation> (BINOP_ADD); }
	;

simple_exp	:	simple_exp '&' simple_exp
			{ ada_wrap2<ada_concat_operation> (BINOP_CONCAT); }
	;

simple_exp	:	simple_exp '-' simple_exp
			{ ada_wrap_op<ada_binop_addsub_operation> (BINOP_SUB); }
	;

relation :	simple_exp
	;

relation :	simple_exp '=' simple_exp
			{ ada_wrap_op<ada_binop_equal_operation> (BINOP_EQUAL); }
	;

relation :	simple_exp NOTEQUAL simple_exp
			{ ada_wrap_op<ada_binop_equal_operation> (BINOP_NOTEQUAL); }
	;

relation :	simple_exp LEQ simple_exp
			{ ada_un_wrap2<leq_operation> (BINOP_LEQ); }
	;

relation :	simple_exp IN simple_exp DOTDOT simple_exp
			{ ada_wrap3<ada_ternop_range_operation> (); }
	|       simple_exp IN primary TICK_RANGE tick_arglist
			{
			  operation_up rhs = ada_pop ();
			  operation_up lhs = ada_pop ();
			  pstate->push_new<ada_binop_in_bounds_operation>
			    (std::move (lhs), std::move (rhs), $5);
			}
	|	simple_exp IN var_or_type	%prec TICK_ACCESS
			{
			  if ($3 == NULL)
			    error (_("Right operand of 'in' must be type"));
			  operation_up arg = ada_pop ();
			  pstate->push_new<ada_unop_range_operation>
			    (std::move (arg), $3);
			}
	|	simple_exp NOT IN simple_exp DOTDOT simple_exp
			{ ada_wrap3<ada_ternop_range_operation> ();
			  ada_wrap<unary_logical_not_operation> (); }
	|       simple_exp NOT IN primary TICK_RANGE tick_arglist
			{
			  operation_up rhs = ada_pop ();
			  operation_up lhs = ada_pop ();
			  pstate->push_new<ada_binop_in_bounds_operation>
			    (std::move (lhs), std::move (rhs), $6);
			  ada_wrap<unary_logical_not_operation> ();
			}
	|	simple_exp NOT IN var_or_type	%prec TICK_ACCESS
			{
			  if ($4 == NULL)
			    error (_("Right operand of 'in' must be type"));
			  operation_up arg = ada_pop ();
			  pstate->push_new<ada_unop_range_operation>
			    (std::move (arg), $4);
			  ada_wrap<unary_logical_not_operation> ();
			}
	;

relation :	simple_exp GEQ simple_exp
			{ ada_un_wrap2<geq_operation> (BINOP_GEQ); }
	;

relation :	simple_exp '<' simple_exp
			{ ada_un_wrap2<less_operation> (BINOP_LESS); }
	;

relation :	simple_exp '>' simple_exp
			{ ada_un_wrap2<gtr_operation> (BINOP_GTR); }
	;

exp	:	relation
	|	and_exp
	|	and_then_exp
	|	or_exp
	|	or_else_exp
	|	xor_exp
	;

and_exp :
		relation _AND_ relation
			{ ada_wrap2<bitwise_and_operation>
			    (BINOP_BITWISE_AND); }
	|	and_exp _AND_ relation
			{ ada_wrap2<bitwise_and_operation>
			    (BINOP_BITWISE_AND); }
	;

and_then_exp :
	       relation _AND_ THEN relation
			{ ada_wrap2<logical_and_operation>
			    (BINOP_LOGICAL_AND); }
	|	and_then_exp _AND_ THEN relation
			{ ada_wrap2<logical_and_operation>
			    (BINOP_LOGICAL_AND); }
	;

or_exp :
		relation OR relation
			{ ada_wrap2<bitwise_ior_operation>
			    (BINOP_BITWISE_IOR); }
	|	or_exp OR relation
			{ ada_wrap2<bitwise_ior_operation>
			    (BINOP_BITWISE_IOR); }
	;

or_else_exp :
	       relation OR ELSE relation
			{ ada_wrap2<logical_or_operation> (BINOP_LOGICAL_OR); }
	|      or_else_exp OR ELSE relation
			{ ada_wrap2<logical_or_operation> (BINOP_LOGICAL_OR); }
	;

xor_exp :       relation XOR relation
			{ ada_wrap2<bitwise_xor_operation>
			    (BINOP_BITWISE_XOR); }
	|	xor_exp XOR relation
			{ ada_wrap2<bitwise_xor_operation>
			    (BINOP_BITWISE_XOR); }
	;

/* Primaries can denote types (OP_TYPE).  In cases such as
   primary TICK_ADDRESS, where a type would be invalid, it will be
   caught when evaluate_subexp in ada-lang.c tries to evaluate the
   primary, expecting a value.  Precedence rules resolve the ambiguity
   in NAME TICK_ACCESS in favor of shifting to form a var_or_type.  A
   construct such as aType'access'access will again cause an error when
   aType'access evaluates to a type that evaluate_subexp attempts to
   evaluate. */
primary :	primary TICK_ACCESS
			{ ada_addrof (); }
	|	primary TICK_ADDRESS
			{ ada_addrof (type_system_address (pstate)); }
	|	primary TICK_COMPLETE
			{
			  pstate->mark_completion (make_tick_completer ($2));
			}
	|	primary TICK_FIRST tick_arglist
			{
			  operation_up arg = ada_pop ();
			  pstate->push_new<ada_unop_atr_operation>
			    (std::move (arg), OP_ATR_FIRST, $3);
			}
	|	primary TICK_LAST tick_arglist
			{
			  operation_up arg = ada_pop ();
			  pstate->push_new<ada_unop_atr_operation>
			    (std::move (arg), OP_ATR_LAST, $3);
			}
	| 	primary TICK_LENGTH tick_arglist
			{
			  operation_up arg = ada_pop ();
			  pstate->push_new<ada_unop_atr_operation>
			    (std::move (arg), OP_ATR_LENGTH, $3);
			}
	|       primary TICK_SIZE
			{ ada_wrap<ada_atr_size_operation> (true); }
	|       primary TICK_OBJECT_SIZE
			{ ada_wrap<ada_atr_size_operation> (false); }
	|	primary TICK_TAG
			{ ada_wrap<ada_atr_tag_operation> (); }
	|       opt_type_prefix TICK_MIN '(' exp ',' exp ')'
			{ ada_wrap2<ada_binop_min_operation> (BINOP_MIN); }
	|       opt_type_prefix TICK_MAX '(' exp ',' exp ')'
			{ ada_wrap2<ada_binop_max_operation> (BINOP_MAX); }
	| 	opt_type_prefix TICK_POS '(' exp ')'
			{ ada_wrap<ada_pos_operation> (); }
	|	type_prefix TICK_VAL '(' exp ')'
			{
			  operation_up arg = ada_pop ();
			  pstate->push_new<ada_atr_val_operation>
			    ($1, std::move (arg));
			}
	|	type_prefix TICK_ENUM_REP '(' exp ')'
			{
			  operation_up arg = ada_pop (true, $1);
			  pstate->push_new<ada_atr_enum_rep_operation>
			    ($1, std::move (arg));
			}
	|	type_prefix TICK_ENUM_VAL '(' exp ')'
			{
			  operation_up arg = ada_pop (true, $1);
			  pstate->push_new<ada_atr_enum_val_operation>
			    ($1, std::move (arg));
			}
	|	type_prefix TICK_MODULUS
			{
			  struct type *type_arg = check_typedef ($1);
			  if (!ada_is_modular_type (type_arg))
			    error (_("'modulus must be applied to modular type"));
			  std::optional<ULONGEST> bound
			    = ada_modular_bound (type_arg);
			  if (!bound.has_value ())
			    error (_("'modulus applied to type with non-constant bound"));
			  gdb_mpz modulus = gdb_mpz (*bound) + 1;
			  /* Pick something that's almost certainly
			     large enough.  */
			  struct type *r_type
			    = language_lookup_primitive_type (pstate->language (),
							      pstate->gdbarch (),
							      "unsigned_long_long_long_integer");
			  pstate->push_new<long_const_operation> (r_type,
								  modulus);
			  ada_wrap<ada_wrapped_operation> ();
			}
	;

tick_arglist :			%prec '('
			{ $$ = 1; }
	| 	'(' INT ')'
			{ $$ = $2.val->as_integer<LONGEST> (); }
	;

type_prefix :
		var_or_type
			{
			  if ($1 == NULL)
			    error (_("Prefix must be type"));
			  $$ = $1;
			}
	;

opt_type_prefix :
		type_prefix
			{ $$ = $1; }
	| 	/* EMPTY */
			{ $$ = parse_type (pstate)->builtin_void; }
	;


primary	:	INT
			{
			  pstate->push_new<long_const_operation> ($1.type, *$1.val);
			  ada_wrap<ada_wrapped_operation> ();
			}
	;

primary	:	CHARLIT
			{
			  pstate->push_new<ada_char_operation> ($1.type, $1.val);
			}
	;

primary	:	FLOAT
			{
			  float_data data;
			  std::copy (std::begin ($1.val), std::end ($1.val),
				     std::begin (data));
			  pstate->push_new<float_const_operation>
			    ($1.type, data);
			  ada_wrap<ada_wrapped_operation> ();
			}
	;

primary	:	NULL_PTR
			{
			  struct type *null_ptr_type
			    = lookup_pointer_type (parse_type (pstate)->builtin_int0);
			  write_int (pstate, 0, null_ptr_type);
			}
	;

primary	:	STRING
			{
			  pstate->push_new<ada_string_operation>
			    (copy_name ($1));
			}
	;

primary :	TRUEKEYWORD
			{
			  write_int (pstate, 1,
				     parse_type (pstate)->builtin_bool);
			}
	|	FALSEKEYWORD
			{
			  write_int (pstate, 0,
				     parse_type (pstate)->builtin_bool);
			}
	;

primary	: 	NEW NAME
			{ error (_("NEW not implemented.")); }
	;

var_or_type:	NAME   	    %prec VAR
				{ $$ = write_var_or_type (pstate, NULL, $1); }
	|	NAME_COMPLETE %prec VAR
				{
				  $$ = write_var_or_type_completion (pstate,
								     NULL,
								     $1);
				}
	|	block NAME  %prec VAR
				{ $$ = write_var_or_type (pstate, $1, $2); }
	|	block NAME_COMPLETE  %prec VAR
				{
				  $$ = write_var_or_type_completion (pstate,
								     $1,
								     $2);
				}
	|       NAME TICK_ACCESS
			{
			  $$ = write_var_or_type (pstate, NULL, $1);
			  if ($$ == NULL)
			    ada_addrof ();
			  else
			    $$ = lookup_pointer_type ($$);
			}
	|	block NAME TICK_ACCESS
			{
			  $$ = write_var_or_type (pstate, $1, $2);
			  if ($$ == NULL)
			    ada_addrof ();
			  else
			    $$ = lookup_pointer_type ($$);
			}
	;

/* GDB extension */
block   :       NAME COLONCOLON
			{ $$ = block_lookup (NULL, $1.ptr); }
	|	block NAME COLONCOLON
			{ $$ = block_lookup ($1, $2.ptr); }
	;

aggregate :
		'(' exp WITH DELTA aggregate_component_list ')'
			{
			  std::vector<ada_component_up> components
			    = pop_components ($5);
			  operation_up base = ada_pop ();

			  push_component<ada_aggregate_component>
			    (std::move (base), std::move (components));
			}
	|	'(' aggregate_component_list ')'
			{
			  std::vector<ada_component_up> components
			    = pop_components ($2);

			  push_component<ada_aggregate_component>
			    (std::move (components));
			}
	;

aggregate_component_list :
		component_groups	 { $$ = $1; }
	|	positional_list exp
			{
			  push_component<ada_positional_component>
			    ($1, ada_pop ());
			  $$ = $1 + 1;
			}
	|	positional_list component_groups
					 { $$ = $1 + $2; }
	;

positional_list :
		exp ','
			{
			  push_component<ada_positional_component>
			    (0, ada_pop ());
			  $$ = 1;
			}
	|	positional_list exp ','
			{
			  push_component<ada_positional_component>
			    ($1, ada_pop ());
			  $$ = $1 + 1;
			}
	;

component_groups:
		others			 { $$ = 1; }
	|	component_group		 { $$ = 1; }
	|	component_group ',' component_groups
					 { $$ = $3 + 1; }
	;

others 	:	OTHERS ARROW exp
			{
			  push_component<ada_others_component> (ada_pop ());
			}
	;

component_group :
		component_associations
			{
			  ada_choices_component *choices = choice_component ();
			  choices->set_associations (pop_associations ($1));
			}
	|	FOR NAME IN
			{
			  std::string name = copy_name ($2);

			  auto iter = ada_parser->iterated_associations.find (name);
			  if (iter != ada_parser->iterated_associations.end ())
			    error (_("Nested use of index parameter '%s'"),
				   name.c_str ());

			  ada_parser->iterated_associations[name] = {};
			}
		component_associations
			{
			  std::string name = copy_name ($2);

			  ada_choices_component *choices = choice_component ();
			  choices->set_associations (pop_associations ($5));

			  auto iter = ada_parser->iterated_associations.find (name);
			  gdb_assert (iter != ada_parser->iterated_associations.end ());
			  for (ada_index_var_operation *var : iter->second)
			    var->set_choices (choices);

			  ada_parser->iterated_associations.erase (name);

			  choices->set_name (std::move (name));
			}
	;

/* We use this somewhat obscure definition in order to handle NAME => and
   NAME | differently from exp => and exp |.  ARROW and '|' have a precedence
   above that of the reduction of NAME to var_or_type.  By delaying
   decisions until after the => or '|', we convert the ambiguity to a
   resolved shift/reduce conflict. */
component_associations :
		NAME ARROW exp
			{
			  push_component<ada_choices_component> (ada_pop ());
			  write_name_assoc (pstate, $1);
			  $$ = 1;
			}
	|	simple_exp ARROW exp
			{
			  push_component<ada_choices_component> (ada_pop ());
			  push_association<ada_name_association> (ada_pop ());
			  $$ = 1;
			}
	|	simple_exp DOTDOT simple_exp ARROW exp
			{
			  push_component<ada_choices_component> (ada_pop ());
			  operation_up rhs = ada_pop ();
			  operation_up lhs = ada_pop ();
			  push_association<ada_discrete_range_association>
			    (std::move (lhs), std::move (rhs));
			  $$ = 1;
			}
	|	NAME '|' component_associations
			{
			  write_name_assoc (pstate, $1);
			  $$ = $3 + 1;
			}
	|	simple_exp '|' component_associations
			{
			  push_association<ada_name_association> (ada_pop ());
			  $$ = $3 + 1;
			}
	|	simple_exp DOTDOT simple_exp '|' component_associations

			{
			  operation_up rhs = ada_pop ();
			  operation_up lhs = ada_pop ();
			  push_association<ada_discrete_range_association>
			    (std::move (lhs), std::move (rhs));
			  $$ = $5 + 1;
			}
	;

/* Some extensions borrowed from C, for the benefit of those who find they
   can't get used to Ada notation in GDB.  */

primary	:	'*' primary		%prec '.'
			{ ada_wrap<ada_unop_ind_operation> (); }
	|	'&' primary		%prec '.'
			{ ada_addrof (); }
	|	primary '[' exp ']'
			{
			  ada_wrap2<subscript_operation> (BINOP_SUBSCRIPT);
			  ada_wrap<ada_wrapped_operation> ();
			}
	;

%%

/* yylex defined in ada-lex-gen.c: Reads one token, getting characters */
/* through lexptr.  */

/* The following kludge was found necessary to prevent conflicts between */
/* defs.h and non-standard stdlib.h files.  */
#define qsort __qsort__dummy
#include "ada-lex-gen.c"
