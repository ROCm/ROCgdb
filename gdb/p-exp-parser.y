/* YACC parser for Pascal expressions, for GDB.
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

/* This file is derived from c-exp-parser.y */

/* Parse a Pascal expression from text in a string,
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

/* Known bugs or limitations:
    - pascal string operations are not supported at all.
    - there are some problems with boolean types.
    - Pascal type hexadecimal constants are not supported
      because they conflict with the internal variables format.
   Probably also lots of other problems, less well defined PM.  */
%{

#include "expression.h"
#include "value.h"
#include "parser-defs.h"
#include "language.h"
#include "p-lang.h"
#include "p-exp-parser.h"
#include "block.h"
#include "expop.h"

using namespace p_exp_parser;
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
    struct symbol *sym;
    struct type *tval;
    struct stoken sval;
    struct ttype tsym;
    struct symtoken ssym;
    int voidval;
    const struct block *bval;
    enum exp_opcode opcode;
    struct internalvar *ivar;

    struct type **tvec;
    int *ivec;
  }

%type <voidval> exp exp1 type_exp start normal_start variable qualified_name
%type <tval> type typebase
/* %type <bval> block */

/* Fancy type parsing.  */
%type <tval> ptype

%token <typed_val_int> INT
%token <typed_val_float> FLOAT

/* Both NAME and TYPENAME tokens represent symbols in the input,
   and both convey their data as strings.
   But a TYPENAME is a string that happens to be defined as a typedef
   or builtin type name (such as int or char)
   and a NAME is any other symbol.
   Contexts where this distinction is not important can use the
   nonterminal "name", which matches either NAME or TYPENAME.  */

%token <sval> STRING
%token <sval> FIELDNAME
%token <voidval> COMPLETE
%token <ssym> NAME /* BLOCKNAME defined below to give it higher precedence.  */
%token <tsym> TYPENAME
%type <sval> name
%type <ssym> name_not_typename

/* A NAME_OR_INT is a symbol which is not known in the symbol table,
   but which would parse as a valid number in the current input radix.
   E.g. "c" when input_radix==16.  Depending on the parse, it will be
   turned into a name or into a number.  */

%token <ssym> NAME_OR_INT

%token STRUCT CLASS SIZEOF COLONCOLON
%token ERROR

/* Special type cases, put in to allow the parser to distinguish different
   legal basetypes.  */

%token <sval> DOLLAR_VARIABLE


/* Object pascal */
%token THIS
%token <lval> TRUEKEYWORD FALSEKEYWORD

%left ','
%left ABOVE_COMMA
%right ASSIGN
%left NOT
%left OR
%left XOR
%left ANDAND
%left '=' NOTEQUAL
%left '<' '>' LEQ GEQ
%left LSH RSH DIV MOD
%left '@'
%left '+' '-'
%left '*' '/'
%right UNARY INCREMENT DECREMENT
%right ARROW '.' '[' '('
%left '^'
%token <ssym> BLOCKNAME
%type <bval> block
%left COLONCOLON


%%

start   :	{ current_type = NULL;
		  search_field = 0;
		  leftdiv_is_integer = 0;
		}
		normal_start {}
	;

normal_start	:
		exp1
	|	type_exp
	;

type_exp:	type
			{
			  pstate->push_new<type_operation> ($1);
			  current_type = $1; } ;

/* Expressions, including the comma operator.  */
exp1	:	exp
	|	exp1 ',' exp
			{ pstate->wrap2<comma_operation> (); }
	;

/* Expressions, not including the comma operator.  */
exp	:	exp '^'   %prec UNARY
			{ pstate->wrap<unop_ind_operation> ();
			  if (current_type)
			    current_type = current_type->target_type (); }
	;

exp	:	'@' exp    %prec UNARY
			{ pstate->wrap<unop_addr_operation> ();
			  if (current_type)
			    current_type = current_type->pointer_type; }
	;

exp	:	'-' exp    %prec UNARY
			{ pstate->wrap<unary_neg_operation> (); }
	;

exp	:	NOT exp    %prec UNARY
			{ pstate->wrap<unary_logical_not_operation> (); }
	;

exp	:	INCREMENT '(' exp ')'   %prec UNARY
			{ pstate->wrap<preinc_operation> (); }
	;

exp	:	DECREMENT  '(' exp ')'   %prec UNARY
			{ pstate->wrap<predec_operation> (); }
	;


field_exp	:	exp '.'	%prec UNARY
			{ search_field = 1; }
	;

exp	:	field_exp FIELDNAME
			{
			  pstate->push_new<structop_operation>
			    (pstate->pop (), copy_name ($2));
			  search_field = 0;
			  if (current_type)
			    {
			      while (current_type->code ()
				     == TYPE_CODE_PTR)
				current_type =
				  current_type->target_type ();
			      current_type = lookup_struct_elt_type (
				current_type, $2.ptr, 0);
			    }
			 }
	;


exp	:	field_exp name
			{
			  pstate->push_new<structop_operation>
			    (pstate->pop (), copy_name ($2));
			  search_field = 0;
			  if (current_type)
			    {
			      while (current_type->code ()
				     == TYPE_CODE_PTR)
				current_type =
				  current_type->target_type ();
			      current_type = lookup_struct_elt_type (
				current_type, $2.ptr, 0);
			    }
			}
	;
exp	:	field_exp  name COMPLETE
			{
			  structop_base_operation *op
			    = new structop_ptr_operation (pstate->pop (),
							  copy_name ($2));
			  pstate->mark_struct_expression (op);
			  pstate->push (operation_up (op));
			}
	;
exp	:	field_exp COMPLETE
			{
			  structop_base_operation *op
			    = new structop_ptr_operation (pstate->pop (), "");
			  pstate->mark_struct_expression (op);
			  pstate->push (operation_up (op));
			}
	;

exp	:	exp '['
			/* We need to save the current_type value.  */
			{ const char *arrayname;
			  int arrayfieldindex
			    = pascal_is_string_type (current_type, NULL, NULL,
						     NULL, NULL, &arrayname);
			  if (arrayfieldindex)
			    {
			      current_type
				= (current_type
				   ->field (arrayfieldindex - 1).type ());
			      pstate->push_new<structop_operation>
				(pstate->pop (), arrayname);
			    }
			  push_current_type ();  }
		exp1 ']'
			{ pop_current_type ();
			  pstate->wrap2<subscript_operation> ();
			  if (current_type)
			    current_type = current_type->target_type (); }
	;

exp	:	exp '('
			/* This is to save the value of arglist_len
			   being accumulated by an outer function call.  */
			{ push_current_type ();
			  pstate->start_arglist (); }
		arglist ')'	%prec ARROW
			{
			  std::vector<operation_up> args
			    = pstate->pop_vector (pstate->end_arglist ());
			  pstate->push_new<funcall_operation>
			    (pstate->pop (), std::move (args));
			  pop_current_type ();
			  if (current_type)
			    current_type = current_type->target_type ();
			}
	;

arglist	:
	 | exp
			{ pstate->arglist_len = 1; }
	 | arglist ',' exp   %prec ABOVE_COMMA
			{ pstate->arglist_len++; }
	;

exp	:	type '(' exp ')' %prec UNARY
			{ if (current_type)
			    {
			      /* Allow automatic dereference of classes.  */
			      if ((current_type->code () == TYPE_CODE_PTR)
				  && (current_type->target_type ()->code () == TYPE_CODE_STRUCT)
				  && (($1)->code () == TYPE_CODE_STRUCT))
				pstate->wrap<unop_ind_operation> ();
			    }
			  pstate->push_new<unop_cast_operation>
			    (pstate->pop (), $1);
			  current_type = $1; }
	;

exp	:	'(' exp1 ')'
			{ }
	;

/* Binary operators in order of decreasing precedence.  */

exp	:	exp '*' exp
			{ pstate->wrap2<mul_operation> (); }
	;

exp	:	exp '/' {
			  if (current_type && is_integral_type (current_type))
			    leftdiv_is_integer = 1;
			}
		exp
			{
			  if (leftdiv_is_integer && current_type
			      && is_integral_type (current_type))
			    {
			      pstate->push_new<unop_cast_operation>
				(pstate->pop (),
				 parse_type (pstate)->builtin_long_double);
			      current_type
				= parse_type (pstate)->builtin_long_double;
			      leftdiv_is_integer = 0;
			    }

			  pstate->wrap2<div_operation> ();
			}
	;

exp	:	exp DIV exp
			{ pstate->wrap2<intdiv_operation> (); }
	;

exp	:	exp MOD exp
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

exp	:	exp '=' exp
			{
			  pstate->wrap2<equal_operation> ();
			  current_type = parse_type (pstate)->builtin_bool;
			}
	;

exp	:	exp NOTEQUAL exp
			{
			  pstate->wrap2<notequal_operation> ();
			  current_type = parse_type (pstate)->builtin_bool;
			}
	;

exp	:	exp LEQ exp
			{
			  pstate->wrap2<leq_operation> ();
			  current_type = parse_type (pstate)->builtin_bool;
			}
	;

exp	:	exp GEQ exp
			{
			  pstate->wrap2<geq_operation> ();
			  current_type = parse_type (pstate)->builtin_bool;
			}
	;

exp	:	exp '<' exp
			{
			  pstate->wrap2<less_operation> ();
			  current_type = parse_type (pstate)->builtin_bool;
			}
	;

exp	:	exp '>' exp
			{
			  pstate->wrap2<gtr_operation> ();
			  current_type = parse_type (pstate)->builtin_bool;
			}
	;

exp	:	exp ANDAND exp
			{ pstate->wrap2<bitwise_and_operation> (); }
	;

exp	:	exp XOR exp
			{ pstate->wrap2<bitwise_xor_operation> (); }
	;

exp	:	exp OR exp
			{ pstate->wrap2<bitwise_ior_operation> (); }
	;

exp	:	exp ASSIGN exp
			{ pstate->wrap2<assign_operation> (); }
	;

exp	:	TRUEKEYWORD
			{
			  pstate->push_new<bool_operation> ($1);
			  current_type = parse_type (pstate)->builtin_bool;
			}
	;

exp	:	FALSEKEYWORD
			{
			  pstate->push_new<bool_operation> ($1);
			  current_type = parse_type (pstate)->builtin_bool;
			}
	;

exp	:	INT
			{
			  pstate->push_new<long_const_operation>
			    ($1.type, $1.val);
			  current_type = $1.type;
			}
	;

exp	:	NAME_OR_INT
			{ YYSTYPE val;
			  parse_number (pstate, $1.stoken.ptr,
					$1.stoken.length, 0, &val);
			  pstate->push_new<long_const_operation>
			    (val.typed_val_int.type,
			     val.typed_val_int.val);
			  current_type = val.typed_val_int.type;
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

			  /* $ is the normal prefix for pascal
			     hexadecimal values but this conflicts
			     with the GDB use for debugger variables
			     so in expression to enter hexadecimal
			     values we still need to use C syntax with
			     0xff */
			  std::string tmp ($1.ptr, $1.length);
			  /* Handle current_type.  */
			  struct internalvar *intvar
			    = lookup_only_internalvar (tmp.c_str () + 1);
			  if (intvar != nullptr)
			    {
			      scoped_value_mark mark;

			      value *val
				= value_of_internalvar (pstate->gdbarch (),
							intvar);
			      current_type = val->type ();
			    }
			}
	;

exp	:	SIZEOF '(' type ')'	%prec UNARY
			{
			  current_type = parse_type (pstate)->builtin_int;
			  $3 = check_typedef ($3);
			  pstate->push_new<long_const_operation>
			    (parse_type (pstate)->builtin_int,
			     $3->length ()); }
	;

exp	:	SIZEOF  '(' exp ')'      %prec UNARY
			{ pstate->wrap<unop_sizeof_operation> ();
			  current_type = parse_type (pstate)->builtin_int; }

exp	:	STRING
			{ /* C strings are converted into array constants with
			     an explicit null byte added at the end.  Thus
			     the array upper bound is the string length.
			     There is no such thing in C as a completely empty
			     string.  */
			  const char *sp = $1.ptr; int count = $1.length;

			  std::vector<operation_up> args (count + 1);
			  for (int i = 0; i < count; ++i)
			    args[i] = (make_operation<long_const_operation>
				       (parse_type (pstate)->builtin_char,
					*sp++));
			  args[count] = (make_operation<long_const_operation>
					 (parse_type (pstate)->builtin_char,
					  '\0'));
			  pstate->push_new<array_operation>
			    (0, $1.length, std::move (args));
			}
	;

/* Object pascal  */
exp	:	THIS
			{
			  struct value * this_val;
			  struct type * this_type;
			  pstate->push_new<op_this_operation> ();
			  /* We need type of this.  */
			  this_val
			    = value_of_this_silent (pstate->language ());
			  if (this_val)
			    this_type = this_val->type ();
			  else
			    this_type = NULL;
			  if (this_type)
			    {
			      if (this_type->code () == TYPE_CODE_PTR)
				{
				  this_type = this_type->target_type ();
				  pstate->wrap<unop_ind_operation> ();
				}
			    }

			  current_type = this_type;
			}
	;

/* end of object pascal.  */

block	:	BLOCKNAME
			{
			  if ($1.sym.symbol != 0)
			      $$ = $1.sym.symbol->value_block ();
			  else
			    {
			      std::string copy = copy_name ($1.stoken);
			      struct symtab *tem =
				  lookup_symtab (current_program_space, copy.c_str ());
			      if (tem)
				$$ = (tem->compunit ().blockvector ()
				      ->static_block ());
			      else
				error (_("No file or function \"%s\"."),
				       copy.c_str ());
			    }
			}
	;

block	:	block COLONCOLON name
			{
			  std::string copy = copy_name ($3);
			  struct symbol *tem
			    = lookup_symbol (copy.c_str (), $1,
					     SEARCH_FUNCTION_DOMAIN,
					     nullptr).symbol;

			  if (tem == nullptr)
			    error (_("No function \"%s\" in specified context."),
				   copy.c_str ());
			  $$ = tem->value_block (); }
	;

variable:	block COLONCOLON name
			{ struct block_symbol sym;

			  std::string copy = copy_name ($3);
			  sym = lookup_symbol (copy.c_str (), $1,
					       SEARCH_VFT, NULL);
			  if (sym.symbol == 0)
			    error (_("No symbol \"%s\" in specified context."),
				   copy.c_str ());

			  pstate->push_new<var_value_operation> (sym);
			}
	;

qualified_name:	typebase COLONCOLON name
			{
			  struct type *type = $1;

			  if (type->code () != TYPE_CODE_STRUCT
			      && type->code () != TYPE_CODE_UNION)
			    error (_("`%s' is not defined as an aggregate type."),
				   type->name ());

			  pstate->push_new<scope_operation>
			    (type, copy_name ($3));
			}
	;

variable:	qualified_name
	|	COLONCOLON name
			{
			  std::string name = copy_name ($2);

			  struct block_symbol sym
			    = lookup_symbol (name.c_str (), nullptr,
					     SEARCH_VFT, nullptr);
			  pstate->push_symbol (name.c_str (), sym);
			}
	;

variable:	name_not_typename
			{ struct block_symbol sym = $1.sym;

			  if (sym.symbol)
			    {
			      if (symbol_read_needs_frame (sym.symbol))
				pstate->block_tracker->update (sym);

			      pstate->push_new<var_value_operation> (sym);
			      current_type = sym.symbol->type (); }
			  else if ($1.is_a_field_of_this)
			    {
			      struct value * this_val;
			      struct type * this_type;
			      /* Object pascal: it hangs off of `this'.  Must
				 not inadvertently convert from a method call
				 to data ref.  */
			      pstate->block_tracker->update (sym);
			      operation_up thisop
				= make_operation<op_this_operation> ();
			      pstate->push_new<structop_operation>
				(std::move (thisop), copy_name ($1.stoken));
			      /* We need type of this.  */
			      this_val
				= value_of_this_silent (pstate->language ());
			      if (this_val)
				this_type = this_val->type ();
			      else
				this_type = NULL;
			      if (this_type)
				current_type = lookup_struct_elt_type (
				  this_type,
				  copy_name ($1.stoken).c_str (), 0);
			      else
				current_type = NULL;
			    }
			  else
			    {
			      std::string arg = copy_name ($1.stoken);

			      bound_minimal_symbol msymbol
				= lookup_minimal_symbol (current_program_space, arg.c_str ());
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


ptype	:	typebase
	;

/* We used to try to recognize more pointer to member types here, but
   that didn't work (shift/reduce conflicts meant that these rules never
   got executed).  The problem is that
     int (foo::bar::baz::bizzle)
   is a function type but
     int (foo::bar::baz::bizzle::*)
   is a pointer to member type.  Stroustrup loses again!  */

type	:	ptype
	;

typebase  /* Implements (approximately): (type-qualifier)* type-specifier */
	:	'^' typebase
			{ $$ = lookup_pointer_type ($2); }
	|	TYPENAME
			{ $$ = $1.type; }
	|	STRUCT name
			{ $$
			    = lookup_struct (copy_name ($2).c_str (),
					     pstate->expression_context_block);
			}
	|	CLASS name
			{ $$
			    = lookup_struct (copy_name ($2).c_str (),
					     pstate->expression_context_block);
			}
	/* "const" and "volatile" are currently ignored.  A type qualifier
	   after the type is handled in the ptype rule.  I think these could
	   be too.  */
	;

name	:	NAME { $$ = $1.stoken; }
	|	BLOCKNAME { $$ = $1.stoken; }
	|	TYPENAME { $$ = $1.stoken; }
	|	NAME_OR_INT  { $$ = $1.stoken; }
	;

name_not_typename :	NAME
	|	BLOCKNAME
/* These would be useful if name_not_typename was useful, but it is just
   a fake for "variable", so these cause reduce/reduce conflicts because
   the parser can't tell whether NAME_OR_INT is a name_not_typename (=variable,
   =exp) or just an exp.  If name_not_typename was ever used in an lvalue
   context where only a name could occur, this might be useful.
	|	NAME_OR_INT
 */
	;
