/* YACC parser for C expressions, for GDB.
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

/* Parse a C expression from text in a string,
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
#include "c-exp-parser.h"
#include "c-lang.h"
#include "block.h"
#include "cp-support.h"
#include "objc-lang.h"
#include "typeprint.h"
#include "cp-abi.h"
#include "type-stack.h"
#include "target-float.h"
#include "c-exp.h"
#include "cli/cli-style.h"

using namespace c_exp_parser;
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
    struct type *tval;
    struct stoken sval;
    c_exp_parser::qualified_name_token qval;
    struct typed_stoken tsval;
    struct ttype tsym;
    struct symtoken ssym;
    int voidval;
    const struct block *bval;
    enum exp_opcode opcode;

    struct stoken_vector svec;
    std::vector<struct type *> *tvec;

    struct type_stack *type_stack;

    struct objc_class_str theclass;
  }

%{
/* YYSTYPE gets defined by %union */
#if defined(YYBISON) && YYBISON < 30800
static void c_print_token (FILE *file, int type, YYSTYPE value);
#define YYPRINT(FILE, TYPE, VALUE) c_print_token (FILE, TYPE, VALUE)
#endif
%}

%type <voidval> exp exp1 type_exp start variable qualified_name lcurly function_method
%type <lval> rcurly
%type <tval> type typebase scalar_type tag_name_or_complete
%type <tvec> nonempty_typelist func_mod parameter_typelist
/* %type <bval> block */

/* Fancy type parsing.  */
%type <tval> ptype
%type <lval> array_mod
%type <tval> conversion_type_id

%type <type_stack> ptr_operator_ts abs_decl direct_abs_decl

%token <typed_val_int> INT COMPLEX_INT
%token <typed_val_float> FLOAT COMPLEX_FLOAT

/* Both NAME and TYPENAME tokens represent symbols in the input,
   and both convey their data as strings.
   But a TYPENAME is a string that happens to be defined as a typedef
   or builtin type name (such as int or char)
   and a NAME is any other symbol.
   Contexts where this distinction is not important can use the
   nonterminal "name", which matches either NAME or TYPENAME.  */

%token <tsval> STRING
%token <tsval> NSSTRING		/* ObjC Foundation "NSString" literal */
%token <sval> SELECTOR		/* ObjC "@selector" pseudo-operator   */
%token <tsval> CHAR
%token <ssym> NAME /* BLOCKNAME defined below to give it higher precedence. */
%token <ssym> UNKNOWN_CPP_NAME
%token <voidval> COMPLETE
%token <tsym> TYPENAME
%token <theclass> CLASSNAME	/* ObjC Class name */
%type <sval> name
%type <qval> qual_field_name field_name field_name_or_complete
%type <qval> field_or_destructor
%type <svec> string_exp
%type <ssym> name_not_typename
%type <tsym> type_name

 /* This is like a '[' token, but is only generated when parsing
    Objective C.  This lets us reuse the same parser without
    erroneously parsing ObjC-specific expressions in C.  */
%token OBJC_LBRAC

/* A NAME_OR_INT is a symbol which is not known in the symbol table,
   but which would parse as a valid number in the current input radix.
   E.g. "c" when input_radix==16.  Depending on the parse, it will be
   turned into a name or into a number.  */

%token <ssym> NAME_OR_INT

%token OPERATOR
%token STRUCT CLASS UNION ENUM SIZEOF ALIGNOF UNSIGNED COLONCOLON
%token TEMPLATE
%token ERROR
%token NEW DELETE
%type <sval> oper
%token REINTERPRET_CAST DYNAMIC_CAST STATIC_CAST CONST_CAST
%token ENTRY
%token TYPEOF
%token DECLTYPE
%token TYPEID

/* Special type cases, put in to allow the parser to distinguish different
   legal basetypes.  */
%token SIGNED_KEYWORD LONG SHORT INT_KEYWORD CONST_KEYWORD VOLATILE_KEYWORD DOUBLE_KEYWORD
%token RESTRICT ATOMIC
%token FLOAT_KEYWORD COMPLEX

%token <sval> DOLLAR_VARIABLE

%token <opcode> ASSIGN_MODIFY

/* C++ */
%token TRUEKEYWORD
%token FALSEKEYWORD


%left ','
%left ABOVE_COMMA
%right '=' ASSIGN_MODIFY
%right '?'
%left OROR
%left ANDAND
%left '|'
%left '^'
%left '&'
%left EQUAL NOTEQUAL
%left '<' '>' LEQ GEQ
%left LSH RSH
%left '@'
%left '+' '-'
%left '*' '/' '%'
%right UNARY INCREMENT DECREMENT
%right ARROW ARROW_STAR '.' DOT_STAR '[' OBJC_LBRAC '('
%token <ssym> BLOCKNAME
%token <bval> FILENAME
%type <bval> block
%left COLONCOLON

%token DOTDOTDOT


%%

start   :	exp1
	|	type_exp
	;

type_exp:	type
			{
			  pstate->push_new<type_operation> ($1);
			}
	|	TYPEOF '(' exp ')'
			{
			  pstate->wrap<typeof_operation> ();
			}
	|	TYPEOF '(' type ')'
			{
			  pstate->push_new<type_operation> ($3);
			}
	|	DECLTYPE '(' exp ')'
			{
			  pstate->wrap<decltype_operation> ();
			}
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
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    pstate->wrap<opencl_not_operation> ();
			  else
			    pstate->wrap<unary_logical_not_operation> ();
			}
	;

exp	:	'~' exp    %prec UNARY
			{ pstate->wrap<unary_complement_operation> (); }
	;

exp	:	INCREMENT exp    %prec UNARY
			{ pstate->wrap<preinc_operation> (); }
	;

exp	:	DECREMENT exp    %prec UNARY
			{ pstate->wrap<predec_operation> (); }
	;

exp	:	exp INCREMENT    %prec UNARY
			{ pstate->wrap<postinc_operation> (); }
	;

exp	:	exp DECREMENT    %prec UNARY
			{ pstate->wrap<postdec_operation> (); }
	;

exp	:	TYPEID '(' exp ')' %prec UNARY
			{ pstate->wrap<typeid_operation> (); }
	;

exp	:	TYPEID '(' type_exp ')' %prec UNARY
			{ pstate->wrap<typeid_operation> (); }
	;

exp	:	SIZEOF exp       %prec UNARY
			{ pstate->wrap<unop_sizeof_operation> (); }
	;

exp	:	ALIGNOF '(' type_exp ')'	%prec UNARY
			{ pstate->wrap<unop_alignof_operation> (); }
	;

exp	:	exp ARROW
			{
			  cpstate->assume_classification = TYPE_CODE_VOID;
			}
		field_name_or_complete
			{
			  cpstate->assume_classification = TYPE_CODE_UNDEF;

			  if ($4.prefix != nullptr)
			    {
			      handle_qualified_field_name ($4);
			      /* exp->type::name becomes exp->*(&type::name) */
			      /* Note: this doesn't work if name is a
				 static member!  FIXME */
			      pstate->wrap<unop_addr_operation> ();
			      pstate->wrap2<structop_mptr_operation> ();
			    }
			  else
			    {
			      structop_base_operation *op
				= new structop_ptr_operation (pstate->pop (),
							      copy_name ($4));
			      if ($4.complete)
				pstate->mark_struct_expression (op);
			      pstate->push (operation_up (op));
			    }
			}
	;

exp	:	exp ARROW_STAR exp
			{ pstate->wrap2<structop_mptr_operation> (); }
	;

exp	:	exp '.'
			{
			  cpstate->assume_classification = TYPE_CODE_VOID;
			}
		field_name_or_complete
			{
			  cpstate->assume_classification = TYPE_CODE_UNDEF;

			  if ($4.prefix != nullptr)
			    {
			      handle_qualified_field_name ($4);
			      /* exp.type::name becomes exp.*(&type::name) */
			      /* Note: this doesn't work if name is a
				 static member!  FIXME */
			      pstate->wrap<unop_addr_operation> ();
			      pstate->wrap2<structop_member_operation> ();
			    }
			  else if (pstate->language ()->la_language
				   == language_opencl
				   && !$4.complete)
			    pstate->push_new<opencl_structop_operation>
			      (pstate->pop (), copy_name ($4));
			  else
			    {
			      structop_base_operation *op
				= new structop_operation (pstate->pop (),
							  copy_name ($4));
			      if ($4.complete)
				pstate->mark_struct_expression (op);
			      pstate->push (operation_up (op));
			    }
			}
	;

exp	:	exp DOT_STAR exp
			{ pstate->wrap2<structop_member_operation> (); }
	;

exp	:	exp '[' exp1 ']'
			{ pstate->wrap2<subscript_operation> (); }
	;

exp	:	exp OBJC_LBRAC exp1 ']'
			{ pstate->wrap2<subscript_operation> (); }
	;

/*
 * The rules below parse ObjC message calls of the form:
 *	'[' target selector {':' argument}* ']'
 */

exp	: 	OBJC_LBRAC TYPENAME
			{
			  CORE_ADDR theclass;

			  std::string copy = copy_name ($2.stoken);
			  theclass = lookup_objc_class (pstate->gdbarch (),
							copy.c_str ());
			  if (theclass == 0)
			    error (_("%s is not an ObjC Class"),
				   copy.c_str ());
			  pstate->push_new<long_const_operation>
			    (parse_type (pstate)->builtin_int,
			     (LONGEST) theclass);
			  start_msglist();
			}
		msglist ']'
			{ end_msglist (pstate); }
	;

exp	:	OBJC_LBRAC CLASSNAME
			{
			  pstate->push_new<long_const_operation>
			    (parse_type (pstate)->builtin_int,
			     (LONGEST) $2.theclass);
			  start_msglist();
			}
		msglist ']'
			{ end_msglist (pstate); }
	;

exp	:	OBJC_LBRAC exp
			{ start_msglist(); }
		msglist ']'
			{ end_msglist (pstate); }
	;

msglist :	name
			{ add_msglist(&$1, 0); }
	|	msgarglist
	;

msgarglist :	msgarg
	|	msgarglist msgarg
	;

msgarg	:	name ':' exp
			{ add_msglist(&$1, 1); }
	|	':' exp	/* Unnamed arg.  */
			{ add_msglist(0, 1);   }
	|	',' exp	/* Variable number of args.  */
			{ add_msglist(0, 0);   }
	;

exp	:	exp '('
			/* This is to save the value of arglist_len
			   being accumulated by an outer function call.  */
			{ pstate->start_arglist (); }
		arglist ')'	%prec ARROW
			{
			  std::vector<operation_up> args
			    = pstate->pop_vector (pstate->end_arglist ());
			  pstate->push_new<funcall_operation>
			    (pstate->pop (), std::move (args));
			}
	;

/* This is here to disambiguate with the production for
   "func()::static_var" further below, which uses
   function_method_void.  */
exp	:	exp '(' ')' %prec ARROW
			{
			  pstate->push_new<funcall_operation>
			    (pstate->pop (), std::vector<operation_up> ());
			}
	;


exp	:	UNKNOWN_CPP_NAME '('
			{
			  /* This could potentially be a an argument defined
			     lookup function (Koenig).  */
			  /* This is to save the value of arglist_len
			     being accumulated by an outer function call.  */
			  pstate->start_arglist ();
			}
		arglist ')'	%prec ARROW
			{
			  std::vector<operation_up> args
			    = pstate->pop_vector (pstate->end_arglist ());
			  pstate->push_new<adl_func_operation>
			    (copy_name ($1.stoken),
			     pstate->expression_context_block,
			     std::move (args));
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

function_method:       exp '(' parameter_typelist ')' const_or_volatile
			{
			  std::vector<struct type *> *type_list = $3;
			  /* Save the const/volatile qualifiers as
			     recorded by the const_or_volatile
			     production's actions.  */
			  type_instance_flags flags
			    = (cpstate->type_stack
			       .follow_type_instance_flags ());
			  pstate->push_new<type_instance_operation>
			    (flags, std::move (*type_list),
			     pstate->pop ());
			}
	;

function_method_void:	    exp '(' ')' const_or_volatile
		       {
			  type_instance_flags flags
			    = (cpstate->type_stack
			       .follow_type_instance_flags ());
			  pstate->push_new<type_instance_operation>
			    (flags, std::vector<type *> (), pstate->pop ());
		       }
       ;

exp     :       function_method
	;

/* Normally we must interpret "func()" as a function call, instead of
   a type.  The user needs to write func(void) to disambiguate.
   However, in the "func()::static_var" case, there's no
   ambiguity.  */
function_method_void_or_typelist: function_method
	|               function_method_void
	;

exp     :       function_method_void_or_typelist COLONCOLON name
			{
			  pstate->push_new<func_static_var_operation>
			    (pstate->pop (), copy_name ($3));
			}
	;

rcurly	:	'}'
			{ $$ = pstate->end_arglist () - 1; }
	;
exp	:	lcurly arglist rcurly	%prec ARROW
			{
			  std::vector<operation_up> args
			    = pstate->pop_vector ($3 + 1);
			  pstate->push_new<array_operation> (0, $3,
							     std::move (args));
			}
	;

exp	:	lcurly type_exp rcurly exp  %prec UNARY
			{ pstate->wrap2<unop_memval_type_operation> (); }
	;

exp	:	'(' type_exp ')' exp  %prec UNARY
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    pstate->wrap2<opencl_cast_type_operation> ();
			  else
			    pstate->wrap2<unop_cast_type_operation> ();
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
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    pstate->wrap2<opencl_equal_operation> ();
			  else
			    pstate->wrap2<equal_operation> ();
			}
	;

exp	:	exp NOTEQUAL exp
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    pstate->wrap2<opencl_notequal_operation> ();
			  else
			    pstate->wrap2<notequal_operation> ();
			}
	;

exp	:	exp LEQ exp
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    pstate->wrap2<opencl_leq_operation> ();
			  else
			    pstate->wrap2<leq_operation> ();
			}
	;

exp	:	exp GEQ exp
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    pstate->wrap2<opencl_geq_operation> ();
			  else
			    pstate->wrap2<geq_operation> ();
			}
	;

exp	:	exp '<' exp
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    pstate->wrap2<opencl_less_operation> ();
			  else
			    pstate->wrap2<less_operation> ();
			}
	;

exp	:	exp '>' exp
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    pstate->wrap2<opencl_gtr_operation> ();
			  else
			    pstate->wrap2<gtr_operation> ();
			}
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
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    {
			      operation_up rhs = pstate->pop ();
			      operation_up lhs = pstate->pop ();
			      pstate->push_new<opencl_logical_binop_operation>
				(BINOP_LOGICAL_AND, std::move (lhs),
				 std::move (rhs));
			    }
			  else
			    pstate->wrap2<logical_and_operation> ();
			}
	;

exp	:	exp OROR exp
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    {
			      operation_up rhs = pstate->pop ();
			      operation_up lhs = pstate->pop ();
			      pstate->push_new<opencl_logical_binop_operation>
				(BINOP_LOGICAL_OR, std::move (lhs),
				 std::move (rhs));
			    }
			  else
			    pstate->wrap2<logical_or_operation> ();
			}
	;

exp	:	exp '?' exp ':' exp	%prec '?'
			{
			  operation_up last = pstate->pop ();
			  operation_up mid = pstate->pop ();
			  operation_up first = pstate->pop ();
			  if (pstate->language ()->la_language
			      == language_opencl)
			    pstate->push_new<opencl_ternop_cond_operation>
			      (std::move (first), std::move (mid),
			       std::move (last));
			  else
			    pstate->push_new<ternop_cond_operation>
			      (std::move (first), std::move (mid),
			       std::move (last));
			}
	;

exp	:	exp '=' exp
			{
			  if (pstate->language ()->la_language
			      == language_opencl)
			    pstate->wrap2<opencl_assign_operation> ();
			  else
			    pstate->wrap2<assign_operation> ();
			}
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

exp	:	COMPLEX_INT
			{
			  operation_up real
			    = (make_operation<long_const_operation>
			       ($1.type->target_type (), 0));
			  operation_up imag
			    = (make_operation<long_const_operation>
			       ($1.type->target_type (), $1.val));
			  pstate->push_new<complex_operation>
			    (std::move (real), std::move (imag), $1.type);
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

exp	:	COMPLEX_FLOAT
			{
			  struct type *underlying = $1.type->target_type ();

			  float_data val;
			  target_float_from_host_double (val.data (),
							 underlying, 0);
			  operation_up real
			    = (make_operation<float_const_operation>
			       (underlying, val));

			  std::copy (std::begin ($1.val), std::end ($1.val),
				     std::begin (val));
			  operation_up imag
			    = (make_operation<float_const_operation>
			       (underlying, val));

			  pstate->push_new<complex_operation>
			    (std::move (real), std::move (imag),
			     $1.type);
			}
	;

exp	:	variable
	;

exp	:	DOLLAR_VARIABLE
			{
			  pstate->push_dollar ($1);
			}
	;

exp	:	SELECTOR
			{
			  pstate->push_new<objc_selector_operation>
			    (copy_name ($1));
			}
	;

exp	:	SIZEOF '(' type ')'	%prec UNARY
			{ struct type *type = $3;
			  struct type *int_type
			    = lookup_signed_typename (pstate->language (),
						      "int");
			  type = check_typedef (type);

			    /* $5.3.3/2 of the C++ Standard (n3290 draft)
			       says of sizeof:  "When applied to a reference
			       or a reference type, the result is the size of
			       the referenced type."  */
			  if (TYPE_IS_REFERENCE (type))
			    type = check_typedef (type->target_type ());

			  pstate->push_new<long_const_operation>
			    (int_type, type->length ());
			}
	;

exp	:	REINTERPRET_CAST '<' type_exp '>' '(' exp ')' %prec UNARY
			{ pstate->wrap2<reinterpret_cast_operation> (); }
	;

exp	:	STATIC_CAST '<' type_exp '>' '(' exp ')' %prec UNARY
			{ pstate->wrap2<unop_cast_type_operation> (); }
	;

exp	:	DYNAMIC_CAST '<' type_exp '>' '(' exp ')' %prec UNARY
			{ pstate->wrap2<dynamic_cast_operation> (); }
	;

exp	:	CONST_CAST '<' type_exp '>' '(' exp ')' %prec UNARY
			{ /* We could do more error checking here, but
			     it doesn't seem worthwhile.  */
			  pstate->wrap2<unop_cast_type_operation> (); }
	;

string_exp:
		STRING
			{
			  /* We copy the string here, and not in the
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

	|	string_exp STRING
			{
			  /* Note that we NUL-terminate here, but just
			     for convenience.  */
			  char *p;
			  ++$$.len;
			  $$.tokens = XRESIZEVEC (struct typed_stoken,
						  $$.tokens, $$.len);

			  p = (char *) malloc ($2.length + 1);
			  memcpy (p, $2.ptr, $2.length + 1);

			  $$.tokens[$$.len - 1].type = $2.type;
			  $$.tokens[$$.len - 1].length = $2.length;
			  $$.tokens[$$.len - 1].ptr = p;
			}
		;

exp	:	string_exp
			{
			  int i;
			  c_string_type type = C_STRING;

			  for (i = 0; i < $1.len; ++i)
			    {
			      switch ($1.tokens[i].type)
				{
				case C_STRING:
				  break;
				case C_WIDE_STRING:
				case C_STRING_16:
				case C_STRING_32:
				  if (type != C_STRING
				      && type != $1.tokens[i].type)
				    error (_("Undefined string concatenation."));
				  type = (enum c_string_type_values) $1.tokens[i].type;
				  break;
				default:
				  /* internal error */
				  internal_error ("unrecognized type in string concatenation");
				}
			    }

			  pstate->push_c_string (type, &$1);
			  for (i = 0; i < $1.len; ++i)
			    free ($1.tokens[i].ptr);
			  free ($1.tokens);
			}
	;

exp     :	NSSTRING
			{
			  /* ObjC NextStep NSString constant of the
			     form '@' '"' string '"'.  */
			  pstate->push_new<objc_nsstring_operation>
			    (std::string ($1.ptr, $1.length));
			}
	;

/* C++.  */
exp     :       TRUEKEYWORD
			{ pstate->push_new<long_const_operation>
			    (parse_type (pstate)->builtin_bool, 1);
			}
	;

exp     :       FALSEKEYWORD
			{ pstate->push_new<long_const_operation>
			    (parse_type (pstate)->builtin_bool, 0);
			}
	;

/* end of C++.  */

block	:	BLOCKNAME
			{
			  if ($1.sym.symbol)
			    $$ = $1.sym.symbol->value_block ();
			  else
			    error (_("No file or function \"%s\"."),
				   copy_name ($1.stoken).c_str ());
			}
	|	FILENAME
			{
			  $$ = $1;
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
			    error (_("No function \"%ps\" in specified context."),
				   styled_string (function_name_style.style (),
						  copy.c_str ()));
			  $$ = tem->value_block (); }
	;

variable:	name_not_typename ENTRY
			{ struct symbol *sym = $1.sym.symbol;

			  if (sym == NULL || !sym->is_argument ()
			      || !symbol_read_needs_frame (sym))
			    error (_("@entry can be used only for function "
				     "parameters, not for \"%s\""),
				   copy_name ($1.stoken).c_str ());

			  pstate->push_new<var_entry_value_operation> (sym);
			}
	;

variable:	block COLONCOLON name
			{
			  std::string copy = copy_name ($3);
			  struct block_symbol sym
			    = lookup_symbol (copy.c_str (), $1,
					     SEARCH_VFT, NULL);

			  if (sym.symbol == 0)
			    error (_("No symbol \"%s\" in specified context."),
				   copy.c_str ());
			  if (symbol_read_needs_frame (sym.symbol))
			    pstate->block_tracker->update (sym);

			  pstate->push_new<var_value_operation> (sym);
			}
	;

qualified_name:	TYPENAME COLONCOLON name
			{
			  struct type *type = $1.type;
			  type = check_typedef (type);
			  if (!type_aggregate_p (type))
			    error (_("`%s' is not defined as an aggregate type."),
				   type->safe_name ());

			  pstate->push_new<scope_operation> (type,
							     copy_name ($3));
			}
	|	TYPENAME COLONCOLON '~' name
			{
			  struct type *type = $1.type;

			  type = check_typedef (type);
			  if (!type_aggregate_p (type))
			    error (_("`%s' is not defined as an aggregate type."),
				   type->safe_name ());
			  std::string name = "~" + std::string ($4.ptr,
								$4.length);

			  /* Check for valid destructor name.  */
			  destructor_name_p (name.c_str (), $1.type);
			  pstate->push_new<scope_operation> (type,
							     std::move (name));
			}
	|	TYPENAME COLONCOLON name COLONCOLON name
			{
			  std::string copy = copy_name ($3);
			  error (_("No type \"%s\" within class "
				   "or namespace \"%s\"."),
				 copy.c_str (), $1.type->safe_name ());
			}
	;

variable:	qualified_name
	|	COLONCOLON name_not_typename
			{
			  std::string name = copy_name ($2.stoken);
			  struct block_symbol sym
			    = lookup_symbol (name.c_str (),
					     (const struct block *) NULL,
					     SEARCH_VFT, NULL);
			  pstate->push_symbol (name.c_str (), sym);
			}
	;

variable:	name_not_typename
			{ struct block_symbol sym = $1.sym;

			  if (sym.symbol)
			    {
			      if (symbol_read_needs_frame (sym.symbol))
				pstate->block_tracker->update (sym);

			      /* If we found a function, see if it's
				 an ifunc resolver that has the same
				 address as the ifunc symbol itself.
				 If so, prefer the ifunc symbol.  */

			      bound_minimal_symbol resolver
				= find_gnu_ifunc (sym.symbol);
			      if (resolver.minsym != NULL)
				pstate->push_new<var_msym_value_operation>
				  (resolver);
			      else
				pstate->push_new<var_value_operation> (sym);
			    }
			  else if ($1.is_a_field_of_this)
			    {
			      /* C++: it hangs off of `this'.  Must
				 not inadvertently convert from a method call
				 to data ref.  */
			      pstate->block_tracker->update (sym);
			      operation_up thisop
				= make_operation<op_this_operation> ();
			      pstate->push_new<structop_ptr_operation>
				(std::move (thisop), copy_name ($1.stoken));
			    }
			  else
			    {
			      std::string arg = copy_name ($1.stoken);

			      bound_minimal_symbol msymbol
				= lookup_minimal_symbol (current_program_space, arg.c_str ());
			      if (msymbol.minsym == NULL)
				{
				  if (!current_program_space->has_full_symbols ()
				      && !current_program_space->has_partial_symbols ())
				    error (_("No symbol table is loaded.  Use the \"%ps\" command."),
					   styled_string (command_style.style (),
							  "file"));
				  else
				    error (_("No symbol \"%s\" in current context."),
					   arg.c_str ());
				}

			      /* This minsym might be an alias for
				 another function.  See if we can find
				 the debug symbol for the target, and
				 if so, use it instead, since it has
				 return type / prototype info.  This
				 is important for example for "p
				 *__errno_location()".  */
			      symbol *alias_target
				= ((msymbol.minsym->type () != mst_text_gnu_ifunc
				    && msymbol.minsym->type () != mst_data_gnu_ifunc)
				   ? find_function_alias_target (msymbol)
				   : NULL);
			      if (alias_target != NULL)
				{
				  block_symbol bsym { alias_target,
				    alias_target->value_block () };
				  pstate->push_new<var_value_operation> (bsym);
				}
			      else
				pstate->push_new<var_msym_value_operation>
				  (msymbol);
			    }
			}
	;

const_or_volatile: const_or_volatile_noopt
	|
	;

single_qualifier:
		CONST_KEYWORD
			{ cpstate->type_stack.insert (tp_const); }
	| 	VOLATILE_KEYWORD
			{ cpstate->type_stack.insert (tp_volatile); }
	| 	ATOMIC
			{ cpstate->type_stack.insert (tp_atomic); }
	| 	RESTRICT
			{ cpstate->type_stack.insert (tp_restrict); }
	|	'@' NAME
		{
		  cpstate->type_stack.insert (pstate->gdbarch (),
					      copy_name ($2.stoken).c_str ());
		}
	|	'@' UNKNOWN_CPP_NAME
		{
		  cpstate->type_stack.insert (pstate->gdbarch (),
					      copy_name ($2.stoken).c_str ());
		}
	;

qualifier_seq_noopt:
		single_qualifier
	| 	qualifier_seq_noopt single_qualifier
	;

qualifier_seq:
		qualifier_seq_noopt
	|
	;

ptr_operator:
		ptr_operator '*'
			{ cpstate->type_stack.insert (tp_pointer); }
		qualifier_seq
	|	'*'
			{ cpstate->type_stack.insert (tp_pointer); }
		qualifier_seq
	|	'&'
			{ cpstate->type_stack.insert (tp_reference); }
	|	'&' ptr_operator
			{ cpstate->type_stack.insert (tp_reference); }
	|       ANDAND
			{ cpstate->type_stack.insert (tp_rvalue_reference); }
	|       ANDAND ptr_operator
			{ cpstate->type_stack.insert (tp_rvalue_reference); }
	;

ptr_operator_ts: ptr_operator
			{
			  $$ = cpstate->type_stack.create ();
			  cpstate->type_stacks.emplace_back ($$);
			}
	;

abs_decl:	ptr_operator_ts direct_abs_decl
			{ $$ = $2->append ($1); }
	|	ptr_operator_ts
	|	direct_abs_decl
	;

direct_abs_decl: '(' abs_decl ')'
			{ $$ = $2; }
	|	direct_abs_decl array_mod
			{
			  cpstate->type_stack.push ($1);
			  cpstate->type_stack.push (tp_array, $2);
			  $$ = cpstate->type_stack.create ();
			  cpstate->type_stacks.emplace_back ($$);
			}
	|	array_mod
			{
			  cpstate->type_stack.push (tp_array, $1);
			  $$ = cpstate->type_stack.create ();
			  cpstate->type_stacks.emplace_back ($$);
			}

	| 	direct_abs_decl func_mod
			{
			  cpstate->type_stack.push ($1);
			  cpstate->type_stack.push ($2);
			  $$ = cpstate->type_stack.create ();
			  cpstate->type_stacks.emplace_back ($$);
			}
	|	func_mod
			{
			  cpstate->type_stack.push ($1);
			  $$ = cpstate->type_stack.create ();
			  cpstate->type_stacks.emplace_back ($$);
			}
	;

array_mod:	'[' ']'
			{ $$ = -1; }
	|	OBJC_LBRAC ']'
			{ $$ = -1; }
	|	'[' INT ']'
			{ $$ = $2.val; }
	|	OBJC_LBRAC INT ']'
			{ $$ = $2.val; }
	;

func_mod:	'(' ')'
			{
			  $$ = new std::vector<struct type *>;
			  cpstate->type_lists.emplace_back ($$);
			}
	|	'(' parameter_typelist ')'
			{ $$ = $2; }
	;

/* We used to try to recognize pointer to member types here, but
   that didn't work (shift/reduce conflicts meant that these rules never
   got executed).  The problem is that
     int (foo::bar::baz::bizzle)
   is a function type but
     int (foo::bar::baz::bizzle::*)
   is a pointer to member type.  Stroustrup loses again!  */

type	:	ptype
	;

/* A helper production that recognizes scalar types that can validly
   be used with _Complex.  */

scalar_type:
		INT_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "int"); }
	|	LONG
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long"); }
	|	SHORT
			{ $$ = lookup_signed_typename (pstate->language (),
						       "short"); }
	|	LONG INT_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long"); }
	|	LONG SIGNED_KEYWORD INT_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long"); }
	|	LONG SIGNED_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long"); }
	|	SIGNED_KEYWORD LONG INT_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long"); }
	|	UNSIGNED LONG INT_KEYWORD
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "long"); }
	|	LONG UNSIGNED INT_KEYWORD
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "long"); }
	|	LONG UNSIGNED
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "long"); }
	|	LONG LONG
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long long"); }
	|	LONG LONG INT_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long long"); }
	|	LONG LONG SIGNED_KEYWORD INT_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long long"); }
	|	LONG LONG SIGNED_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long long"); }
	|	SIGNED_KEYWORD LONG LONG
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long long"); }
	|	SIGNED_KEYWORD LONG LONG INT_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "long long"); }
	|	UNSIGNED LONG LONG
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "long long"); }
	|	UNSIGNED LONG LONG INT_KEYWORD
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "long long"); }
	|	LONG LONG UNSIGNED
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "long long"); }
	|	LONG LONG UNSIGNED INT_KEYWORD
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "long long"); }
	|	SHORT INT_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "short"); }
	|	SHORT SIGNED_KEYWORD INT_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "short"); }
	|	SHORT SIGNED_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "short"); }
	|	UNSIGNED SHORT INT_KEYWORD
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "short"); }
	|	SHORT UNSIGNED
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "short"); }
	|	SHORT UNSIGNED INT_KEYWORD
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "short"); }
	|	DOUBLE_KEYWORD
			{ $$ = lookup_typename (pstate->language (),
						"double",
						NULL,
						0); }
	|	FLOAT_KEYWORD
			{ $$ = lookup_typename (pstate->language (),
						"float",
						NULL,
						0); }
	|	LONG DOUBLE_KEYWORD
			{ $$ = lookup_typename (pstate->language (),
						"long double",
						NULL,
						0); }
	|	UNSIGNED type_name
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 $2.type->name ()); }
	|	UNSIGNED
			{ $$ = lookup_unsigned_typename (pstate->language (),
							 "int"); }
	|	SIGNED_KEYWORD type_name
			{ $$ = lookup_signed_typename (pstate->language (),
						       $2.type->name ()); }
	|	SIGNED_KEYWORD
			{ $$ = lookup_signed_typename (pstate->language (),
						       "int"); }
	;

/* Implements (approximately): (type-qualifier)* type-specifier.

   When type-specifier is only ever a single word, like 'float' then these
   arrive as pre-built TYPENAME tokens thanks to the classify_name
   function.  However, when a type-specifier can contain multiple words,
   for example 'double' can appear as just 'double' or 'long double', and
   similarly 'long' can appear as just 'long' or in 'long double', then
   these type-specifiers are parsed into their own tokens in the function
   lex_one_token and the ident_tokens array.  These separate tokens are all
   recognised here.  */
typebase
	:	TYPENAME
			{ $$ = $1.type; }
	|	scalar_type
			{ $$ = $1; }
	|	COMPLEX scalar_type
			{
			  $$ = init_complex_type (nullptr, $2);
			}
	|	STRUCT
			{
			  cpstate->assume_classification = TYPE_CODE_STRUCT;
			}
		tag_name_or_complete
			{
			  $$ = $3;
			}
	|	CLASS
			{
			  cpstate->assume_classification = TYPE_CODE_STRUCT;
			}
		tag_name_or_complete
			{
			  $$ = $3;
			}
	|	ENUM
			{
			  cpstate->assume_classification = TYPE_CODE_ENUM;
			}
		tag_name_or_complete
			{
			  $$ = $3;
			}
	|	UNION
			{
			  cpstate->assume_classification = TYPE_CODE_UNION;
			}
		tag_name_or_complete
			{
			  $$ = $3;
			}
		/* It appears that this rule for templates is never
		   reduced; template recognition happens by lookahead
		   in the token processing code in yylex. */
	|	TEMPLATE name '<' type '>'
			{ $$ = lookup_template_type
			    (copy_name($2).c_str (), $4,
			     pstate->expression_context_block);
			}
	|	qualifier_seq_noopt typebase
			{ $$ = cpstate->type_stack.follow_types ($2); }
	|	typebase qualifier_seq_noopt
			{ $$ = cpstate->type_stack.follow_types ($1); }
	;

type_name:	TYPENAME
	|	INT_KEYWORD
		{
		  $$.stoken.ptr = "int";
		  $$.stoken.length = 3;
		  $$.type = lookup_signed_typename (pstate->language (),
						    "int");
		}
	|	LONG
		{
		  $$.stoken.ptr = "long";
		  $$.stoken.length = 4;
		  $$.type = lookup_signed_typename (pstate->language (),
						    "long");
		}
	|	SHORT
		{
		  $$.stoken.ptr = "short";
		  $$.stoken.length = 5;
		  $$.type = lookup_signed_typename (pstate->language (),
						    "short");
		}
	;

parameter_typelist:
		nonempty_typelist
			{ check_parameter_typelist ($1); }
	|	nonempty_typelist ',' DOTDOTDOT
			{
			  $1->push_back (NULL);
			  check_parameter_typelist ($1);
			  $$ = $1;
			}
	;

nonempty_typelist
	:	type
		{
		  std::vector<struct type *> *typelist
		    = new std::vector<struct type *>;
		  cpstate->type_lists.emplace_back (typelist);

		  typelist->push_back ($1);
		  $$ = typelist;
		}
	|	nonempty_typelist ',' type
		{
		  $1->push_back ($3);
		  $$ = $1;
		}
	;

ptype	:	typebase
	|	ptype abs_decl
		{
		  cpstate->type_stack.push ($2);
		  $$ = cpstate->type_stack.follow_types ($1);
		}
	;

conversion_type_id: typebase conversion_declarator
		{ $$ = cpstate->type_stack.follow_types ($1); }
	;

conversion_declarator:  /* Nothing.  */
	| ptr_operator conversion_declarator
	;

const_and_volatile: 	CONST_KEYWORD VOLATILE_KEYWORD
	| 		VOLATILE_KEYWORD CONST_KEYWORD
	;

const_or_volatile_noopt:  	const_and_volatile
			{ cpstate->type_stack.insert (tp_const);
			  cpstate->type_stack.insert (tp_volatile);
			}
	| 		CONST_KEYWORD
			{ cpstate->type_stack.insert (tp_const); }
	| 		VOLATILE_KEYWORD
			{ cpstate->type_stack.insert (tp_volatile); }
	;

oper:	OPERATOR NEW
			{ $$ = operator_stoken (" new"); }
	|	OPERATOR DELETE
			{ $$ = operator_stoken (" delete"); }
	|	OPERATOR NEW '[' ']'
			{ $$ = operator_stoken (" new[]"); }
	|	OPERATOR DELETE '[' ']'
			{ $$ = operator_stoken (" delete[]"); }
	|	OPERATOR NEW OBJC_LBRAC ']'
			{ $$ = operator_stoken (" new[]"); }
	|	OPERATOR DELETE OBJC_LBRAC ']'
			{ $$ = operator_stoken (" delete[]"); }
	|	OPERATOR '+'
			{ $$ = operator_stoken ("+"); }
	|	OPERATOR '-'
			{ $$ = operator_stoken ("-"); }
	|	OPERATOR '*'
			{ $$ = operator_stoken ("*"); }
	|	OPERATOR '/'
			{ $$ = operator_stoken ("/"); }
	|	OPERATOR '%'
			{ $$ = operator_stoken ("%"); }
	|	OPERATOR '^'
			{ $$ = operator_stoken ("^"); }
	|	OPERATOR '&'
			{ $$ = operator_stoken ("&"); }
	|	OPERATOR '|'
			{ $$ = operator_stoken ("|"); }
	|	OPERATOR '~'
			{ $$ = operator_stoken ("~"); }
	|	OPERATOR '!'
			{ $$ = operator_stoken ("!"); }
	|	OPERATOR '='
			{ $$ = operator_stoken ("="); }
	|	OPERATOR '<'
			{ $$ = operator_stoken ("<"); }
	|	OPERATOR '>'
			{ $$ = operator_stoken (">"); }
	|	OPERATOR ASSIGN_MODIFY
			{ const char *op = " unknown";
			  switch ($2)
			    {
			    case BINOP_RSH:
			      op = ">>=";
			      break;
			    case BINOP_LSH:
			      op = "<<=";
			      break;
			    case BINOP_ADD:
			      op = "+=";
			      break;
			    case BINOP_SUB:
			      op = "-=";
			      break;
			    case BINOP_MUL:
			      op = "*=";
			      break;
			    case BINOP_DIV:
			      op = "/=";
			      break;
			    case BINOP_REM:
			      op = "%=";
			      break;
			    case BINOP_BITWISE_IOR:
			      op = "|=";
			      break;
			    case BINOP_BITWISE_AND:
			      op = "&=";
			      break;
			    case BINOP_BITWISE_XOR:
			      op = "^=";
			      break;
			    default:
			      break;
			    }

			  $$ = operator_stoken (op);
			}
	|	OPERATOR LSH
			{ $$ = operator_stoken ("<<"); }
	|	OPERATOR RSH
			{ $$ = operator_stoken (">>"); }
	|	OPERATOR EQUAL
			{ $$ = operator_stoken ("=="); }
	|	OPERATOR NOTEQUAL
			{ $$ = operator_stoken ("!="); }
	|	OPERATOR LEQ
			{ $$ = operator_stoken ("<="); }
	|	OPERATOR GEQ
			{ $$ = operator_stoken (">="); }
	|	OPERATOR ANDAND
			{ $$ = operator_stoken ("&&"); }
	|	OPERATOR OROR
			{ $$ = operator_stoken ("||"); }
	|	OPERATOR INCREMENT
			{ $$ = operator_stoken ("++"); }
	|	OPERATOR DECREMENT
			{ $$ = operator_stoken ("--"); }
	|	OPERATOR ','
			{ $$ = operator_stoken (","); }
	|	OPERATOR ARROW_STAR
			{ $$ = operator_stoken ("->*"); }
	|	OPERATOR ARROW
			{ $$ = operator_stoken ("->"); }
	|	OPERATOR '(' ')'
			{ $$ = operator_stoken ("()"); }
	|	OPERATOR '[' ']'
			{ $$ = operator_stoken ("[]"); }
	|	OPERATOR OBJC_LBRAC ']'
			{ $$ = operator_stoken ("[]"); }
	|	OPERATOR conversion_type_id
			{
			  string_file buf;
			  c_print_type ($2, NULL, &buf, -1, 0,
					pstate->language ()->la_language,
					&type_print_raw_options);
			  std::string name = buf.release ();

			  /* This also needs canonicalization.  */
			  gdb::unique_xmalloc_ptr<char> canon
			    = cp_canonicalize_string (name.c_str ());
			  if (canon != nullptr)
			    name = canon.get ();
			  $$ = operator_stoken ((" " + name).c_str ());
			}
	;

qual_field_name:
	qual_field_name COLONCOLON name
		{
		  $$.complete = false;
		  if ($1.prefix == nullptr)
		    $$.prefix = $1.name;
		  else
		    $$.prefix = obconcat (&cpstate->expansion_obstack,
					  $1.prefix, "::", $1.name, nullptr);
		  $$.name = obstack_strndup (&cpstate->expansion_obstack,
					     $3.ptr, $3.length);
		}
	| name
		{
		  $$.complete = false;
		  $$.prefix = nullptr;
		  $$.name = obstack_strndup (&cpstate->expansion_obstack,
					     $1.ptr, $1.length);
		}
	;

field_or_destructor:
	qual_field_name
	| qual_field_name COLONCOLON '~' name
		{
		  $$.complete = false;
		  if ($1.prefix == nullptr)
		    $$.prefix = $1.name;
		  else
		    $$.prefix = obconcat (&cpstate->expansion_obstack,
					  $1.prefix, "::", $1.name, nullptr);
		  char *name
		    = (char *) obstack_alloc (&cpstate->expansion_obstack,
					      $4.length + 2);
		  name[0] = '~';
		  memcpy (&name[1], $4.ptr, $4.length);
		  name[$4.length + 1] = '\0';
		  $$.name = name;
		}
	| '~' name
		{
		  $$.complete = false;
		  $$.prefix = nullptr;
		  char *name
		    = (char *) obstack_alloc (&cpstate->expansion_obstack,
					      $2.length + 2);
		  name[0] = '~';
		  memcpy (&name[1], $2.ptr, $2.length);
		  name[$2.length + 1] = '\0';
		  $$.name = name;
		}
	;

/* This rule exists in order to allow some tokens that would not normally
   match the 'name' rule to appear as fields within a struct.  The example
   that initially motivated this was the RISC-V target which models the
   floating point registers as a union with fields called 'float' and
   'double'.  */
field_name
	:	field_or_destructor
	|	DOUBLE_KEYWORD { $$ = typename_stoken ("double"); }
	|	FLOAT_KEYWORD { $$ = typename_stoken ("float"); }
	|	INT_KEYWORD { $$ = typename_stoken ("int"); }
	|	LONG { $$ = typename_stoken ("long"); }
	|	SHORT { $$ = typename_stoken ("short"); }
	|	SIGNED_KEYWORD { $$ = typename_stoken ("signed"); }
	|	UNSIGNED { $$ = typename_stoken ("unsigned"); }
	;

field_name_or_complete
	:	field_name
	|	field_name COMPLETE
			{
			  $$ = $1;
			  $$.complete = true;
			}
	|	COMPLETE
			{
			  $$ = typename_stoken ("");
			  $$.complete = true;
			}
	;

/* This rule is used when the preceding token is a keyword that takes
   a tag name (e.g., "struct").  The "caller" should disable name
   lookup, see c_parse_state::assume_classification.  */
tag_name_or_complete
	:	NAME
		{
		  switch (cpstate->assume_classification)
		    {
		    case TYPE_CODE_STRUCT:
		      $$ = lookup_struct (copy_name ($1.stoken).c_str (),
					  pstate->expression_context_block);
		      break;
		    case TYPE_CODE_ENUM:
		      $$ = lookup_enum (copy_name ($1.stoken).c_str (),
					pstate->expression_context_block);
		      break;
		    case TYPE_CODE_UNION:
		      $$ = lookup_union (copy_name ($1.stoken).c_str (),
					 pstate->expression_context_block);
		      break;
		    default:
		      gdb_assert_not_reached ();
		    }
		  cpstate->assume_classification = TYPE_CODE_UNDEF;
		}
	|	COMPLETE
		{
		  pstate->mark_completion_tag (cpstate->assume_classification,
					       "", 0);
		  cpstate->assume_classification = TYPE_CODE_UNDEF;
		  $$ = nullptr;
		}
	|	NAME COMPLETE
		{
		  pstate->mark_completion_tag (cpstate->assume_classification,
					       $1.stoken.ptr, $1.stoken.length);
		  cpstate->assume_classification = TYPE_CODE_UNDEF;
		  $$ = nullptr;
		}
	;

name	:	NAME { $$ = $1.stoken; }
	|	BLOCKNAME { $$ = $1.stoken; }
	|	TYPENAME { $$ = $1.stoken; }
	|	NAME_OR_INT  { $$ = $1.stoken; }
	|	UNKNOWN_CPP_NAME  { $$ = $1.stoken; }
	|	oper { $$ = $1; }
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
	|	oper
			{
			  struct field_of_this_result is_a_field_of_this;

			  $$.stoken = $1;
			  $$.sym
			    = lookup_symbol ($1.ptr,
					     pstate->expression_context_block,
					     SEARCH_VFT,
					     &is_a_field_of_this);
			  $$.is_a_field_of_this
			    = is_a_field_of_this.type != NULL;
			}
	|	UNKNOWN_CPP_NAME
	;

%%

#if defined(YYBISON) && YYBISON < 30800

/* This is called via the YYPRINT macro when parser debugging is
   enabled.  It prints a token's value.  */

static void
c_print_token (FILE *file, int type, YYSTYPE value)
{
  switch (type)
    {
    case INT:
      parser_fprintf (file, "typed_val_int<%s, %s>",
		      value.typed_val_int.type->safe_name (),
		      pulongest (value.typed_val_int.val));
      break;

    case CHAR:
    case STRING:
      parser_fprintf (file, "tsval<type=%d, %.*s>", value.tsval.type,
		      value.tsval.length, value.tsval.ptr);
      break;

    case NSSTRING:
    case DOLLAR_VARIABLE:
    case SELECTOR:
      parser_fprintf (file, "sval<%s>", copy_name (value.sval).c_str ());
      break;

    case TYPENAME:
      parser_fprintf (file, "tsym<type=%s, name=%s>",
		      value.tsym.type->safe_name (),
		      copy_name (value.tsym.stoken).c_str ());
      break;

    case NAME:
    case UNKNOWN_CPP_NAME:
    case NAME_OR_INT:
    case BLOCKNAME:
      parser_fprintf (file, "ssym<name=%s, sym=%s, field_of_this=%d>",
		       copy_name (value.ssym.stoken).c_str (),
		       (value.ssym.sym.symbol == NULL
			? "(null)" : value.ssym.sym.symbol->print_name ()),
		       value.ssym.is_a_field_of_this);
      break;

    case FILENAME:
      parser_fprintf (file, "bval<%s>", host_address_to_string (value.bval));
      break;
    }
}

#endif
