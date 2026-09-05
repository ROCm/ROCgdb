/* YACC parser for C++ names, for GDB.

   Copyright (C) 2003-2026 Free Software Foundation, Inc.

   Parts of the lexer are based on c-exp.y from GDB.

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

/* Note that malloc's and realloc's in this file are transformed to
   xmalloc and xrealloc respectively by the same sed command in the
   makefile that remaps any other malloc/realloc inserted by the parser
   generator.  Doing this with #defines and trying to control the interaction
   with include files (<malloc.h> and <stdlib.h> for example) just became
   too messy, particularly when such includes can be inserted at random
   times by the parser generator.  */

/* The Bison manual says that %pure-parser is deprecated, but we use
   it anyway because it also works with Byacc.  That is also why
   this uses %lex-param and %parse-param rather than the simpler
   %param -- Byacc does not support the latter.  */
%pure-parser
%lex-param {cp_name_parser::cpname_state *state}
%parse-param {cp_name_parser::cpname_state *state}

%{

#include "demangle.h"
#include "cp-name-parser.h"
#include "cp-support.h"

%}

%union
  {
    struct demangle_component *comp;
    struct nested {
      struct demangle_component *comp;
      struct demangle_component **last;
    } nested;
    struct {
      struct demangle_component *comp, *last;
    } nested1;
    struct {
      struct demangle_component *comp, **last;
      struct nested fn;
      struct demangle_component *start;
      int fold_flag;
    } abstract;
    int lval;
    const char *opname;
  }

%type <comp> exp exp1 type start start_opt oper colon_name
%type <comp> unqualified_name colon_ext_name
%type <comp> templ template_arg
%type <comp> builtin_type
%type <comp> typespec_2 array_indicator
%type <comp> colon_ext_only ext_only_name

%type <comp> demangler_special function conversion_op
%type <nested> conversion_op_name

%type <abstract> abstract_declarator direct_abstract_declarator
%type <abstract> abstract_declarator_fn
%type <nested> declarator direct_declarator function_arglist

%type <nested> declarator_1 direct_declarator_1

%type <nested> template_params function_args
%type <nested> ptr_operator

%type <nested1> nested_name

%type <lval> qualifier qualifiers qualifiers_opt

%type <lval> int_part int_seq

%token <comp> INT
%token <comp> FLOAT

%token <comp> NAME
%type <comp> name

%token STRUCT CLASS UNION ENUM SIZEOF UNSIGNED COLONCOLON
%token TEMPLATE
%token ERROR
%token NEW DELETE OPERATOR
%token STATIC_CAST REINTERPRET_CAST DYNAMIC_CAST

/* Special type cases, put in to allow the parser to distinguish different
   legal basetypes.  */
%token SIGNED_KEYWORD LONG SHORT INT_KEYWORD CONST_KEYWORD VOLATILE_KEYWORD DOUBLE_KEYWORD BOOL
%token ELLIPSIS RESTRICT VOID FLOAT_KEYWORD CHAR WCHAR_T

%token <opname> ASSIGN_MODIFY

/* C++ */
%token TRUEKEYWORD
%token FALSEKEYWORD

/* Non-C++ things we get from the demangler.  */
%token <lval> DEMANGLER_SPECIAL
%token CONSTRUCTION_VTABLE CONSTRUCTION_IN

/* Precedence declarations.  */

/* Give NAME lower precedence than COLONCOLON, so that nested_name will
   associate greedily.  */
%nonassoc NAME

/* Give NEW and DELETE lower precedence than ']', because we can not
   have an array of type operator new.  This causes NEW '[' to be
   parsed as operator new[].  */
%nonassoc NEW DELETE

/* Give VOID higher precedence than NAME.  Then we can use %prec NAME
   to prefer (VOID) to (function_args).  */
%nonassoc VOID

/* Give VOID lower precedence than ')' for similar reasons.  */
%nonassoc ')'

%left ','
%right '=' ASSIGN_MODIFY
%right '?'
%left OROR
%left ANDAND
%left '|'
%left '^'
%left '&'
%left EQUAL NOTEQUAL
%left '<' '>' LEQ GEQ SPACESHIP
%left LSH RSH
%left '@'
%left '+' '-'
%left '*' '/' '%'
%right UNARY INCREMENT DECREMENT

/* We don't need a precedence for '(' in this reduced grammar, and it
   can mask some unpleasant bugs, so disable it for now.  */

%right ARROW '.' '[' /* '(' */
%left COLONCOLON


%%

result		:	start
			{
			  state->global_result = $1;

			  /* Avoid warning about "yynerrs" being unused.  */
			  (void) yynerrs;
			}
		;

start		:	type

		|	demangler_special

		|	function

		;

start_opt	:	/* */
			{ $$ = NULL; }
		|	COLONCOLON start
			{ $$ = $2; }
		;

function
		/* Function with a return type.  declarator_1 is used to prevent
		   ambiguity with the next rule.  */
		:	typespec_2 declarator_1
			{ $$ = $2.comp;
			  *$2.last = $1;
			}

		/* Function without a return type.  We need to use typespec_2
		   to prevent conflicts from qualifiers_opt - harmless.  The
		   start_opt is used to handle "function-local" variables and
		   types.  */
		|	typespec_2 function_arglist start_opt
			{ $$ = state->fill_comp (DEMANGLE_COMPONENT_TYPED_NAME,
					  $1, $2.comp);
			  if ($3)
			    $$ = state->fill_comp (DEMANGLE_COMPONENT_LOCAL_NAME,
						   $$, $3);
			}
		|	colon_ext_only function_arglist start_opt
			{ $$ = state->fill_comp (DEMANGLE_COMPONENT_TYPED_NAME, $1, $2.comp);
			  if ($3) $$ = state->fill_comp (DEMANGLE_COMPONENT_LOCAL_NAME, $$, $3); }
		|	colon_ext_only
			{
			  /* This production is a hack to handle
			     something like "name::operator new[]" --
			     without arguments, this ordinarily would
			     not parse, but canonicalizing it is
			     important.  So we infer the "()" and then
			     remove it when converting back to string.
			     Note that this works because this
			     production is terminal.  */
			  demangle_component *comp
			    = state->fill_comp (DEMANGLE_COMPONENT_FUNCTION_TYPE,
						nullptr, nullptr);
			  $$ = state->fill_comp (DEMANGLE_COMPONENT_TYPED_NAME, $1, comp);
			  state->demangle_info->added_parens = true;
			}

		|	conversion_op_name start_opt
			{ $$ = $1.comp;
			  if ($2) $$ = state->fill_comp (DEMANGLE_COMPONENT_LOCAL_NAME, $$, $2); }
		|	conversion_op_name abstract_declarator_fn
			{ if ($2.last)
			    {
			       /* First complete the abstract_declarator's type using
				  the typespec from the conversion_op_name.  */
			      *$2.last = *$1.last;
			      /* Then complete the conversion_op_name with the type.  */
			      *$1.last = $2.comp;
			    }
			  /* If we have an arglist, build a function type.  */
			  if ($2.fn.comp)
			    $$ = state->fill_comp (DEMANGLE_COMPONENT_TYPED_NAME, $1.comp, $2.fn.comp);
			  else
			    $$ = $1.comp;
			  if ($2.start) $$ = state->fill_comp (DEMANGLE_COMPONENT_LOCAL_NAME, $$, $2.start);
			}
		;

demangler_special
		:	DEMANGLER_SPECIAL start
			{ $$ = state->fill_comp ((enum demangle_component_type) $1, $2, NULL); }
		|	CONSTRUCTION_VTABLE start CONSTRUCTION_IN start
			{ $$ = state->fill_comp (DEMANGLE_COMPONENT_CONSTRUCTION_VTABLE, $2, $4); }
		;

oper	:	OPERATOR NEW
			{
			  /* Match the whitespacing of cplus_demangle_operators.
			     It would abort on unrecognized string otherwise.  */
			  $$ = state->make_operator ("new", 3);
			}
		|	OPERATOR DELETE
			{
			  /* Match the whitespacing of cplus_demangle_operators.
			     It would abort on unrecognized string otherwise.  */
			  $$ = state->make_operator ("delete ", 1);
			}
		|	OPERATOR NEW '[' ']'
			{
			  /* Match the whitespacing of cplus_demangle_operators.
			     It would abort on unrecognized string otherwise.  */
			  $$ = state->make_operator ("new[]", 3);
			}
		|	OPERATOR DELETE '[' ']'
			{
			  /* Match the whitespacing of cplus_demangle_operators.
			     It would abort on unrecognized string otherwise.  */
			  $$ = state->make_operator ("delete[] ", 1);
			}
		|	OPERATOR '+'
			{ $$ = state->make_operator ("+", 2); }
		|	OPERATOR '-'
			{ $$ = state->make_operator ("-", 2); }
		|	OPERATOR '*'
			{ $$ = state->make_operator ("*", 2); }
		|	OPERATOR '/'
			{ $$ = state->make_operator ("/", 2); }
		|	OPERATOR '%'
			{ $$ = state->make_operator ("%", 2); }
		|	OPERATOR '^'
			{ $$ = state->make_operator ("^", 2); }
		|	OPERATOR '&'
			{ $$ = state->make_operator ("&", 2); }
		|	OPERATOR '|'
			{ $$ = state->make_operator ("|", 2); }
		|	OPERATOR '~'
			{ $$ = state->make_operator ("~", 1); }
		|	OPERATOR '!'
			{ $$ = state->make_operator ("!", 1); }
		|	OPERATOR '='
			{ $$ = state->make_operator ("=", 2); }
		|	OPERATOR '<'
			{ $$ = state->make_operator ("<", 2); }
		|	OPERATOR '>'
			{ $$ = state->make_operator (">", 2); }
		|	OPERATOR ASSIGN_MODIFY
			{ $$ = state->make_operator ($2, 2); }
		|	OPERATOR LSH
			{ $$ = state->make_operator ("<<", 2); }
		|	OPERATOR RSH
			{ $$ = state->make_operator (">>", 2); }
		|	OPERATOR EQUAL
			{ $$ = state->make_operator ("==", 2); }
		|	OPERATOR NOTEQUAL
			{ $$ = state->make_operator ("!=", 2); }
		|	OPERATOR LEQ
			{ $$ = state->make_operator ("<=", 2); }
		|	OPERATOR GEQ
			{ $$ = state->make_operator (">=", 2); }
		|	OPERATOR SPACESHIP
			{ $$ = state->make_operator ("<=>", 2); }
		|	OPERATOR ANDAND
			{ $$ = state->make_operator ("&&", 2); }
		|	OPERATOR OROR
			{ $$ = state->make_operator ("||", 2); }
		|	OPERATOR INCREMENT
			{ $$ = state->make_operator ("++", 1); }
		|	OPERATOR DECREMENT
			{ $$ = state->make_operator ("--", 1); }
		|	OPERATOR ','
			{ $$ = state->make_operator (",", 2); }
		|	OPERATOR ARROW '*'
			{ $$ = state->make_operator ("->*", 2); }
		|	OPERATOR ARROW
			{ $$ = state->make_operator ("->", 2); }
		|	OPERATOR '(' ')'
			{ $$ = state->make_operator ("()", 2); }
		|	OPERATOR '[' ']'
			{ $$ = state->make_operator ("[]", 2); }
		;

		/* Conversion operators.  We don't try to handle some of
		   the wackier demangler output for function pointers,
		   since it's not clear that it's parseable.  */
conversion_op
		:	OPERATOR typespec_2
			{ $$ = state->fill_comp (DEMANGLE_COMPONENT_CONVERSION, $2, NULL); }
		;

conversion_op_name
		:	nested_name conversion_op
			{ $$.comp = $1.comp;
			  d_right ($1.last) = $2;
			  $$.last = &d_left ($2);
			}
		|	conversion_op
			{ $$.comp = $1;
			  $$.last = &d_left ($1);
			}
		|	COLONCOLON nested_name conversion_op
			{ $$.comp = $2.comp;
			  d_right ($2.last) = $3;
			  $$.last = &d_left ($3);
			}
		|	COLONCOLON conversion_op
			{ $$.comp = $2;
			  $$.last = &d_left ($2);
			}
		;

/* DEMANGLE_COMPONENT_NAME */
/* This accepts certain invalid placements of '~'.  */
unqualified_name:	oper
		|	oper '<' template_params '>'
			{ $$ = state->fill_comp (DEMANGLE_COMPONENT_TEMPLATE, $1, $3.comp); }
		|	oper '<' template_params RSH
			{
			  $$ = state->fill_comp (DEMANGLE_COMPONENT_TEMPLATE, $1, $3.comp);
			  state->unpush ('>');
			}
		|	'~' NAME
			{ $$ = state->make_dtor (gnu_v3_complete_object_dtor, $2); }
		;

/* This rule is used in name and nested_name, and expanded inline there
   for efficiency.  */
/*
scope_id	:	NAME
		|	template
		;
*/

colon_name	:	name
		|	COLONCOLON name
			{ $$ = $2; }
		;

/* DEMANGLE_COMPONENT_QUAL_NAME */
/* DEMANGLE_COMPONENT_CTOR / DEMANGLE_COMPONENT_DTOR ? */
name		:	nested_name NAME %prec NAME
			{ $$ = $1.comp; d_right ($1.last) = $2; }
		|	NAME %prec NAME
		|	nested_name templ %prec NAME
			{ $$ = $1.comp; d_right ($1.last) = $2; }
		|	templ %prec NAME
		;

colon_ext_name	:	colon_name
		|	colon_ext_only
		;

colon_ext_only	:	ext_only_name
		|	COLONCOLON ext_only_name
			{ $$ = $2; }
		;

ext_only_name	:	nested_name unqualified_name
			{ $$ = $1.comp; d_right ($1.last) = $2; }
		|	unqualified_name
		;

nested_name	:	NAME COLONCOLON
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_QUAL_NAME, $1, NULL);
			  $$.last = $$.comp;
			}
		|	nested_name NAME COLONCOLON
			{ $$.comp = $1.comp;
			  d_right ($1.last) = state->fill_comp (DEMANGLE_COMPONENT_QUAL_NAME, $2, NULL);
			  $$.last = d_right ($1.last);
			}
		|	templ COLONCOLON
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_QUAL_NAME, $1, NULL);
			  $$.last = $$.comp;
			}
		|	nested_name templ COLONCOLON
			{ $$.comp = $1.comp;
			  d_right ($1.last) = state->fill_comp (DEMANGLE_COMPONENT_QUAL_NAME, $2, NULL);
			  $$.last = d_right ($1.last);
			}
		;

/* DEMANGLE_COMPONENT_TEMPLATE */
/* DEMANGLE_COMPONENT_TEMPLATE_ARGLIST */
templ	:	NAME '<' template_params '>'
			{ $$ = state->fill_comp (DEMANGLE_COMPONENT_TEMPLATE, $1, $3.comp); }
		| NAME '<' template_params RSH
			{
			  $$ = state->fill_comp (DEMANGLE_COMPONENT_TEMPLATE, $1, $3.comp);
			  state->unpush ('>');
			}
		;

template_params	:	template_arg
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_TEMPLATE_ARGLIST, $1, NULL);
			$$.last = &d_right ($$.comp); }
		|	template_params ',' template_arg
			{ $$.comp = $1.comp;
			  *$1.last = state->fill_comp (DEMANGLE_COMPONENT_TEMPLATE_ARGLIST, $3, NULL);
			  $$.last = &d_right (*$1.last);
			}
		;

/* "type" is inlined into template_arg and function_args.  */

/* Also an integral constant-expression of integral type, and a
   pointer to member (?) */
template_arg	:	typespec_2
		|	typespec_2 abstract_declarator
			{ $$ = $2.comp;
			  *$2.last = $1;
			}
		|	'&' start
			{ $$ = state->fill_comp (DEMANGLE_COMPONENT_UNARY, state->make_operator ("&", 1), $2); }
		|	'&' '(' start ')'
			{ $$ = state->fill_comp (DEMANGLE_COMPONENT_UNARY, state->make_operator ("&", 1), $3); }
		|	exp
		|	function
		;

function_args	:	typespec_2
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_ARGLIST, $1, NULL);
			  $$.last = &d_right ($$.comp);
			}
		|	typespec_2 abstract_declarator
			{ *$2.last = $1;
			  $$.comp = state->fill_comp (DEMANGLE_COMPONENT_ARGLIST, $2.comp, NULL);
			  $$.last = &d_right ($$.comp);
			}
		|	function_args ',' typespec_2
			{ *$1.last = state->fill_comp (DEMANGLE_COMPONENT_ARGLIST, $3, NULL);
			  $$.comp = $1.comp;
			  $$.last = &d_right (*$1.last);
			}
		|	function_args ',' typespec_2 abstract_declarator
			{ *$4.last = $3;
			  *$1.last = state->fill_comp (DEMANGLE_COMPONENT_ARGLIST, $4.comp, NULL);
			  $$.comp = $1.comp;
			  $$.last = &d_right (*$1.last);
			}
		|	function_args ',' ELLIPSIS
			{ *$1.last
			    = state->fill_comp (DEMANGLE_COMPONENT_ARGLIST,
					   state->make_builtin_type ("..."),
					   NULL);
			  $$.comp = $1.comp;
			  $$.last = &d_right (*$1.last);
			}
		;

function_arglist:	'(' function_args ')' qualifiers_opt %prec NAME
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_FUNCTION_TYPE, NULL, $2.comp);
			  $$.last = &d_left ($$.comp);
			  $$.comp = state->d_qualify ($$.comp, $4, 1); }
		|	'(' VOID ')' qualifiers_opt
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_FUNCTION_TYPE, NULL, NULL);
			  $$.last = &d_left ($$.comp);
			  $$.comp = state->d_qualify ($$.comp, $4, 1); }
		|	'(' ')' qualifiers_opt
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_FUNCTION_TYPE, NULL, NULL);
			  $$.last = &d_left ($$.comp);
			  $$.comp = state->d_qualify ($$.comp, $3, 1); }
		;

/* Should do something about DEMANGLE_COMPONENT_VENDOR_TYPE_QUAL */
qualifiers_opt	:	/* epsilon */
			{ $$ = 0; }
		|	qualifiers
		;

qualifier	:	RESTRICT
			{ $$ = QUAL_RESTRICT; }
		|	VOLATILE_KEYWORD
			{ $$ = QUAL_VOLATILE; }
		|	CONST_KEYWORD
			{ $$ = QUAL_CONST; }
		;

qualifiers	:	qualifier
		|	qualifier qualifiers
			{ $$ = $1 | $2; }
		;

/* This accepts all sorts of invalid constructions and produces
   invalid output for them - an error would be better.  */

int_part	:	INT_KEYWORD
			{ $$ = 0; }
		|	SIGNED_KEYWORD
			{ $$ = INT_SIGNED; }
		|	UNSIGNED
			{ $$ = INT_UNSIGNED; }
		|	CHAR
			{ $$ = INT_CHAR; }
		|	LONG
			{ $$ = INT_LONG; }
		|	SHORT
			{ $$ = INT_SHORT; }
		;

int_seq		:	int_part
		|	int_seq int_part
			{ $$ = $1 | $2; if ($1 & $2 & INT_LONG) $$ = $1 | INT_LLONG; }
		;

builtin_type	:	int_seq
			{ $$ = state->d_int_type ($1); }
		|	FLOAT_KEYWORD
			{ $$ = state->make_builtin_type ("float"); }
		|	DOUBLE_KEYWORD
			{ $$ = state->make_builtin_type ("double"); }
		|	LONG DOUBLE_KEYWORD
			{ $$ = state->make_builtin_type ("long double"); }
		|	BOOL
			{ $$ = state->make_builtin_type ("bool"); }
		|	WCHAR_T
			{ $$ = state->make_builtin_type ("wchar_t"); }
		|	VOID
			{ $$ = state->make_builtin_type ("void"); }
		;

ptr_operator	:	'*' qualifiers_opt
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_POINTER, NULL, NULL);
			  $$.last = &d_left ($$.comp);
			  $$.comp = state->d_qualify ($$.comp, $2, 0); }
		/* g++ seems to allow qualifiers after the reference?  */
		|	'&'
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_REFERENCE, NULL, NULL);
			  $$.last = &d_left ($$.comp); }
		|	ANDAND
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_RVALUE_REFERENCE, NULL, NULL);
			  $$.last = &d_left ($$.comp); }
		|	nested_name '*' qualifiers_opt
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_PTRMEM_TYPE, $1.comp, NULL);
			  /* Convert the innermost DEMANGLE_COMPONENT_QUAL_NAME to a DEMANGLE_COMPONENT_NAME.  */
			  *$1.last = *d_left ($1.last);
			  $$.last = &d_right ($$.comp);
			  $$.comp = state->d_qualify ($$.comp, $3, 0); }
		|	COLONCOLON nested_name '*' qualifiers_opt
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_PTRMEM_TYPE, $2.comp, NULL);
			  /* Convert the innermost DEMANGLE_COMPONENT_QUAL_NAME to a DEMANGLE_COMPONENT_NAME.  */
			  *$2.last = *d_left ($2.last);
			  $$.last = &d_right ($$.comp);
			  $$.comp = state->d_qualify ($$.comp, $4, 0); }
		;

array_indicator	:	'[' ']'
			{ $$ = state->fill_comp (DEMANGLE_COMPONENT_ARRAY_TYPE, NULL, NULL); }
		|	'[' INT ']'
			{ $$ = state->fill_comp (DEMANGLE_COMPONENT_ARRAY_TYPE, $2, NULL); }
		;

/* Details of this approach inspired by the G++ < 3.4 parser.  */

/* This rule is only used in typespec_2, and expanded inline there for
   efficiency.  */
/*
typespec	:	builtin_type
		|	colon_name
		;
*/

typespec_2	:	builtin_type qualifiers
			{ $$ = state->d_qualify ($1, $2, 0); }
		|	builtin_type
		|	qualifiers builtin_type qualifiers
			{ $$ = state->d_qualify ($2, $1 | $3, 0); }
		|	qualifiers builtin_type
			{ $$ = state->d_qualify ($2, $1, 0); }

		|	name qualifiers
			{ $$ = state->d_qualify ($1, $2, 0); }
		|	name
		|	qualifiers name qualifiers
			{ $$ = state->d_qualify ($2, $1 | $3, 0); }
		|	qualifiers name
			{ $$ = state->d_qualify ($2, $1, 0); }

		|	COLONCOLON name qualifiers
			{ $$ = state->d_qualify ($2, $3, 0); }
		|	COLONCOLON name
			{ $$ = $2; }
		|	qualifiers COLONCOLON name qualifiers
			{ $$ = state->d_qualify ($3, $1 | $4, 0); }
		|	qualifiers COLONCOLON name
			{ $$ = state->d_qualify ($3, $1, 0); }
		;

abstract_declarator
		:	ptr_operator
			{ $$.comp = $1.comp; $$.last = $1.last;
			  $$.fn.comp = NULL; $$.fn.last = NULL; }
		|	ptr_operator abstract_declarator
			{ $$ = $2; $$.fn.comp = NULL; $$.fn.last = NULL;
			  if ($2.fn.comp) { $$.last = $2.fn.last; *$2.last = $2.fn.comp; }
			  *$$.last = $1.comp;
			  $$.last = $1.last; }
		|	direct_abstract_declarator
			{ $$.fn.comp = NULL; $$.fn.last = NULL;
			  if ($1.fn.comp) { $$.last = $1.fn.last; *$1.last = $1.fn.comp; }
			}
		;

direct_abstract_declarator
		:	'(' abstract_declarator ')'
			{ $$ = $2; $$.fn.comp = NULL; $$.fn.last = NULL; $$.fold_flag = 1;
			  if ($2.fn.comp) { $$.last = $2.fn.last; *$2.last = $2.fn.comp; }
			}
		|	direct_abstract_declarator function_arglist
			{ $$.fold_flag = 0;
			  if ($1.fn.comp) { $$.last = $1.fn.last; *$1.last = $1.fn.comp; }
			  if ($1.fold_flag)
			    {
			      *$$.last = $2.comp;
			      $$.last = $2.last;
			    }
			  else
			    $$.fn = $2;
			}
		|	direct_abstract_declarator array_indicator
			{ $$.fn.comp = NULL; $$.fn.last = NULL; $$.fold_flag = 0;
			  if ($1.fn.comp) { $$.last = $1.fn.last; *$1.last = $1.fn.comp; }
			  *$1.last = $2;
			  $$.last = &d_right ($2);
			}
		|	array_indicator
			{ $$.fn.comp = NULL; $$.fn.last = NULL; $$.fold_flag = 0;
			  $$.comp = $1;
			  $$.last = &d_right ($1);
			}
		/* G++ has the following except for () and (type).  Then
		   (type) is handled in regcast_or_absdcl and () is handled
		   in fcast_or_absdcl.

		   However, this is only useful for function types, and
		   generates reduce/reduce conflicts with direct_declarator.
		   We're interested in pointer-to-function types, and in
		   functions, but not in function types - so leave this
		   out.  */
		/* |	function_arglist */
		;

abstract_declarator_fn
		:	ptr_operator
			{ $$.comp = $1.comp; $$.last = $1.last;
			  $$.fn.comp = NULL; $$.fn.last = NULL; $$.start = NULL; }
		|	ptr_operator abstract_declarator_fn
			{ $$ = $2;
			  if ($2.last)
			    *$$.last = $1.comp;
			  else
			    $$.comp = $1.comp;
			  $$.last = $1.last;
			}
		|	direct_abstract_declarator
			{ $$.comp = $1.comp; $$.last = $1.last; $$.fn = $1.fn; $$.start = NULL; }
		|	direct_abstract_declarator function_arglist COLONCOLON start
			{ $$.start = $4;
			  if ($1.fn.comp) { $$.last = $1.fn.last; *$1.last = $1.fn.comp; }
			  if ($1.fold_flag)
			    {
			      *$$.last = $2.comp;
			      $$.last = $2.last;
			    }
			  else
			    $$.fn = $2;
			}
		|	function_arglist start_opt
			{ $$.fn = $1;
			  $$.start = $2;
			  $$.comp = NULL; $$.last = NULL;
			}
		;

type		:	typespec_2
		|	typespec_2 abstract_declarator
			{ $$ = $2.comp;
			  *$2.last = $1;
			}
		;

declarator	:	ptr_operator declarator
			{ $$.comp = $2.comp;
			  $$.last = $1.last;
			  *$2.last = $1.comp; }
		|	direct_declarator
		;

direct_declarator
		:	'(' declarator ')'
			{ $$ = $2; }
		|	direct_declarator function_arglist
			{ $$.comp = $1.comp;
			  *$1.last = $2.comp;
			  $$.last = $2.last;
			}
		|	direct_declarator array_indicator
			{ $$.comp = $1.comp;
			  *$1.last = $2;
			  $$.last = &d_right ($2);
			}
		|	colon_ext_name
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_TYPED_NAME, $1, NULL);
			  $$.last = &d_right ($$.comp);
			}
		;

/* These are similar to declarator and direct_declarator except that they
   do not permit ( colon_ext_name ), which is ambiguous with a function
   argument list.  They also don't permit a few other forms with redundant
   parentheses around the colon_ext_name; any colon_ext_name in parentheses
   must be followed by an argument list or an array indicator, or preceded
   by a pointer.  */
declarator_1	:	ptr_operator declarator_1
			{ $$.comp = $2.comp;
			  $$.last = $1.last;
			  *$2.last = $1.comp; }
		|	colon_ext_name
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_TYPED_NAME, $1, NULL);
			  $$.last = &d_right ($$.comp);
			}
		|	direct_declarator_1

			/* Function local variable or type.  The typespec to
			   our left is the type of the containing function.
			   This should be OK, because function local types
			   can not be templates, so the return types of their
			   members will not be mangled.  If they are hopefully
			   they'll end up to the right of the ::.  */
		|	colon_ext_name function_arglist COLONCOLON start
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_TYPED_NAME, $1, $2.comp);
			  $$.last = $2.last;
			  $$.comp = state->fill_comp (DEMANGLE_COMPONENT_LOCAL_NAME, $$.comp, $4);
			}
		|	direct_declarator_1 function_arglist COLONCOLON start
			{ $$.comp = $1.comp;
			  *$1.last = $2.comp;
			  $$.last = $2.last;
			  $$.comp = state->fill_comp (DEMANGLE_COMPONENT_LOCAL_NAME, $$.comp, $4);
			}
		;

direct_declarator_1
		:	'(' ptr_operator declarator ')'
			{ $$.comp = $3.comp;
			  $$.last = $2.last;
			  *$3.last = $2.comp; }
		|	direct_declarator_1 function_arglist
			{ $$.comp = $1.comp;
			  *$1.last = $2.comp;
			  $$.last = $2.last;
			}
		|	direct_declarator_1 array_indicator
			{ $$.comp = $1.comp;
			  *$1.last = $2;
			  $$.last = &d_right ($2);
			}
		|	colon_ext_name function_arglist
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_TYPED_NAME, $1, $2.comp);
			  $$.last = $2.last;
			}
		|	colon_ext_name array_indicator
			{ $$.comp = state->fill_comp (DEMANGLE_COMPONENT_TYPED_NAME, $1, $2);
			  $$.last = &d_right ($2);
			}
		;

exp	:	'(' exp1 ')'
		{ $$ = $2; }
	;

/* Silly trick.  Only allow '>' when parenthesized, in order to
   handle conflict with templates.  */
exp1	:	exp
	;

exp1	:	exp '>' exp
		{ $$ = state->d_binary (">", $1, $3); }
	;

/* References.  Not allowed everywhere in template parameters, only
   at the top level, but treat them as expressions in case they are wrapped
   in parentheses.  */
exp1	:	'&' start
		{ $$ = state->fill_comp (DEMANGLE_COMPONENT_UNARY, state->make_operator ("&", 1), $2); }
	|	'&' '(' start ')'
		{ $$ = state->fill_comp (DEMANGLE_COMPONENT_UNARY, state->make_operator ("&", 1), $3); }
	;

/* Expressions, not including the comma operator.  */
exp	:	'-' exp    %prec UNARY
		{ $$ = state->d_unary ("-", $2); }
	;

exp	:	'!' exp    %prec UNARY
		{ $$ = state->d_unary ("!", $2); }
	;

exp	:	'~' exp    %prec UNARY
		{ $$ = state->d_unary ("~", $2); }
	;

/* Casts.  First your normal C-style cast.  If exp is a LITERAL, just change
   its type.  */

exp	:	'(' type ')' exp  %prec UNARY
		{ if ($4->type == DEMANGLE_COMPONENT_LITERAL
		      || $4->type == DEMANGLE_COMPONENT_LITERAL_NEG)
		    {
		      $$ = $4;
		      d_left ($4) = $2;
		    }
		  else
		    $$ = state->fill_comp (DEMANGLE_COMPONENT_UNARY,
				      state->fill_comp (DEMANGLE_COMPONENT_CAST, $2, NULL),
				      $4);
		}
	;

/* Mangling does not differentiate between these, so we don't need to
   either.  */
exp	:	STATIC_CAST '<' type '>' '(' exp1 ')' %prec UNARY
		{ $$ = state->fill_comp (DEMANGLE_COMPONENT_UNARY,
				    state->fill_comp (DEMANGLE_COMPONENT_CAST, $3, NULL),
				    $6);
		}
	;

exp	:	DYNAMIC_CAST '<' type '>' '(' exp1 ')' %prec UNARY
		{ $$ = state->fill_comp (DEMANGLE_COMPONENT_UNARY,
				    state->fill_comp (DEMANGLE_COMPONENT_CAST, $3, NULL),
				    $6);
		}
	;

exp	:	REINTERPRET_CAST '<' type '>' '(' exp1 ')' %prec UNARY
		{ $$ = state->fill_comp (DEMANGLE_COMPONENT_UNARY,
				    state->fill_comp (DEMANGLE_COMPONENT_CAST, $3, NULL),
				    $6);
		}
	;

/* Another form of C++-style cast is "type ( exp1 )".  This creates too many
   conflicts to support.  For a while we supported the simpler
   "typespec_2 ( exp1 )", but that conflicts with "& ( start )" as a
   reference, deep within the wilderness of abstract declarators:
   Qux<int(&(*))> vs Qux<int(&(var))>, a shift-reduce conflict at the
   innermost left parenthesis.  So we do not support function-like casts.
   Fortunately they never appear in demangler output.  */

/* TO INVESTIGATE: ._0 style anonymous names; anonymous namespaces */

/* Binary operators in order of decreasing precedence.  */

exp	:	exp '*' exp
		{ $$ = state->d_binary ("*", $1, $3); }
	;

exp	:	exp '/' exp
		{ $$ = state->d_binary ("/", $1, $3); }
	;

exp	:	exp '%' exp
		{ $$ = state->d_binary ("%", $1, $3); }
	;

exp	:	exp '+' exp
		{ $$ = state->d_binary ("+", $1, $3); }
	;

exp	:	exp '-' exp
		{ $$ = state->d_binary ("-", $1, $3); }
	;

exp	:	exp LSH exp
		{ $$ = state->d_binary ("<<", $1, $3); }
	;

exp	:	exp RSH exp
		{ $$ = state->d_binary (">>", $1, $3); }
	;

exp	:	exp EQUAL exp
		{ $$ = state->d_binary ("==", $1, $3); }
	;

exp	:	exp NOTEQUAL exp
		{ $$ = state->d_binary ("!=", $1, $3); }
	;

exp	:	exp LEQ exp
		{ $$ = state->d_binary ("<=", $1, $3); }
	;

exp	:	exp GEQ exp
		{ $$ = state->d_binary (">=", $1, $3); }
	;

exp	:	exp SPACESHIP exp
		{ $$ = state->d_binary ("<=>", $1, $3); }
	;

exp	:	exp '<' exp
		{ $$ = state->d_binary ("<", $1, $3); }
	;

exp	:	exp '&' exp
		{ $$ = state->d_binary ("&", $1, $3); }
	;

exp	:	exp '^' exp
		{ $$ = state->d_binary ("^", $1, $3); }
	;

exp	:	exp '|' exp
		{ $$ = state->d_binary ("|", $1, $3); }
	;

exp	:	exp ANDAND exp
		{ $$ = state->d_binary ("&&", $1, $3); }
	;

exp	:	exp OROR exp
		{ $$ = state->d_binary ("||", $1, $3); }
	;

/* Not 100% sure these are necessary, but they're harmless.  */
exp	:	exp ARROW NAME
		{ $$ = state->d_binary ("->", $1, $3); }
	;

exp	:	exp '.' NAME
		{ $$ = state->d_binary (".", $1, $3); }
	;

exp	:	exp '?' exp ':' exp	%prec '?'
		{ $$ = state->fill_comp (DEMANGLE_COMPONENT_TRINARY, state->make_operator ("?", 3),
				    state->fill_comp (DEMANGLE_COMPONENT_TRINARY_ARG1, $1,
						 state->fill_comp (DEMANGLE_COMPONENT_TRINARY_ARG2, $3, $5)));
		}
	;

exp	:	INT
	;

/* Not generally allowed.  */
exp	:	FLOAT
	;

exp	:	SIZEOF '(' type ')'	%prec UNARY
		{
		  /* Match the whitespacing of cplus_demangle_operators.
		     It would abort on unrecognized string otherwise.  */
		  $$ = state->d_unary ("sizeof ", $3);
		}
	;

/* C++.  */
exp     :       TRUEKEYWORD
		{ struct demangle_component *i;
		  i = state->make_name ("1", 1);
		  $$ = state->fill_comp (DEMANGLE_COMPONENT_LITERAL,
				    state->make_builtin_type ( "bool"),
				    i);
		}
	;

exp     :       FALSEKEYWORD
		{ struct demangle_component *i;
		  i = state->make_name ("0", 1);
		  $$ = state->fill_comp (DEMANGLE_COMPONENT_LITERAL,
				    state->make_builtin_type ("bool"),
				    i);
		}
	;

/* end of C++.  */
