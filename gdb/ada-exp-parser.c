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

#include "ada-exp-parser.h"
#include "ada-exp-parser-gen.h"
#include "ada-lang.h"
#include "ada-lex-gen.h"
#include "block.h"
#include "cli/cli-style.h"
#include "gdbsupport/scoped_restore.h"
#include "parser-defs.h"

using namespace expr;

/* The entry point of the bison/yacc-generated parser, defined in
   ada-exp-parser-gen.c.  Bison produces a declaration for ada_yyparse in
   ada-exp-parser-gen.h, but byacc does not, hence this declaration.  */

int ada_yyparse ();

/* Likewise, byacc does not produce a declaration for ada_yydebug.  */

extern int ada_yydebug;

namespace ada_exp_parser
{

/* See ada-exp-parser.h.  */

struct parser_state *pstate;

/* See ada-exp-parser.h.  */

struct ada_parse_state *ada_parser;

/* See ada-exp-parser.h.  */

void
canonicalizeNumeral (char *s1, const char *s2)
{
  for (; *s2 != '\000'; s2 += 1)
    {
      if (*s2 != '_')
	{
	  *s1 = c_tolower(*s2);
	  s1 += 1;
	}
    }
  s1[0] = '\000';
}

/* See ada-exp-parser.h.  */

int
processInt (struct parser_state *par_state, const char *base0,
	    const char *num0, const char *exp0)
{
  long exp;
  int base;
  /* For the based literal with an "f" prefix, we'll return a
     floating-point number.  This counts the number of "l"s seen,
     to decide the width of the floating-point number to return.  -1
     means no "f".  */
  int floating_point_l_count = -1;

  if (base0 == NULL)
    base = 10;
  else
    {
      char *end_of_base;
      base = strtol (base0, &end_of_base, 10);
      if (base < 2 || base > 16)
	error (_("Invalid base: %d."), base);
      while (*end_of_base == 'l')
	{
	  ++floating_point_l_count;
	  ++end_of_base;
	}
      /* This assertion is ensured by the pattern.  */
      gdb_assert (floating_point_l_count == -1 || *end_of_base == 'f');
      if (*end_of_base == 'f')
	{
	  ++end_of_base;
	  ++floating_point_l_count;
	}
      /* This assertion is ensured by the pattern.  */
      gdb_assert (*end_of_base == '#');
    }

  if (exp0 == NULL)
    exp = 0;
  else
    exp = strtol(exp0, (char **) NULL, 10);

  gdb_mpz result;
  while (c_isxdigit (*num0))
    {
      int dig = fromhex (*num0);
      if (dig >= base)
	error (_("Invalid digit `%c' in based literal"), *num0);
      result *= base;
      result += dig;
      ++num0;
    }

  while (exp > 0)
    {
      result *= base;
      exp -= 1;
    }

  if (floating_point_l_count > -1)
    {
      struct type *fp_type;
      if (floating_point_l_count == 0)
	fp_type = language_lookup_primitive_type (par_state->language (),
						  par_state->gdbarch (),
						  "float");
      else if (floating_point_l_count == 1)
	fp_type = language_lookup_primitive_type (par_state->language (),
						  par_state->gdbarch (),
						  "long_float");
      else
	{
	  /* This assertion is ensured by the pattern.  */
	  gdb_assert (floating_point_l_count == 2);
	  fp_type = language_lookup_primitive_type (par_state->language (),
						    par_state->gdbarch (),
						    "long_long_float");
	}

      ada_yylval.typed_val_float.type = fp_type;
      result.write (gdb::make_array_view (ada_yylval.typed_val_float.val,
					  fp_type->length ()),
		    type_byte_order (fp_type),
		    true);

      return FLOAT;
    }

  const gdb_mpz *value = ada_parser->push_integer (std::move (result));

  int int_bits = gdbarch_int_bit (par_state->gdbarch ());
  int long_bits = gdbarch_long_bit (par_state->gdbarch ());
  int long_long_bits = gdbarch_long_long_bit (par_state->gdbarch ());

  if (fits_in_type (1, *value, int_bits, true))
    ada_yylval.typed_val.type = parse_type (par_state)->builtin_int;
  else if (fits_in_type (1, *value, long_bits, true))
    ada_yylval.typed_val.type = parse_type (par_state)->builtin_long;
  else if (fits_in_type (1, *value, long_bits, false))
    ada_yylval.typed_val.type
      = builtin_type (par_state->gdbarch ())->builtin_unsigned_long;
  else if (fits_in_type (1, *value, long_long_bits, true))
    ada_yylval.typed_val.type = parse_type (par_state)->builtin_long_long;
  else if (fits_in_type (1, *value, long_long_bits, false))
    ada_yylval.typed_val.type
      = builtin_type (par_state->gdbarch ())->builtin_unsigned_long_long;
  else if (fits_in_type (1, *value, 128, true))
    ada_yylval.typed_val.type
      = language_lookup_primitive_type (par_state->language (),
					par_state->gdbarch (),
					"long_long_long_integer");
  else if (fits_in_type (1, *value, 128, false))
    ada_yylval.typed_val.type
      = language_lookup_primitive_type (par_state->language (),
					par_state->gdbarch (),
					"unsigned_long_long_long_integer");
  else
    error (_("Integer literal out of range"));

  ada_yylval.typed_val.val = value;
  return INT;
}

/* See ada-exp-parser.h.  */

int
processReal (struct parser_state *par_state, const char *num0)
{
  ada_yylval.typed_val_float.type = parse_type (par_state)->builtin_long_double;

  bool parsed = parse_float (num0, strlen (num0),
			     ada_yylval.typed_val_float.type,
			     ada_yylval.typed_val_float.val);
  gdb_assert (parsed);
  return FLOAT;
}


/* See ada-exp-parser.h.  */

struct stoken
processId (const char *name0, int len)
{
  char *name = (char *) obstack_alloc (&ada_parser->temp_space, len + 11);
  int i0, i;
  struct stoken result;

  result.ptr = name;
  while (len > 0 && c_isspace (name0[len-1]))
    len -= 1;

  if (name0[0] == '<' || strstr (name0, "___") != NULL)
    {
      strncpy (name, name0, len);
      name[len] = '\000';
      result.length = len;
      return result;
    }

  bool in_quotes = false;
  i = i0 = 0;
  while (i0 < len)
    {
      if (name0[i0] == COMPLETE_CHAR)
	{
	  /* Just ignore.  */
	  ++i0;
	}
      else if (in_quotes)
	name[i++] = name0[i0++];
      else if (c_isalnum (name0[i0]))
	{
	  name[i] = c_tolower (name0[i0]);
	  i += 1; i0 += 1;
	}
      else if (c_isspace (name0[i0]))
	i0 += 1;
      else if (name0[i0] == '\'')
	{
	  /* Copy the starting quote, but not the ending quote.  */
	  if (!in_quotes)
	    name[i++] = name0[i0++];
	  in_quotes = !in_quotes;
	}
      else
	name[i++] = name0[i0++];
    }
  name[i] = '\000';

  result.length = i;
  return result;
}

/* See ada-exp-parser.h.  */

struct stoken
processString (const char *text, int len)
{
  const char *p;
  char *q;
  const char *lim = text + len;
  struct stoken result;

  q = (char *) obstack_alloc (&ada_parser->temp_space, len);
  result.ptr = q;
  p = text;
  while (p < lim)
    {
      if (p[0] == '[' && p[1] == '"' && p+2 < lim)
	 {
	   if (p[2] == '"')  /* "...["""]... */
	     {
	       *q = '"';
	       p += 4;
	     }
	   else
	     {
	       const char *end;
	       ULONGEST chr = strtoulst (p + 2, &end, 16);
	       if (chr > 0xff)
		 error (_("wide strings are not yet supported"));
	       *q = (char) chr;
	       p = end + 1;
	     }
	 }
       else
	 *q = *p;
       q += 1;
       p += 1;
     }
  result.length = q - result.ptr;
  return result;
}

/* See ada-exp-parser.h.  */

int
find_dot_all (const char *str)
{
  int i;

  for (i = 0; str[i] != '\000'; i++)
    if (str[i] == '.')
      {
	int i0 = i;

	do
	  i += 1;
	while (c_isspace (str[i]));

	if (strncasecmp (str + i, "all", 3) == 0
	    && !c_isalnum (str[i + 3]) && str[i + 3] != '_')
	  return i0;
      }
  return -1;
}

/* Returns non-zero iff string SUBSEQ matches a subsequence of STR, ignoring
   case.  */

static int
subseqMatch (const char *subseq, const char *str)
{
  if (subseq[0] == '\0')
    return 1;
  else if (str[0] == '\0')
    return 0;
  else if (c_tolower (subseq[0]) == c_tolower (str[0]))
    return subseqMatch (subseq+1, str+1) || subseqMatch (subseq, str+1);
  else
    return subseqMatch (subseq, str+1);
}


static const struct { const char *name; int code; }
attributes[] = {
  { "address", TICK_ADDRESS },
  { "unchecked_access", TICK_ACCESS },
  { "unrestricted_access", TICK_ACCESS },
  { "access", TICK_ACCESS },
  { "first", TICK_FIRST },
  { "last", TICK_LAST },
  { "length", TICK_LENGTH },
  { "max", TICK_MAX },
  { "min", TICK_MIN },
  { "modulus", TICK_MODULUS },
  { "object_size", TICK_OBJECT_SIZE },
  { "pos", TICK_POS },
  { "range", TICK_RANGE },
  { "size", TICK_SIZE },
  { "tag", TICK_TAG },
  { "val", TICK_VAL },
  { "enum_rep", TICK_ENUM_REP },
  { "enum_val", TICK_ENUM_VAL },
};

/* See ada-exp-parser.h.  */

int
processAttribute (const char *str)
{
  gdb_assert (*str == '\'');
  ++str;
  while (c_isspace (*str))
    ++str;

  int len = strlen (str);
  if (len > 0 && str[len - 1] == COMPLETE_CHAR)
    {
      /* This is enforced by YY_INPUT.  */
      gdb_assert (pstate->parse_completion);
      ada_yylval.sval.ptr = obstack_strndup (&ada_parser->temp_space,
					 str, len - 1);
      ada_yylval.sval.length = len - 1;
      return TICK_COMPLETE;
    }

  for (const auto &item : attributes)
    if (strcasecmp (str, item.name) == 0)
      return item.code;

  std::optional<int> found;
  for (const auto &item : attributes)
    if (subseqMatch (str, item.name))
      {
	if (!found.has_value ())
	  found = item.code;
	else
	  error (_("ambiguous attribute name: `%s'"), str);
      }
  if (!found.has_value ())
    error (_("unrecognized attribute: `%s'"), str);

  return *found;
}

bool
ada_tick_completer::complete (struct expression *exp,
			      completion_tracker &tracker)
{
  completion_list output;
  for (const auto &item : attributes)
    {
      if (strncasecmp (item.name, m_name.c_str (), m_name.length ()) == 0)
	output.emplace_back (xstrdup (item.name));
    }
  tracker.add_completions (std::move (output));
  return true;
}

/* See ada-exp-parser.h.  */

void
rewind_to_char (int ch)
{
  pstate->lexptr -= ada_yyleng;
  while (c_toupper (*pstate->lexptr) != c_toupper (ch))
    pstate->lexptr -= 1;
  ada_yyrestart (NULL);
}

/* See ada-exp-parser.h.  */

operation_up
resolve (operation_up &&op, bool deprocedure_p, struct type *context_type)
{
  operation_up result = std::move (op);
  ada_resolvable *res = dynamic_cast<ada_resolvable *> (result.get ());
  if (res != nullptr)
    return res->replace (std::move (result),
			 pstate->expout.get (),
			 deprocedure_p,
			 pstate->parse_completion,
			 pstate->block_tracker,
			 context_type);
  return result;
}

/* See ada-exp-parser.h.  */

operation_up
ada_pop (bool deprocedure_p, struct type *context_type)
{
  /* Of course it's ok to call parser_state::pop here... */
  return resolve (pstate->pop (), deprocedure_p, context_type);
}

/* See ada-exp-parser.h.  */

void
ada_addrof (struct type *type)
{
  operation_up arg = ada_pop (false);
  operation_up addr = make_operation<unop_addr_operation> (std::move (arg));
  operation_up wrapped
    = make_operation<ada_wrapped_operation> (std::move (addr));
  if (type != nullptr)
    wrapped = make_operation<unop_cast_operation> (std::move (wrapped), type);
  pstate->push (std::move (wrapped));
}

/* See ada-exp-parser.h.  */

operation_up
maybe_overload (enum exp_opcode op, operation_up &lhs, operation_up &rhs)
{
  struct value *args[2];

  int nargs = 1;
  args[0] = lhs->evaluate (nullptr, pstate->expout.get (),
			   EVAL_AVOID_SIDE_EFFECTS);
  if (rhs == nullptr)
    args[1] = nullptr;
  else
    {
      args[1] = rhs->evaluate (nullptr, pstate->expout.get (),
			       EVAL_AVOID_SIDE_EFFECTS);
      ++nargs;
    }

  block_symbol fn = ada_find_operator_symbol (op, pstate->parse_completion,
					      nargs, args);
  if (fn.symbol == nullptr)
    return {};

  if (symbol_read_needs_frame (fn.symbol))
    pstate->block_tracker->update (fn.block, INNERMOST_BLOCK_FOR_SYMBOLS);
  operation_up callee = make_operation<ada_var_value_operation> (fn);

  std::vector<operation_up> argvec;
  argvec.push_back (std::move (lhs));
  if (rhs != nullptr)
    argvec.push_back (std::move (rhs));
  return make_operation<ada_funcall_operation> (std::move (callee),
						std::move (argvec));
}

/* See ada-exp-parser.h.  */

void
ada_funcall (int nargs)
{
  /* We use the ordinary pop here, because we're going to do
     resolution in a separate step, in order to handle array
     indices.  */
  std::vector<operation_up> args = pstate->pop_vector (nargs);
  /* Call parser_state::pop here, because we don't want to
     function-convert the callee slot of a call we're already
     constructing.  */
  operation_up callee = pstate->pop ();

  ada_var_value_operation *vvo
    = dynamic_cast<ada_var_value_operation *> (callee.get ());
  int array_arity = 0;
  struct type *callee_t = nullptr;
  if (vvo == nullptr
      || vvo->get_symbol ()->domain () != UNDEF_DOMAIN)
    {
      struct value *callee_v = callee->evaluate (nullptr,
						 pstate->expout.get (),
						 EVAL_AVOID_SIDE_EFFECTS);
      callee_t = ada_check_typedef (callee_v->type ());
      array_arity = ada_array_arity (callee_t);
    }

  for (int i = 0; i < nargs; ++i)
    {
      struct type *subtype = nullptr;
      if (i < array_arity)
	subtype = ada_index_type (callee_t, i + 1, "array type");
      args[i] = resolve (std::move (args[i]), true, subtype);
    }

  std::unique_ptr<ada_funcall_operation> funcall
    (new ada_funcall_operation (std::move (callee), std::move (args)));
  funcall->resolve (pstate->expout.get (), true, pstate->parse_completion,
		    pstate->block_tracker, nullptr);
  pstate->push (std::move (funcall));
}

/* See ada-exp-parser.h.  */

ada_choices_component *
choice_component ()
{
  ada_component *last = ada_parser->components.back ().get ();
  return gdb::checked_static_cast<ada_choices_component *> (last);
}

/* See ada-exp-parser.h.  */

ada_component_up
pop_component ()
{
  ada_component_up result = std::move (ada_parser->components.back ());
  ada_parser->components.pop_back ();
  return result;
}

/* See ada-exp-parser.h.  */

std::vector<ada_component_up>
pop_components (int n)
{
  std::vector<ada_component_up> result (n);
  for (int i = 1; i <= n; ++i)
    result[n - i] = pop_component ();
  return result;
}

/* Pop the most recent association from the global stack, and return
   it.  */
static ada_association_up
pop_association ()
{
  ada_association_up result = std::move (ada_parser->associations.back ());
  ada_parser->associations.pop_back ();
  return result;
}

/* See ada-exp-parser.h.  */

std::vector<ada_association_up>
pop_associations (int n)
{
  std::vector<ada_association_up> result (n);
  for (int i = 1; i <= n; ++i)
    result[n - i] = pop_association ();
  return result;
}

/* See ada-exp-parser.h.  */

std::unique_ptr<expr_completion_base>
make_tick_completer (struct stoken tok)
{
  return (std::unique_ptr<expr_completion_base>
	  (new ada_tick_completer (std::string (tok.ptr, tok.length))));
}

/* Emit expression to access an instance of SYM, in block BLOCK (if
   non-NULL).  */

static void
write_var_from_sym (struct parser_state *par_state, block_symbol sym)
{
  if (symbol_read_needs_frame (sym.symbol))
    par_state->block_tracker->update (sym.block, INNERMOST_BLOCK_FOR_SYMBOLS);

  par_state->push_new<ada_var_value_operation> (sym);
}

/* See ada-exp-parser.h.  */

void
write_int (struct parser_state *par_state, LONGEST arg, struct type *type)
{
  pstate->push_new<long_const_operation> (type, arg);
  ada_wrap<ada_wrapped_operation> ();
}

/* Emit expression corresponding to the renamed object named
   designated by RENAMED_ENTITY[0 .. RENAMED_ENTITY_LEN-1] in the
   context of ORIG_LEFT_CONTEXT, to which is applied the operations
   encoded by RENAMING_EXPR.  MAX_DEPTH is the maximum number of
   cascaded renamings to allow.  If ORIG_LEFT_CONTEXT is null, it
   defaults to the currently selected block. ORIG_SYMBOL is the
   symbol that originally encoded the renaming.  It is needed only
   because its prefix also qualifies any index variables used to index
   or slice an array.  It should not be necessary once we go to the
   new encoding entirely (FIXME pnh 7/20/2007).  */

static void
write_object_renaming (struct parser_state *par_state,
		       const struct block *orig_left_context,
		       const char *renamed_entity, int renamed_entity_len,
		       const char *renaming_expr, int max_depth)
{
  char *name;
  enum { SIMPLE_INDEX, LOWER_BOUND, UPPER_BOUND } slice_state;

  if (max_depth <= 0)
    error (_("Could not find renamed symbol"));

  if (orig_left_context == NULL)
    orig_left_context = get_selected_block ();

  name = obstack_strndup (&ada_parser->temp_space, renamed_entity,
			  renamed_entity_len);
  block_symbol sym_info = ada_lookup_encoded_symbol (name, orig_left_context,
						     SEARCH_VFT);
  if (sym_info.symbol == NULL)
    error (_("Could not find renamed variable: %ps"),
	   styled_string (variable_name_style.style (),
			  ada_decode (name).c_str ()));
  else if (sym_info.symbol->loc_class () == LOC_TYPEDEF)
    /* We have a renaming of an old-style renaming symbol.  Don't
       trust the block information.  */
    sym_info.block = orig_left_context;

  {
    const char *inner_renamed_entity;
    int inner_renamed_entity_len;
    const char *inner_renaming_expr;

    switch (ada_parse_renaming (sym_info.symbol, &inner_renamed_entity,
				&inner_renamed_entity_len,
				&inner_renaming_expr))
      {
      case ADA_NOT_RENAMING:
	write_var_from_sym (par_state, sym_info);
	break;
      case ADA_OBJECT_RENAMING:
	write_object_renaming (par_state, sym_info.block,
			       inner_renamed_entity, inner_renamed_entity_len,
			       inner_renaming_expr, max_depth - 1);
	break;
      default:
	goto BadEncoding;
      }
  }

  slice_state = SIMPLE_INDEX;
  while (*renaming_expr == 'X')
    {
      renaming_expr += 1;

      switch (*renaming_expr) {
      case 'A':
	renaming_expr += 1;
	ada_wrap<ada_unop_ind_operation> ();
	break;
      case 'L':
	slice_state = LOWER_BOUND;
	[[fallthrough]];
      case 'S':
	renaming_expr += 1;
	if (c_isdigit (*renaming_expr))
	  {
	    char *next;
	    long val = strtol (renaming_expr, &next, 10);
	    if (next == renaming_expr)
	      goto BadEncoding;
	    renaming_expr = next;
	    write_int (par_state, val, parse_type (par_state)->builtin_int);
	  }
	else
	  {
	    const char *end;
	    char *index_name;

	    end = strchr (renaming_expr, 'X');
	    if (end == NULL)
	      end = renaming_expr + strlen (renaming_expr);

	    index_name = obstack_strndup (&ada_parser->temp_space,
					  renaming_expr,
					  end - renaming_expr);
	    renaming_expr = end;

	    block_symbol index_sym_info
	      = ada_lookup_encoded_symbol (index_name, orig_left_context,
					   SEARCH_VFT);
	    if (index_sym_info.symbol == NULL)
	      error (_("Could not find %s"), index_name);
	    else if (index_sym_info.symbol->loc_class () == LOC_TYPEDEF)
	      /* Index is an old-style renaming symbol.  */
	      index_sym_info.block = orig_left_context;
	    write_var_from_sym (par_state, index_sym_info);
	  }
	if (slice_state == SIMPLE_INDEX)
	  ada_funcall (1);
	else if (slice_state == LOWER_BOUND)
	  slice_state = UPPER_BOUND;
	else if (slice_state == UPPER_BOUND)
	  {
	    ada_wrap3<ada_ternop_slice_operation> ();
	    slice_state = SIMPLE_INDEX;
	  }
	break;

      case 'R':
	{
	  const char *end;

	  renaming_expr += 1;

	  if (slice_state != SIMPLE_INDEX)
	    goto BadEncoding;
	  end = strchr (renaming_expr, 'X');
	  if (end == NULL)
	    end = renaming_expr + strlen (renaming_expr);

	  operation_up arg = ada_pop ();
	  pstate->push_new<ada_structop_operation>
	    (std::move (arg), std::string (renaming_expr,
					   end - renaming_expr));
	  renaming_expr = end;
	  break;
	}

      default:
	goto BadEncoding;
      }
    }
  if (slice_state == SIMPLE_INDEX)
    return;

 BadEncoding:
  error (_("Internal error in encoding of renaming declaration"));
}

/* See ada-exp-parser.h.  */

const struct block*
block_lookup (const struct block *context, const char *raw_name)
{
  const char *name;
  struct symtab *symtab;
  const struct block *result = NULL;

  std::string name_storage;
  if (raw_name[0] == '\'')
    {
      raw_name += 1;
      name = raw_name;
    }
  else
    {
      name_storage = ada_encode (raw_name);
      name = name_storage.c_str ();
    }

  std::vector<struct block_symbol> syms
    = ada_lookup_symbol_list (name, context, SEARCH_FUNCTION_DOMAIN);

  if (context == NULL
      && (syms.empty () || syms[0].symbol->loc_class () != LOC_BLOCK))
    symtab = lookup_symtab (current_program_space, name);
  else
    symtab = NULL;

  if (symtab != NULL)
    result = symtab->compunit ().blockvector ()->static_block ();
  else if (syms.empty () || syms[0].symbol->loc_class () != LOC_BLOCK)
    {
      if (context == NULL)
	error (_("No file or function \"%s\"."), raw_name);
      else
	error (_("No function \"%s\" in specified context."), raw_name);
    }
  else
    {
      if (syms.size () > 1)
	warning (_("Function name \"%s\" ambiguous here"), raw_name);
      result = syms[0].symbol->value_block ();
    }

  return result;
}

static struct symbol*
select_possible_type_sym (const std::vector<struct block_symbol> &syms)
{
  int i;
  int preferred_index;
  struct type *preferred_type;

  preferred_index = -1; preferred_type = NULL;
  for (i = 0; i < syms.size (); i += 1)
    switch (syms[i].symbol->loc_class ())
      {
      case LOC_TYPEDEF:
	if (ada_prefer_type (syms[i].symbol->type (), preferred_type))
	  {
	    preferred_index = i;
	    preferred_type = syms[i].symbol->type ();
	  }
	break;
      case LOC_REGISTER:
      case LOC_ARG:
      case LOC_REF_ARG:
      case LOC_REGPARM_ADDR:
      case LOC_LOCAL:
      case LOC_COMPUTED:
	return NULL;
      default:
	break;
      }
  if (preferred_type == NULL)
    return NULL;
  return syms[preferred_index].symbol;
}

static struct type*
find_primitive_type (struct parser_state *par_state, const char *name)
{
  struct type *type;
  type = language_lookup_primitive_type (par_state->language (),
					 par_state->gdbarch (),
					 name);
  if (type == NULL && streq ("system__address", name))
    type = type_system_address (par_state);

  if (type != NULL)
    {
      /* Check to see if we have a regular definition of this
	 type that just didn't happen to have been read yet.  */
      struct symbol *sym;
      char *expanded_name =
	(char *) alloca (strlen (name) + sizeof ("standard__"));
      strcpy (expanded_name, "standard__");
      strcat (expanded_name, name);
      sym = ada_lookup_symbol (expanded_name, NULL, SEARCH_TYPE_DOMAIN).symbol;
      if (sym != NULL && sym->loc_class () == LOC_TYPEDEF)
	type = sym->type ();
    }

  return type;
}

static int
chop_selector (const char *name, int end)
{
  int i;
  for (i = end - 1; i > 0; i -= 1)
    if (name[i] == '.' || (name[i] == '_' && name[i+1] == '_'))
      return i;
  return -1;
}

/* If NAME is a string beginning with a separator (either '__', or
   '.'), chop this separator and return the result; else, return
   NAME.  */

static const char *
chop_separator (const char *name)
{
  if (*name == '.')
   return name + 1;

  if (name[0] == '_' && name[1] == '_')
    return name + 2;

  return name;
}

/* Given that SELS is a string of the form (<sep><identifier>)*, where
   <sep> is '__' or '.', write the indicated sequence of
   STRUCTOP_STRUCT expression operators.  Returns a pointer to the
   last operation that was pushed.  */
static ada_structop_operation *
write_selectors (struct parser_state *par_state, const char *sels)
{
  ada_structop_operation *result = nullptr;
  while (*sels != '\0')
    {
      const char *p = chop_separator (sels);
      sels = p;
      while (*sels != '\0' && *sels != '.'
	     && (sels[0] != '_' || sels[1] != '_'))
	sels += 1;
      operation_up arg = ada_pop ();
      result = new ada_structop_operation (std::move (arg),
					   std::string (p, sels - p));
      pstate->push (operation_up (result));
    }
  return result;
}

/* Write a variable access (OP_VAR_VALUE) to ambiguous encoded name
   NAME[0..LEN-1], in block context BLOCK, to be resolved later.  Writes
   a temporary symbol that is valid until the next call to ada_parse.
   */
static void
write_ambiguous_var (struct parser_state *par_state,
		     const struct block *block, const char *name, int len)
{
  struct symbol *sym = new (&ada_parser->temp_space) symbol ();

  sym->set_domain (UNDEF_DOMAIN);
  sym->set_linkage_name (obstack_strndup (&ada_parser->temp_space, name, len));
  sym->set_language (language_ada, nullptr);

  block_symbol bsym { sym, block };
  par_state->push_new<ada_var_value_operation> (bsym);
}

/* A convenient wrapper around ada_get_field_index that takes
   a non NUL-terminated FIELD_NAME0 and a FIELD_NAME_LEN instead
   of a NUL-terminated field name.  */

static int
ada_nget_field_index (const struct type *type, const char *field_name0,
		      int field_name_len, int maybe_missing)
{
  char *field_name = (char *) alloca ((field_name_len + 1) * sizeof (char));

  strncpy (field_name, field_name0, field_name_len);
  field_name[field_name_len] = '\0';
  return ada_get_field_index (type, field_name, maybe_missing);
}

/* If encoded_field_name is the name of a field inside symbol SYM,
   then return the type of that field.  Otherwise, return NULL.

   This function is actually recursive, so if ENCODED_FIELD_NAME
   doesn't match one of the fields of our symbol, then try to see
   if ENCODED_FIELD_NAME could not be a succession of field names
   (in other words, the user entered an expression of the form
   TYPE_NAME.FIELD1.FIELD2.FIELD3), in which case we evaluate
   each field name sequentially to obtain the desired field type.
   In case of failure, we return NULL.  */

static struct type *
get_symbol_field_type (struct symbol *sym, const char *encoded_field_name)
{
  const char *field_name = encoded_field_name;
  const char *subfield_name;
  struct type *type = sym->type ();
  int fieldno;

  if (type == NULL || field_name == NULL)
    return NULL;
  type = check_typedef (type);

  while (field_name[0] != '\0')
    {
      field_name = chop_separator (field_name);

      fieldno = ada_get_field_index (type, field_name, 1);
      if (fieldno >= 0)
	return type->field (fieldno).type ();

      subfield_name = field_name;
      while (*subfield_name != '\0' && *subfield_name != '.'
	     && (subfield_name[0] != '_' || subfield_name[1] != '_'))
	subfield_name += 1;

      if (subfield_name[0] == '\0')
	return NULL;

      fieldno = ada_nget_field_index (type, field_name,
				      subfield_name - field_name, 1);
      if (fieldno < 0)
	return NULL;

      type = type->field (fieldno).type ();
      field_name = subfield_name;
    }

  return NULL;
}

/* See ada-exp-parser.h.  */

struct type*
write_var_or_type (struct parser_state *par_state,
		   const struct block *block, struct stoken name0)
{
  int depth;
  char *encoded_name;
  int name_len;

  std::string name_storage = ada_encode (name0.ptr);

  if (block == nullptr)
    {
      auto iter = ada_parser->iterated_associations.find (name_storage);
      if (iter != ada_parser->iterated_associations.end ())
	{
	  auto op = std::make_unique<ada_index_var_operation> ();
	  iter->second.push_back (op.get ());
	  par_state->push (std::move (op));
	  return nullptr;
	}

      block = par_state->expression_context_block;
    }

  name_len = name_storage.size ();
  encoded_name = obstack_strndup (&ada_parser->temp_space,
				  name_storage.c_str (),
				  name_len);
  for (depth = 0; depth < MAX_RENAMING_CHAIN_LENGTH; depth += 1)
    {
      int tail_index;

      tail_index = name_len;
      while (tail_index > 0)
	{
	  struct symbol *type_sym;
	  struct symbol *renaming_sym;
	  const char* renaming;
	  int renaming_len;
	  const char* renaming_expr;
	  int terminator = encoded_name[tail_index];

	  encoded_name[tail_index] = '\0';
	  /* In order to avoid double-encoding, we want to only pass
	     the decoded form to lookup functions.  */
	  std::string decoded_name = ada_decode (encoded_name);
	  encoded_name[tail_index] = terminator;

	  std::vector<struct block_symbol> syms
	    = ada_lookup_symbol_list (decoded_name.c_str (), block,
				      SEARCH_VFT);

	  type_sym = select_possible_type_sym (syms);

	  if (type_sym != NULL)
	    renaming_sym = type_sym;
	  else if (syms.size () == 1)
	    renaming_sym = syms[0].symbol;
	  else
	    renaming_sym = NULL;

	  switch (ada_parse_renaming (renaming_sym, &renaming,
				      &renaming_len, &renaming_expr))
	    {
	    case ADA_NOT_RENAMING:
	      break;
	    case ADA_PACKAGE_RENAMING:
	    case ADA_EXCEPTION_RENAMING:
	    case ADA_SUBPROGRAM_RENAMING:
	      {
		int alloc_len = renaming_len + name_len - tail_index + 1;
		char *new_name
		  = (char *) obstack_alloc (&ada_parser->temp_space,
					    alloc_len);
		strncpy (new_name, renaming, renaming_len);
		strcpy (new_name + renaming_len, encoded_name + tail_index);
		encoded_name = new_name;
		name_len = renaming_len + name_len - tail_index;
		goto TryAfterRenaming;
	      }
	    case ADA_OBJECT_RENAMING:
	      write_object_renaming (par_state, block, renaming, renaming_len,
				     renaming_expr, MAX_RENAMING_CHAIN_LENGTH);
	      write_selectors (par_state, encoded_name + tail_index);
	      return NULL;
	    default:
	      internal_error (_("impossible value from ada_parse_renaming"));
	    }

	  if (type_sym != NULL)
	    {
	      struct type *field_type;

	      if (tail_index == name_len)
		return type_sym->type ();

	      /* We have some extraneous characters after the type name.
		 If this is an expression "TYPE_NAME.FIELD0.[...].FIELDN",
		 then try to get the type of FIELDN.  */
	      field_type
		= get_symbol_field_type (type_sym, encoded_name + tail_index);
	      if (field_type != NULL)
		return field_type;
	      else
		error (_("Invalid attempt to select from type: \"%s\"."),
		       name0.ptr);
	    }
	  else if (tail_index == name_len && syms.empty ())
	    {
	      struct type *type = find_primitive_type (par_state,
						       encoded_name);

	      if (type != NULL)
		return type;
	    }

	  if (syms.size () == 1)
	    {
	      write_var_from_sym (par_state, syms[0]);
	      write_selectors (par_state, encoded_name + tail_index);
	      return NULL;
	    }
	  else if (syms.empty ())
	    {
	      struct objfile *objfile = nullptr;
	      if (block != nullptr)
		objfile = block->objfile ();

	      bound_minimal_symbol msym
		= ada_lookup_simple_minsym (decoded_name.c_str (), objfile);
	      if (msym.minsym != NULL)
		{
		  par_state->push_new<ada_var_msym_value_operation> (msym);
		  /* Maybe cause error here rather than later? FIXME? */
		  write_selectors (par_state, encoded_name + tail_index);
		  return NULL;
		}

	      if (tail_index == name_len
		  && strncmp (encoded_name, "standard__",
			      sizeof ("standard__") - 1) == 0)
		error (_("No definition of \"%s\" found."), name0.ptr);

	      tail_index = chop_selector (encoded_name, tail_index);
	    }
	  else
	    {
	      write_ambiguous_var (par_state, block, encoded_name,
				   tail_index);
	      write_selectors (par_state, encoded_name + tail_index);
	      return NULL;
	    }
	}

      if (!current_program_space->has_full_symbols ()
	  && !current_program_space->has_partial_symbols ()
	  && block == NULL)
	error (_("No symbol table is loaded.  Use the \"%ps\" command."),
	       styled_string (command_style.style (), "file"));
      if (block == par_state->expression_context_block)
	error (_("No definition of \"%s\" in current context."), name0.ptr);
      else
	error (_("No definition of \"%s\" in specified context."), name0.ptr);

    TryAfterRenaming: ;
    }

  error (_("Could not find renamed symbol \"%s\""), name0.ptr);

}

/* Because ada_completer_word_break_characters does not contain '.' --
   and it cannot easily be added, this breaks other completions -- we
   have to recreate the completion word-splitting here, so that we can
   provide a prefix that is then used when completing field names.
   Without this, an attempt like "complete print abc.d" will give a
   result like "print def" rather than "print abc.def".  */

std::string
ada_parse_state::find_completion_bounds ()
{
  const char *end = pstate->lexptr;
  /* First the end of the prefix.  Here we stop at the token start or
     at '.' or space.  */
  for (; end > m_original_expr && end[-1] != '.' && !c_isspace (end[-1]); --end)
    {
      /* Nothing.  */
    }
  /* Now find the start of the prefix.  */
  const char *ptr = end;
  /* Here we allow '.'.  */
  for (;
       ptr > m_original_expr && (ptr[-1] == '.'
				 || ptr[-1] == '_'
				 || (ptr[-1] >= 'a' && ptr[-1] <= 'z')
				 || (ptr[-1] >= 'A' && ptr[-1] <= 'Z')
				 || (ptr[-1] & 0xff) >= 0x80);
       --ptr)
    {
      /* Nothing.  */
    }
  /* ... except, skip leading spaces.  */
  ptr = skip_spaces (ptr);

  return std::string (ptr, end);
}

/* See ada-exp-parser.h.  */

struct type *
write_var_or_type_completion (struct parser_state *par_state,
			      const struct block *block, struct stoken name0)
{
  int tail_index = chop_selector (name0.ptr, name0.length);
  /* If there's no separator, just defer to ordinary symbol
     completion.  */
  if (tail_index == -1)
    return write_var_or_type (par_state, block, name0);

  std::string copy (name0.ptr, tail_index);
  struct type *type = write_var_or_type (par_state, block,
					 { copy.c_str (),
					   (int) copy.length () });
  /* For completion purposes, it's enough that we return a type
     here.  */
  if (type != nullptr)
    return type;

  ada_structop_operation *op = write_selectors (par_state,
						name0.ptr + tail_index);
  op->set_prefix (ada_parser->find_completion_bounds ());
  par_state->mark_struct_expression (op);
  return nullptr;
}

/* See ada-exp-parser.h.  */

void
write_name_assoc (struct parser_state *par_state, struct stoken name)
{
  if (strchr (name.ptr, '.') == NULL)
    {
      std::vector<struct block_symbol> syms
	= ada_lookup_symbol_list (name.ptr,
				  par_state->expression_context_block,
				  SEARCH_VFT);

      if (syms.size () != 1 || syms[0].symbol->loc_class () == LOC_TYPEDEF)
	pstate->push_new<ada_string_operation> (copy_name (name));
      else
	write_var_from_sym (par_state, syms[0]);
    }
  else
    if (write_var_or_type (par_state, NULL, name) != NULL)
      error (_("Invalid use of type."));

  push_association<ada_name_association> (ada_pop ());
}

/* See ada-exp-parser.h.  */

struct type *
type_for_char (struct parser_state *par_state, ULONGEST value)
{
  if (value <= 0xff)
    return language_string_char_type (par_state->language (),
				      par_state->gdbarch ());
  else if (value <= 0xffff)
    return language_lookup_primitive_type (par_state->language (),
					   par_state->gdbarch (),
					   "wide_character");
  return language_lookup_primitive_type (par_state->language (),
					 par_state->gdbarch (),
					 "wide_wide_character");
}

/* See ada-exp-parser.h.  */

struct type *
type_system_address (struct parser_state *par_state)
{
  struct type *type
    = language_lookup_primitive_type (par_state->language (),
				      par_state->gdbarch (),
				      "system__address");
  return  type != NULL ? type : parse_type (par_state)->builtin_data_ptr;
}

/* See ada-exp-parser.h.  */

void
ada_yyerror (const char *msg)
{
  ada_exp_parser::pstate->parse_error (msg);
}

} /* namespace ada_exp_parser */

/* See ada-exp-parser.h.  */

int
ada_parse (struct parser_state *par_state)
{
  using namespace ada_exp_parser;

  /* Setting up the parser state.  */
  scoped_restore pstate_restore = make_scoped_restore (&pstate, par_state);
  gdb_assert (par_state != NULL);

  ada_parse_state parser (par_state->lexptr);
  scoped_restore parser_restore = make_scoped_restore (&ada_parser, &parser);

  scoped_restore restore_yydebug = make_scoped_restore (&ada_yydebug,
							par_state->debug);

  lexer_init (ada_yyin);		/* (Re-)initialize lexer.  */

  int result = ada_yyparse ();
  if (!result)
    {
      struct type *context_type = nullptr;
      if (par_state->void_context_p)
	context_type = parse_type (par_state)->builtin_void;
      pstate->set_operation (ada_pop (true, context_type));
    }
  return result;
}
