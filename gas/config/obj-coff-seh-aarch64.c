/* SEH .pdata/.xdata COFF object file format on AArch64
   Copyright (C) 2026 Free Software Foundation, Inc.

   This file is part of GAS.

   GAS is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   GAS is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GAS; see the file COPYING.  If not, write to the Free
   Software Foundation, 51 Franklin Street - Fifth Floor, Boston, MA
   02110-1301, USA.  */

#include "obj-coff-seh-aarch64.h"

static struct seh_aarch64_context *seh_ctx_root = NULL;
static bool in_seh_proc = false;

struct aarch64_unwind_info {
  const char *directive;
  unsigned char size;
  unsigned char code_bits;
  unsigned char code;
  unsigned char offset_bits;
  unsigned char offset_shift;
  unsigned char offset_addend;
  unsigned char reg_bits;
  unsigned char reg_shift;
  unsigned char reg_addend;
  char reg_type;
  bool reg_pair;
};

/* Unwind codes for AArch64 are described based on
   "Microsoft ARM64 exception handling, unwind codes documentation"
   and calculated in seh_aarch64_add_unwind_element function.
   aarch64_unwind_code_data is indexed by the seh_aarch64_unwind_types enum.  */

static const struct aarch64_unwind_info
aarch64_unwind_code_data[] = {
  {
    .size = 1,
    .code_bits = 3, .code = 0x0,
    .offset_bits = 5, .offset_shift = 4
  },
  {
    .size = 2,
    .code_bits = 5, .code = 0x18,
    .offset_bits = 11, .offset_shift = 4
  },
  {
    .size = 4,
    .code_bits = 8, .code = 0xe0,
    .offset_bits = 24, .offset_shift = 4
  },
  {
    .directive = ".seh_save_reg",
    .size = 2,
    .code_bits = 6, .code = 0x34,
    .offset_bits = 6, .offset_shift = 3,
    .reg_bits = 4, .reg_addend = 19, .reg_type = 'x'
  },
  {
    .directive = ".seh_save_reg_x",
    .size = 2,
    .code_bits = 7, .code = 0x6a,
    .offset_bits = 5, .offset_shift = 3, .offset_addend = 1,
    .reg_bits = 4, .reg_addend = 19, .reg_type = 'x'
  },
  {
    .directive = ".seh_save_regp",
    .size = 2,
    .code_bits = 6, .code = 0x32,
    .offset_bits = 6, .offset_shift = 3,
    .reg_bits = 4, .reg_addend = 19, .reg_type = 'x', .reg_pair = true
  },
  {
    .directive = ".seh_save_regp_x",
    .size = 2,
    .code_bits = 6, .code = 0x33,
    .offset_bits = 6, .offset_shift = 3, .offset_addend = 1,
    .reg_bits = 4, .reg_addend = 19, .reg_type = 'x', .reg_pair = true
  },
  {
    .directive = ".seh_save_fregp",
    .size = 2,
    .code_bits = 7, .code = 0x6c,
    .offset_bits = 6, .offset_shift = 3,
    .reg_bits = 3, .reg_addend = 8, .reg_type = 'd'
  },
  {
    .directive = ".seh_save_fregp_x",
    .size = 2,
    .code_bits = 7, .code = 0x6d,
    .offset_bits = 6, .offset_shift = 3, .offset_addend = 1,
    .reg_bits = 3, .reg_addend = 8, .reg_type = 'd'
  },
  {
    .directive = ".seh_save_freg",
    .size = 2,
    .code_bits = 7, .code = 0x6e,
    .offset_bits = 6, .offset_shift = 3,
    .reg_bits = 3, .reg_addend = 8, .reg_type = 'd'
  },
  {
    .directive = ".seh_save_freg_x",
    .size = 2,
    .code_bits = 8, .code = 0xde,
    .offset_bits = 5, .offset_shift = 3, .offset_addend = 1,
    .reg_bits = 3, .reg_addend = 8, .reg_type = 'd'
  },
  {
    .directive = ".seh_save_lrpair",
    .size = 2,
    .code_bits = 7, .code = 0x6b,
    .offset_bits = 6, .offset_shift = 3,
    .reg_bits = 3, .reg_shift = 1, .reg_addend = 19, .reg_type = 'x'
  },
  {
    .directive = ".seh_save_fplr",
    .size = 1,
    .code_bits = 2, .code = 0x1,
    .offset_bits = 6, .offset_shift = 3
  },
  {
    .directive = ".seh_save_fplr_x",
    .size = 1,
    .code_bits = 2, .code = 0x2,
    .offset_bits = 6, .offset_shift = 3, .offset_addend = 1
  },
  {
    .directive = ".seh_save_r19r20_x",
    .size = 1,
    .code_bits = 3, .code = 0x1,
    .offset_bits = 5, .offset_shift = 3
  },
  {
    .directive = ".seh_add_fp",
    .size = 2,
    .code_bits = 8, .code = 0xe2,
    .offset_bits = 8, .offset_shift = 3
  },
  {
    .directive = ".seh_set_fp",
    .size = 1,
    .code_bits = 8, .code = 0xe1
  },
  {
    .directive = ".seh_save_next",
    .size = 1,
    .code_bits = 8, .code = 0xe6
  },
  {
    .directive = ".seh_nop",
    .size = 1,
    .code_bits = 8, .code = 0xe3
  },
  {
    .directive = ".seh_pac_sign_lr",
    .size = 1,
    .code_bits = 8, .code = 0xfc
  },
  {
    .size = 1,
    .code_bits = 8, .code = 0xe4
  },
};

/* Set for current context the default handler.  */
static void
obj_coff_seh_handler (const int what ATTRIBUTE_UNUSED)
{
  char *symbol_name;
  char name_end;

  if (!verify_context (".seh_handler"))
    return;

  if (*input_line_pointer == 0 || *input_line_pointer == '\n')
    as_bad (_(".seh_handler requires a handler"));

  SKIP_WHITESPACE ();

  if (*input_line_pointer == '@')
    {
      name_end = get_symbol_name (&symbol_name);

      seh_ctx_cur->handler.X_op = O_constant;
      seh_ctx_cur->handler.X_add_number = 0;

      if (strcasecmp (symbol_name, "@1") == 0)
	seh_ctx_cur->handler.X_add_number = 1;
      else if (strcasecmp (symbol_name, "@0")
	       && strcasecmp (symbol_name, "@null"))
	as_bad (_("unknown constant value '%s' for handler"), symbol_name);

      (void) restore_line_pointer (name_end);
    }
  else
    expression (&seh_ctx_cur->handler);

  seh_ctx_cur->handler_data.X_op = O_constant;
  seh_ctx_cur->handler_data.X_add_number = 0;
  seh_ctx_cur->has_exception_data = true;

  while (skip_whitespace_and_comma (0))
    {
      name_end = get_symbol_name (&symbol_name);
      (void) restore_line_pointer (name_end);
    }
}

/* Switch to subsection for handler data for exception region.  */
static void
obj_coff_seh_handlerdata (const int what ATTRIBUTE_UNUSED)
{
  demand_empty_rest_of_line ();

  switch_xdata (seh_ctx_cur->subsection + 1, seh_ctx_cur->code_seg);
  seh_ctx_cur->handler_data_xdata_addr = symbol_temp_new_now ();
}

/* Switch back to the code section.  */
static void
obj_coff_seh_code (int ignored ATTRIBUTE_UNUSED)
{
  subseg_set (seh_ctx_cur->code_seg, 0);
}

/* Obtain available unwind element.  */
static void
seh_aarch64_add_unwind_element (const seh_aarch64_unwind_types unwind_type,
				unsigned offset, unsigned reg)
{
  const struct aarch64_unwind_info *info
    = aarch64_unwind_code_data + unwind_type;
  unsigned value_offset_bits = 0;

  if ((seh_ctx_cur->unwind_codes_byte_count
      + info->size) > AARCH64_MAX_UNWIND_CODES_SIZE)
    as_bad (_("no unwind element available."));

  unsigned value = 0;

  if (info->offset_bits)
    {
      const unsigned offset_multiplier = 1u << info->offset_shift;
      if (offset & (offset_multiplier - 1))
	as_bad (_("offset should be a multiple of %u"), offset_multiplier);
      offset = (offset >> info->offset_shift) - info->offset_addend;
      if (offset >= (1u << info->offset_bits))
	as_bad (_("offset overflows expected range"));
      value |= offset << value_offset_bits;
      value_offset_bits += info->offset_bits;
    }

  if (info->reg_bits)
    {
      const unsigned reg_multiplier = 1u << info->reg_shift;
      reg -= info->reg_addend;
      if (reg & (reg_multiplier - 1))
	as_bad (_("unexpected register number"));
      reg >>= info->reg_shift;
      if (reg >= (1u << info->reg_bits))
	as_bad (_("unexpected register number"));
      value |= reg << value_offset_bits;
      value_offset_bits += info->reg_bits;
    }

  const unsigned code = info->code;
  gas_assert (code < (1u << info->code_bits));
  value |= code << value_offset_bits;

  seh_aarch64_unwind_code *element;
  element = seh_ctx_cur->unwind_codes + seh_ctx_cur->unwind_codes_count++;
  element->type = unwind_type;
  element->value = value;

  seh_ctx_cur->unwind_codes_byte_count += info->size;
}

/* Mark begin of new context.  */
static void
obj_coff_seh_proc (const int what ATTRIBUTE_UNUSED)
{
  char *symbol_name;
  char name_end;

  if (in_seh_proc)
    as_bad (_("previous SEH entry not closed (missing .seh_endproc)"));

  if (*input_line_pointer == 0 || *input_line_pointer == '\n')
    as_bad (_(".seh_proc requires function label name"));


  if (!seh_ctx_root)
    {
      seh_ctx_root = XCNEW (seh_context);
      seh_ctx_cur = seh_ctx_root;
    }
  else
    {
      seh_ctx_cur->next = XCNEW (seh_context);
      seh_ctx_cur = seh_ctx_cur->next;
    }

  seh_ctx_cur->next = NULL;
  seh_ctx_cur->code_seg = now_seg;

  /* The current implementation always use a pair of .pdata and .xdata
     records.  */
  const bool use_xdata = true;

  if (use_xdata)
    {
      x_segcur = seh_hash_find_or_make (seh_ctx_cur->code_seg, ".xdata");
      seh_ctx_cur->subsection = x_segcur->subseg;
      x_segcur->subseg += 2;

      /* Initialize an empty .xdata record.  */
      seh_ctx_cur->unwind_codes_count = 0;
      seh_ctx_cur->unwind_codes_byte_count = 0;
      seh_ctx_cur->epilogue_scopes_count = 0;
      seh_ctx_cur->epilogue_scopes_capacity = 0;
      seh_ctx_cur->epilogue_scopes = NULL;
      seh_ctx_cur->has_exception_data = false;
    }

  SKIP_WHITESPACE ();

  name_end = get_symbol_name (&symbol_name);
  seh_ctx_cur->func_name = xstrdup (symbol_name);
  (void) restore_line_pointer (name_end);

  demand_empty_rest_of_line ();

  seh_ctx_cur->start_addr = symbol_temp_new_now ();
  in_seh_proc = true;
}

/* Mark end of prologue for current context.  */
static void
obj_coff_seh_endprologue (const int what ATTRIBUTE_UNUSED)
{
  if (!verify_context (".seh_endprologue")
      || !seh_validate_seg (".seh_endprologue"))
    return;
  demand_empty_rest_of_line ();

  if (seh_ctx_cur->endprologue_addr != NULL)
    as_warn (_("duplicate .seh_endprologue in .seh_proc block"));
  else
    seh_ctx_cur->endprologue_addr = symbol_temp_new_now ();

  /* Unwind codes need to be reversed.  */
  for (unsigned i = 0, n = seh_ctx_cur->unwind_codes_count; i < n / 2; ++i)
    {
      seh_aarch64_unwind_code *unwind_codes = seh_ctx_cur->unwind_codes;
      const seh_aarch64_unwind_code temp = unwind_codes[i];
      unwind_codes[i] = unwind_codes[n-i-1];
      unwind_codes[n-i-1] = temp;
    }

   seh_aarch64_add_unwind_element (unwind_end, 0, 0);
}

/* Mark end of current context.  */
static void
obj_coff_seh_endproc (const int what ATTRIBUTE_UNUSED)
{
  demand_empty_rest_of_line ();
  if (!in_seh_proc)
    {
      as_bad (_(".seh_endproc used without .seh_proc"));
      return;
    }

  seh_validate_seg (".seh_endproc");

  seh_ctx_cur->end_addr = symbol_temp_new_now ();
  in_seh_proc = false;
}

static void
obj_coff_seh_startepilogue (const int what ATTRIBUTE_UNUSED)
{
  if (!verify_context (".seh_startepilogue")
      || !seh_validate_seg (".seh_startepilogue"))
    return;
  demand_empty_rest_of_line ();

  if (seh_ctx_cur->epilogue_scopes_count >= AARCH64_MAX_EPILOGUE_SCOPES)
    as_bad (_("no epilogue scopes available."));

  symbolS *epilogue_start_addr = symbol_temp_new_now ();
  expressionS exp;
  exp.X_op = O_subtract;
  exp.X_add_symbol = epilogue_start_addr;
  exp.X_op_symbol = seh_ctx_cur->start_addr;
  exp.X_add_number = 0;

  if (!resolve_expression (&exp) || exp.X_op != O_constant
      || exp.X_add_number < 0)
    as_bad (_(".seh_startepilogue offset expression for %s "
	    "does not evaluate to a non-negative constant"),
	    S_GET_NAME (epilogue_start_addr));

  if (seh_ctx_cur->epilogue_scopes_count
      >= seh_ctx_cur->epilogue_scopes_capacity)
    {
      const unsigned initial_capacity = 32;
      if (seh_ctx_cur->epilogue_scopes_capacity)
	seh_ctx_cur->epilogue_scopes_capacity *= 2;
      else
	seh_ctx_cur->epilogue_scopes_capacity = initial_capacity;

      seh_ctx_cur->epilogue_scopes
	= XRESIZEVEC (seh_aarch64_epilogue_scope, seh_ctx_cur->epilogue_scopes,
		     seh_ctx_cur->epilogue_scopes_capacity);
    }

  seh_aarch64_epilogue_scope *epilogue_scope = seh_ctx_cur->epilogue_scopes
    + seh_ctx_cur->epilogue_scopes_count;
  epilogue_scope->epilogue_start_offset = exp.X_add_number / 4;
  epilogue_scope->reserved = 0;
  epilogue_scope->epilogue_start_index = 0;
  seh_ctx_cur->epilogue_scopes_count++;
}

static void
obj_coff_seh_endepilogue (const int what ATTRIBUTE_UNUSED)
{
  if (!verify_context (".seh_endepilogue")
      || !seh_validate_seg (".seh_endepilogue"))
    return;

  demand_empty_rest_of_line ();

  expressionS exp;
  symbolS *epilogue_end_addr = symbol_temp_new_now ();
  exp.X_op = O_subtract;
  exp.X_add_symbol = epilogue_end_addr;
  exp.X_op_symbol = seh_ctx_cur->start_addr;
  exp.X_add_number = 0;

  if (!resolve_expression (&exp) || exp.X_op != O_constant
      || exp.X_add_number < 0)
    as_bad (_(".seh_endepilogue offset expression for %s "
	    "does not evaluate to a non-negative constant"),
	    S_GET_NAME (epilogue_end_addr));

   seh_aarch64_epilogue_scope *epilogue_scope = seh_ctx_cur->epilogue_scopes
     + seh_ctx_cur->epilogue_scopes_count - 1;

   epilogue_scope->epilogue_end_offset = exp.X_add_number;

  /* End code.  */
  seh_aarch64_add_unwind_element (unwind_end, 0, 0);
}

/* End-of-file hook.  */
static void
free_seh_ctx (struct seh_aarch64_context *seh_ctx)
{
  free (seh_ctx->func_name);
  const seh_aarch64_func_fragment *fragment = seh_ctx->func_fragment.next;
  while (fragment)
    {
      const seh_aarch64_func_fragment *next = fragment->next;
      XDELETE (fragment);
      fragment = next;
    }
  XDELETEVEC (seh_ctx->epilogue_scopes);
  free (seh_ctx);
}

static void
obj_coff_seh_save_reg (const int type)
{
  gas_assert (type >= 0 && type <= unwind_last_type);

  const struct aarch64_unwind_info *info
    = aarch64_unwind_code_data + type;

  SKIP_WHITESPACE ();

  char *symbol_name = NULL;
  unsigned reg = -1;

  if (info->reg_bits)
    {
      char name_end = get_symbol_name (&symbol_name);
      if (info->reg_type != *symbol_name)
	as_bad ("unexpected register name");

      reg = atoi (symbol_name + 1);
      (void) restore_line_pointer (name_end);

      if (!skip_whitespace_and_comma (1))
	return;

      /* Check that referenced registers are not higher than x30.  */
      if (info->reg_type == 'x' && (reg + (info->reg_pair ? 1 : 0)) > 30)
	as_bad (_("unexpected register number"));
    }

  offsetT off = -1;
  if (info->offset_bits)
    {
      off = get_absolute_expression ();

      if (off < 0)
	as_bad (_("offset is negative"));
    }

  demand_empty_rest_of_line ();

  if (!in_seh_proc)
  {
    as_bad (_("SEH entry has not been found (missing .seh_proc)"));
    return;
  }

  if (!info->directive || !seh_validate_seg (info->directive))
    return;

  seh_aarch64_add_unwind_element (type, off, reg);
}

/* Add a stack-allocation token to current context.  */
static void
obj_coff_seh_stackalloc (const int what ATTRIBUTE_UNUSED)
{
  const offsetT off = get_absolute_expression ();
  demand_empty_rest_of_line ();

  if (!in_seh_proc)
    {
      as_bad (_("SEH entry has not been found (missing .seh_proc)"));
      return;
    }

  if (off < 0x200)
    seh_aarch64_add_unwind_element (unwind_alloc_s, off, 0);
  else if (off < 0x8000)
    seh_aarch64_add_unwind_element (unwind_alloc_m, off, 0);
  else if (off < 0x10000000)
    seh_aarch64_add_unwind_element (unwind_alloc_l, off, 0);
  else
    as_bad (_(".seh_stackalloc offset is out of range"));
}

/* Data writing routines.  */
static void
seh_aarch64_emit_epilogue_scopes (const seh_context *seh_ctx,
				  const uint64_t fragment_offset,
				  const unsigned first_fragment_scope,
				  const unsigned last_fragment_scope)
{
  for (unsigned i = first_fragment_scope; i < last_fragment_scope; ++i)
    {
      seh_aarch64_epilogue_scope scope = seh_ctx->epilogue_scopes[i];
      scope.epilogue_start_offset_reduced = (scope.epilogue_start_offset
					    - fragment_offset) >> 2;

      const unsigned char epilogue_start_index_shift = 22;
      uint32_t scope_code = scope.epilogue_start_offset_reduced;
      scope_code |= scope.epilogue_start_index << epilogue_start_index_shift;

      md_number_to_chars (frag_more (4), scope_code, 4);
    }
}

static void
seh_aarch64_emit_unwind_codes (const seh_context *seh_ctx,
			       const bool has_phantom_prologue)
{
  unsigned total_byte_count = 0;

  if (has_phantom_prologue)
    {
      ++total_byte_count;
      const unsigned endc_code = 0xe5;
      md_number_to_chars (frag_more (1), endc_code, 1);
    }

  for (unsigned i = 0; i < seh_ctx->unwind_codes_count; ++i)
    {
      const seh_aarch64_unwind_code *code = seh_ctx->unwind_codes + i;
      const unsigned byte_count = aarch64_unwind_code_data[code->type].size;

      /*  emit unwind code bytes in big endian.  */
      number_to_chars_bigendian (frag_more (byte_count), code->value,
				 byte_count);
      total_byte_count += byte_count;
    }

    /* Handle word alignment.  */
    const unsigned required_padding = (-total_byte_count) % 4;
    if (required_padding)
      {
	/* Use the nop unwind code for alignment.  */
	const uint32_t nop_chain = 0xe3e3e3e3;

	md_number_to_chars (frag_more (required_padding), nop_chain,
			    required_padding);
      }
}

static void
seh_aarch64_emit_xdata_record (struct seh_aarch64_context *seh_ctx,
			       const uintptr_t frag_size,
			       const uintptr_t fragment_offset,
			       const unsigned first_fragment_scope,
			       const unsigned last_fragment_scope)
{
  unsigned epilogue_count = last_fragment_scope - first_fragment_scope;

  /* Calculate how many unwind bytes will be emitted in .xdata record.  */
  unsigned unwind_bytes = seh_ctx->unwind_codes_byte_count;

  /* Check if current fragment has a phantom prologue.  If yes, then
      the unwinding size should be adjusted.  */
  const bool has_phantom_prologue = fragment_offset != 0;
  if (has_phantom_prologue)
    unwind_bytes += 1;

  /* Calculate the number of code words with 4-byte alignment.  */
  unsigned code_words = (unwind_bytes + 3) / 4;

  /* Initialize the .xdata header.  */
  const unsigned char has_exception_data_shift = 20;
  const unsigned char epilogue_count_shift = 22;
  const unsigned char code_words_shift = 27;
  const unsigned char ext_epilogue_count_shift = 32;
  const unsigned char ext_code_words_shift = 48;
  const uint32_t func_length_encoded = frag_size >> 2;
  uint64_t header = 0;
  header |= func_length_encoded;
  header |= seh_ctx->has_exception_data << has_exception_data_shift;

  /* Check if short or extended header for a .xdata record should be
      used.  */
  unsigned header_size = 8;
  if (code_words < 32 && epilogue_count < 32)
    {
      header_size = 4;
      header |= epilogue_count << epilogue_count_shift;
      header |= code_words << code_words_shift;
    }
  else
    {
      header |= (uint64_t) epilogue_count << ext_epilogue_count_shift;
      header |= (uint64_t) code_words << ext_code_words_shift;
    }

  md_number_to_chars (frag_more (header_size), header, header_size);

  if (epilogue_count)
    seh_aarch64_emit_epilogue_scopes (seh_ctx,
				      fragment_offset,
				      first_fragment_scope,
				      last_fragment_scope);

  seh_aarch64_emit_unwind_codes (seh_ctx, has_phantom_prologue);

  if (seh_ctx->has_exception_data)
    {
      if (seh_ctx->handler.X_op == O_symbol)
	seh_ctx->handler.X_op = O_symbol_rva;

      emit_expr (&seh_ctx->handler, 4);

      /* Emit the fragment offset.  */
      md_number_to_chars (frag_more (4), fragment_offset, 4);

      /* Use the same SEH handler data for all fragments.
	 The SEH handler data is emitted after the last fragment.  */
      expressionS exp;
      memset (&exp, 0, sizeof (expressionS));
      exp.X_op = O_symbol_rva;
      exp.X_add_symbol = seh_ctx->handler_data_xdata_addr;
      emit_expr (&exp, 4);
    }
}

static bool
seh_function_size (const struct seh_aarch64_context *seh_ctx,
		  uintptr_t *size)
{
  fragS *start_frag, *end_frag;
  addressT start_offset, end_offset;
  start_frag = symbol_get_frag_and_value (seh_ctx->start_addr, &start_offset);
  end_frag = symbol_get_frag_and_value (seh_ctx->end_addr, &end_offset);

  intptr_t func_size = end_frag->fr_address + end_offset
		       - start_frag->fr_address - start_offset;
  if (func_size < 0)
    return false;

  *size = func_size;
  return true;
}

/* Write out the xdata information for one function.  */
static void
seh_aarch64_write_function_xdata (struct seh_aarch64_context *seh_ctx)
{
  if (!seh_ctx->unwind_codes_byte_count)
    return;

  const segT save_seg = now_seg;
  const subsegT save_subseg = now_subseg;

  switch_xdata (seh_ctx->subsection, seh_ctx->code_seg);

  /* Set 4-byte alignment.  */
  frag_align (2, 0, 0);

  uintptr_t func_size = 0;
  gas_assert (seh_function_size (seh_ctx, &func_size));

  /* The large functions should be split into fragments smaller than 1MB with
     4 bytes alignment, based on
     "Microsoft ARM64 exception handling, large functions documentation".  */
  const unsigned max_frag_size = (1 << 20) - 4;

  seh_aarch64_func_fragment *fragment = &seh_ctx->func_fragment;
  uintptr_t fragment_offset = 0;
  unsigned first_fragment_scope = 0;
  unsigned last_fragment_scope = 0;

  /* Large functions (>= 1MB) will be split into multiple fragments.
     However, it is expected the most of the functions will have only one
     fragment.  This loop iterates fragments and emits them.  */
  while (true)
    {
      fragment->xdata_addr = symbol_temp_new_now ();
      fragment->offset = fragment_offset;
      fragment->next = NULL;

      /* Calculate current fragment size.  */
      uintptr_t frag_size = func_size - fragment_offset;
      if (frag_size > max_frag_size)
	frag_size = max_frag_size;

      /* If it is a fragmented function, the epilogue range should be calculated
	 and will be emitted for the current fragment, otherwise all epilogues
	 will be emitted.  */
      const bool is_fragmented_function = func_size > max_frag_size;
      if (!is_fragmented_function)
	last_fragment_scope = seh_ctx->epilogue_scopes_count;
      else
	{
	  first_fragment_scope = last_fragment_scope;
	  for (unsigned i = first_fragment_scope;
	       i < seh_ctx->epilogue_scopes_count; ++i)
	    {
	      const seh_aarch64_epilogue_scope *scope
		= seh_ctx->epilogue_scopes;
	      scope += i;
	      if (scope->epilogue_start_offset >= (fragment_offset + frag_size))
		break;

	      if (scope->epilogue_end_offset >= (fragment_offset + frag_size))
		{
		  frag_size = scope->epilogue_start_offset - fragment_offset;
		  break;
		}

	      if (scope->epilogue_start_offset >= fragment_offset)
		last_fragment_scope = i + 1;
	    }
	}


      /* Emit a .xdata record for the current fragment.  */
      seh_aarch64_emit_xdata_record (seh_ctx,
				     frag_size, fragment_offset,
				     first_fragment_scope, last_fragment_scope);

      fragment_offset += frag_size;
      /* Exit the loop if it is the latest fragment.  */
      if (fragment_offset == func_size)
	break;

      /* Allocate a new fragment that will be used also for emitting a .pdata
	 record.  */
      fragment->next = XCNEW (seh_aarch64_func_fragment);
      fragment = fragment->next;
    }

  subseg_set (save_seg, save_subseg);
}

/* Write out pdata for one function.  */
static void
seh_aarch64_write_function_pdata (const seh_context *seh_ctx)
{
  expressionS exp;
  const segT save_seg = now_seg;
  const subsegT save_subseg = now_subseg;
  memset (&exp, 0, sizeof (expressionS));
  switch_pdata (seh_ctx->code_seg);

  if (seh_ctx->unwind_codes_byte_count)
    {
      const seh_aarch64_func_fragment *fragment = &seh_ctx->func_fragment;
      while (fragment)
	{
	  exp.X_op = O_symbol_rva;
	  exp.X_add_number = fragment->offset;
	  exp.X_add_symbol = seh_ctx->start_addr;
	  emit_expr (&exp, 4);

	  exp.X_op = O_symbol_rva;
	  /* TODO: Implementing packed unwind data.  */
	  exp.X_add_number = 0;
	  exp.X_add_symbol = fragment->xdata_addr;
	  emit_expr (&exp, 4);
	  fragment = fragment->next;
	}
    }

  subseg_set (save_seg, save_subseg);
}

void
seh_aarch64_write_data (void)
{
  if (in_seh_proc)
  {
    as_bad (_("open SEH entry at end of file (missing .seh_endproc)"));
    return;
  }

  if (!seh_ctx_root)
    return;

  struct seh_aarch64_context *seh_ctx = seh_ctx_root;
  seh_ctx_root = NULL;

  /* Relax the segment to be able to calculate the function sizes.  */
  subsegs_finish_section (seh_ctx->code_seg);
  const segment_info_type *seginfo = seg_info (seh_ctx->code_seg);
  relax_segment (seginfo->frchainP->frch_root, seh_ctx->code_seg, 0);

  while (seh_ctx)
  {
    seh_aarch64_write_function_xdata (seh_ctx);
    seh_aarch64_write_function_pdata (seh_ctx);
    struct seh_aarch64_context *next = seh_ctx->next;
    free_seh_ctx (seh_ctx);
    seh_ctx = next;
  }
}

void
obj_coff_seh_do_final (void)
{
}
