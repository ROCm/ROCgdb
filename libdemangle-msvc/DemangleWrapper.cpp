/* MSVC demangling implementation for GDB and binutils.
   Copyright (C) 2026 Advanced Micro Devices, Inc.

   Licensed under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.  */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <cctype>

#include "demangle.h"
#include "demangle-msvc.h"
#include "demangle-msvc-internal.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/MicrosoftDemangle.h"
#include "llvm/Demangle/MicrosoftDemangleNodes.h"
#include "llvm/Demangle/ItaniumDemangle.h"

using namespace llvm::ms_demangle;

/* Remove all MSVC related keywords from the demangled string.  */
void
strip_msvc_keywords_inplace (char *s)
{
  if (s == nullptr || *s == '\0')
    return;

  struct keyword_entry
  {
    const char *text;
    size_t len;
  };

  constexpr keyword_entry msvc_keyword_table[]
    = { { "__cdecl", 7 },
	{ "__stdcall", 9 },
	{ "__fastcall", 10 },
	{ "__thiscall", 10 },
	{ "__vectorcall", 12 },
	{ "__clrcall", 9 },
	{ "__ptr32", 7 },
	{ "__ptr64", 7 },
	{ "__ptr128", 8 },
	{ "__w64", 5 },
	{ "__unaligned", 11 },
	{ "__restrict", 10 },
	{ "__far", 5 },
	{ "__near", 6 },
	{ "__forceinline", 12 },
	{ "__inline", 8 },
	{ "__gc", 4 },
	{ "__value", 7 },
	{ "__interface", 12 },
	{ "class", 5 },
	{ "struct", 6 },
	{ "union", 5 },
	{ "enum", 4 },
	{ "public:", 7 },
	{ "private:", 8 },
	{ "protected:", 10 },
	{ "noexcept", 8 },
	{ "__declspec", 10 } };

  auto is_token_char = [] (char c) -> bool
    {
      unsigned char uc = static_cast<unsigned char> (c);
      return std::isalnum (uc) || c == '_' || c == ':' || c == '~';
    };

  auto is_msvc_keyword_token = [&] (const char *tok, size_t len) -> bool
    {
      for (const auto &kw : msvc_keyword_table)
	if (kw.len == len && std::memcmp (tok, kw.text, len) == 0)
	  return true;
      return false;
    };

  auto skip_spaces = [] (char *p) -> char *
    {
      while (*p && std::isspace (static_cast<unsigned char> (*p)))
	++p;
      return p;
    };

  auto skip_parenthesized_block = [] (char *p) -> char *
    {
      if (*p != '(')
	return p;

      int depth = 1;
      ++p;
      while (*p && depth > 0)
	{
	  if (*p == '(')
	    ++depth;
	  else if (*p == ')')
	    --depth;
	  ++p;
	}

      return p;
    };

  char *read = s;
  char *write = s;

  while (*read)
    {
      if (is_token_char (*read))
	{
	  char *tok_start = read;
	  while (is_token_char (*read))
	    ++read;

	  size_t tok_len = read - tok_start;
	  if (tok_len == 0)
	    continue;

	  bool consumed = false;

	  if (tok_len >= 10 && std::memcmp (tok_start, "__declspec", 10) == 0)
	    {
	      char *cursor = skip_spaces (read);
	      if (*cursor == '(')
		read = skip_parenthesized_block (cursor);
	      consumed = true;
	    }
	  else if ((tok_len == 5 && std::memcmp (tok_start, "throw", 5) == 0)
		   || (tok_len == 8
		       && std::memcmp (tok_start, "nothrow", 8) == 0)
		   || (tok_len == 8
		       && std::memcmp (tok_start, "noexcept", 8) == 0))
	    {
	      char *cursor = skip_spaces (read);
	      if (*cursor == '(')
		read = skip_parenthesized_block (cursor);
	      consumed = true;
	    }

	  if (consumed || is_msvc_keyword_token (tok_start, tok_len))
	    {
	      /* The removed word takes one adjoining space with it: the
		 one after it ("(class A" -> "(A"), or the one before it
		 when nothing follows ("char * __ptr64)" -> "char *)").  */
	      read = skip_spaces (read);
	      if ((*read == '\0' || std::strchr (")>,]", *read) != nullptr)
		  && write > s && write[-1] == ' ')
		--write;
	      continue;
	    }

	  if (write != tok_start)
	    std::memmove (write, tok_start, tok_len);
	  write += tok_len;
	  continue;
	}

      *write++ = *read++;
    }

  *write = '\0';

  /* Collapse spaces inplace below.  */

  char *src = s;
  char *dst = s;
  bool prev_space = false;

  while (*src)
    {
      if (std::isspace (static_cast<unsigned char> (*src)))
	{
	  if (!prev_space && dst != s)
	    {
	      *dst++ = ' ';
	      prev_space = true;
	    }
	}
      else
	{
	  *dst++ = *src;
	  prev_space = false;
	}

      ++src;
    }
  if (dst > s && dst[-1] == ' ')
    --dst;
  *dst = '\0';
}

/* LLVM prints an empty parameter list as "(void)"; write "()" as the
   Itanium demangler does, since GDB's C++ name parser reduces both
   spellings to "()" and compares symbol names against that.  */

static void
msvc_replace_empty_params (char *text)
{
  while ((text = std::strstr (text, "(void)")) != nullptr)
    {
      std::memmove (text + 1, text + 5, std::strlen (text + 5) + 1);
      text += 2;
    }
}

/* MSVC spells an anonymous namespace "`anonymous namespace'"; Itanium
   and GDB spell it "(anonymous namespace)".  Same length, so in place.  */

static void
msvc_replace_anonymous_namespace (char *text)
{
  static const char msvc_spelling[] = "`anonymous namespace'";
  static const char itanium_spelling[] = "(anonymous namespace)";

  while ((text = std::strstr (text, msvc_spelling)) != nullptr)
    {
      std::memcpy (text, itanium_spelling, sizeof itanium_spelling - 1);
      text += sizeof itanium_spelling - 1;
    }
}

/* Return a copy of TEXT with every "__int64" replaced by "long long",
   the name the debug info gives that type.  */

static std::string
msvc_replace_int64 (const char *text)
{
  static const char msvc_spelling[] = "__int64";
  static const char itanium_spelling[] = "long long";

  std::string out (text);
  for (size_t pos = out.find (msvc_spelling); pos != std::string::npos;
       pos = out.find (msvc_spelling, pos + sizeof itanium_spelling - 1))
    out.replace (pos, sizeof msvc_spelling - 1, itanium_spelling);
  return out;
}

/* Static locals, local classes and their members are named in a local
   scope, which MSVC (and LLVM) print as "`S'::`N'::", S the function (or
   other symbol) the scope is in and N the scope number.  Itanium names
   print "S::" instead, so each local scope in TEXT, the demangled
   MANGLED, is replaced with "S::", S demangled with FLAGS from its own
   mangled name, which MANGLED gives after "?N?".  For
   ?x@?1??f@@YAXXZ@4HA:
     MSVC: int `void __cdecl f(void)'::`2'::x
     GDB:  f()::x
   Other backtick names, such as "`anonymous namespace'", are not local
   scopes.  */

static std::string
msvc_respell_local_scopes (const char *mangled, int flags, std::string text)
{
  /* Collect each local scope S from MANGLED: its MSVC spelling
     "`S'::`" as it appears in TEXT, and S's own mangled name.  */
  std::vector<std::pair<std::string, std::string>> scopes;
  for (const char *p = std::strstr (mangled, "@?"); p != nullptr;
       p = std::strstr (p + 1, "@?"))
    {
      /* A scope number is '@' (zero), a digit, or hex digits A-P ended
	 by '@'.  */
      const char *q = p + 2;
      if (*q == '@' || std::isdigit (static_cast<unsigned char> (*q)))
	++q;
      else
	{
	  const char *start = q;
	  while (*q >= 'A' && *q <= 'P')
	    ++q;
	  if (q == start || *q != '@')
	    continue;
	  ++q;
	}
      if (q[0] != '?' || q[1] != '?')
	continue;

      Demangler D;
      std::string_view sv{ q + 1 };
      SymbolNode *scope = D.parse (sv);
      if (D.Error || scope == nullptr)
	continue;
      scopes.emplace_back ("`" + scope->toString (OF_Default) + "'::`",
			   std::string (q + 1,
					std::strlen (q + 1) - sv.size ()));
    }

  /* Replace each "`S'::`N'::" in TEXT with S's mangled name demangled
     by llvm_msvc_demangle with FLAGS, as any other symbol: keywords
     stripped, and local scopes within S respelled by this function in
     turn.  */
  size_t open = 0;
  while ((open = text.find ('`', open)) != std::string::npos)
    {
      /* S may contain quotes of its own, as in "`dynamic initializer for
	 'v''", so its end is found from its known spelling.  */
      size_t end = std::string::npos;
      char *demangled = nullptr;
      for (const auto &s : scopes)
	if (text.compare (open, s.first.size (), s.first) == 0)
	  {
	    end = text.find ("'::", open + s.first.size ());
	    if (end != std::string::npos)
	      demangled = llvm_msvc_demangle (s.second.c_str (), flags);
	    break;
	  }
      if (demangled == nullptr)
	{
	  ++open;
	  continue;
	}
      std::string replacement = std::string (demangled) + "::";
      std::free (demangled);
      text.replace (open, end + 3 - open, replacement);
      open += replacement.size ();
    }
  return text;
}

/* Template instances of conversion operators and of operator<, and the
   constructors and destructors of class template instances, are named
   differently by MSVC (and LLVM) than by GDB's C++ name parser and the
   debug info.  TEXT, the demangled SYM, is returned with SYM's
   unqualified name respelled as GDB spells it:
   - a conversion operator template instance, ??$?BH@conv@@QEBAPEAHXZ:
       LLVM: conv::operator<int> int *(void) const
       GDB:  conv::operator int *<int>() const
   - an operator< template instance,
     ??$?MV?$Empty@H@@@@YAHAEAV?$Empty@H@@0@Z, which MSVC's spelling
     makes operator<<:
       LLVM: operator<<class Empty<int>>(class Empty<int> &,
				      class Empty<int> &)
       GDB:  operator< <Empty<int>>(Empty<int> &, Empty<int> &)
   - a constructor or destructor of a class template instance, without
     the class's template arguments, ??0?$tpl@HM@@QEAA@XZ:
       LLVM: tpl<int, float>::tpl<int, float>(void)
       GDB:  tpl<int, float>::tpl()  */

static std::string
msvc_respell_operators_and_structors (SymbolNode *sym, OutputFlags flags,
				      std::string text)
{
  if (sym->Name == nullptr || sym->Name->Components == nullptr
      || sym->Name->Components->Count == 0)
    return text;

  IdentifierNode *id = sym->Name->getUnqualifiedIdentifier ();
  if (id == nullptr)
    return text;

  if (id->kind () == NodeKind::StructorIdentifier)
    {
      auto *st = static_cast<StructorIdentifierNode *> (id);
      if (st->Class == nullptr || st->Class->TemplateParams == nullptr)
	return text;
      std::string cls = st->Class->toString (flags);
      std::string cls_args
	= "<" + st->Class->TemplateParams->toString (flags) + ">";
      if (cls.size () <= cls_args.size ()
	  || cls.compare (cls.size () - cls_args.size (), cls_args.size (),
			  cls_args) != 0)
	return text;
      std::string structor = "::" + st->toString (flags);
      size_t pos = text.rfind (structor);
      if (pos != std::string::npos)
	text.erase (pos + 2 + (st->IsDestructor ? 1 : 0)
		    + cls.size () - cls_args.size (),
		    cls_args.size ());
      return text;
    }

  if (id->TemplateParams == nullptr)
    return text;

  std::string args = "<" + id->TemplateParams->toString (flags) + ">";

  /* "operator<" followed by template arguments.  */
  if (id->kind () == NodeKind::IntrinsicFunctionIdentifier)
    {
      std::string full = id->toString (flags);
      if (full.size () > args.size ()
	  && full.compare (full.size () - args.size (), args.size (), args) == 0
	  && full[full.size () - args.size () - 1] == '<')
	{
	  size_t pos = text.find (full);
	  if (pos != std::string::npos)
	    text.insert (pos + full.size () - args.size (), " ");
	}
      return text;
    }

  if (id->kind () != NodeKind::ConversionOperatorIdentifier)
    return text;

  auto *conv = static_cast<ConversionOperatorIdentifierNode *> (id);
  if (conv->TargetType == nullptr)
    return text;

  std::string target = conv->TargetType->toString (flags);
  std::string llvm_spelling = "operator" + args + " " + target;
  size_t pos = text.find (llvm_spelling);
  if (pos != std::string::npos)
    text.replace (pos, llvm_spelling.size (), "operator " + target + args);
  return text;
}

/* Set OUT to the demangled type of the RTTI type descriptor RTTI0
   ("??_R0<type>@8").  */

static bool
msvc_rtti0_type (const std::string &rtti0, int flags, std::string &out)
{
  Demangler D;
  std::string_view sv{ rtti0 };
  SymbolNode *sym = D.parse (sv);
  if (D.Error || sym == nullptr || sym->kind () != NodeKind::VariableSymbol)
    return false;

  TypeNode *type = static_cast<VariableSymbolNode *> (sym)->Type;
  if (type == nullptr)
    return false;

  out = type->toString (OF_Default);
  if ((flags & DMGL_MSVC) == 0)
    {
      out = msvc_replace_int64 (out.c_str ());
      strip_msvc_keywords_inplace (&out[0]);
      msvc_replace_empty_params (&out[0]);
      out.resize (std::strlen (out.c_str ()));
    }
  return true;
}

/* Return a malloc'd copy of S, or NULL if allocation fails.  Callers
   free the result with free, like other demangler results.  */

static char *
msvc_xstrdup (const std::string &s)
{
  char *r = static_cast<char *> (std::malloc (s.size () + 1));
  if (r != nullptr)
    std::memcpy (r, s.c_str (), s.size () + 1);
  return r;
}

/* Demangle the EH throw-info names emitted for a thrown type T:
     _CT??_R0<T>@8[<copy ctor>]<size>[<offsets>]   catchable type
     _CTA<count><T>                               catchable type array
     _TI[C][V][U]<count><T>                       throw info
   These are concatenations, not symbols LLVM's parser accepts, so each
   mangled part is demangled separately.  The prefix and the numbers are
   kept, separated by spaces, e.g. "_CTA2 std::runtime_error".  */

static char *
msvc_demangle_throw_info (const char *name, int flags)
{
  std::string_view sv{ name };

  if (sv.substr (0, 8) == "_CT??_R0")
    {
      std::string_view rest = sv.substr (3);
      for (size_t at = rest.find ("@8"); at != std::string_view::npos;
	   at = rest.find ("@8", at + 1))
	{
	  std::string type;
	  if (!msvc_rtti0_type (std::string (rest.substr (0, at + 2)), flags,
				type))
	    continue;

	  std::string out = "_CT " + type;
	  rest.remove_prefix (at + 2);
	  if (!rest.empty () && rest[0] == '?')
	    {
	      Demangler D;
	      std::string_view tail = rest;
	      SymbolNode *ctor = D.parse (tail);
	      if (D.Error || ctor == nullptr)
		return nullptr;

	      std::string ctor_name (rest.substr (0, rest.size ()
						      - tail.size ()));
	      char *ctor_dem = llvm_msvc_demangle (ctor_name.c_str (), flags);
	      if (ctor_dem == nullptr)
		return nullptr;
	      out += ' ';
	      out += ctor_dem;
	      std::free (ctor_dem);
	      rest = tail;
	    }

	  if (rest.empty ())
	    return nullptr;
	  for (char c : rest)
	    if (!std::isdigit (static_cast<unsigned char> (c)))
	      return nullptr;
	  out += ' ';
	  out += rest;
	  return msvc_xstrdup (out);
	}
      return nullptr;
    }

  size_t n;
  if (sv.substr (0, 4) == "_CTA")
    n = 4;
  else if (sv.substr (0, 3) == "_TI")
    for (n = 3; n < sv.size () && std::strchr ("CVU", sv[n]) != nullptr; ++n)
      ;
  else
    return nullptr;

  size_t type_start = n;
  while (type_start < sv.size ()
	 && std::isdigit (static_cast<unsigned char> (sv[type_start])))
    ++type_start;
  if (type_start == n || type_start == sv.size ())
    return nullptr;

  std::string type;
  if (!msvc_rtti0_type ("??_R0" + std::string (sv.substr (type_start)) + "@8",
			flags, type))
    return nullptr;

  return msvc_xstrdup (std::string (sv.substr (0, type_start)) + ' ' + type);
}

char *
llvm_msvc_demangle (const char *mangled_name, int flags)
{
  if (mangled_name[0] == '_')
    return msvc_demangle_throw_info (mangled_name, flags);

  Demangler D;
  std::string_view Name{ mangled_name };
  SymbolNode *ast = D.parse (Name);
  if (D.Error || ast == nullptr)
    return nullptr;

  /* Demangle flags control what is included in
     demangled string.  */
  OutputFlags OF = OF_Default;

  /* Remove attributes not present in itanium.  */
  if ((flags & DMGL_MSVC) == 0)
    {
      OF = OutputFlags (OF | OF_NoCallingConvention);
      OF = OutputFlags (OF | OF_NoAccessSpecifier);
    }

  /* Support demangle flags
       DMGL_VERBOSE
	 libiberty ignores this one and just
	 prints everything always (matches LLVM).
	 FIXME - need implementation for cxxfilt.
       DMGL_ANSI
	 libiberty ignores this one and just
	 prints everything always (matches LLVM).
       DMGL_TYPES
	 libiberty ignores this one and just
	 prints types always (matches LLVM).
       DMGL_PARAMS
       DMGL_RET_DROP
	 Implemented.
       DMGL_MSVC
	 Demangle as MSVC; when set explicitly,
	 preserve MSVC-specific decorations
	 (calling conventions, access specifiers,
	 special symbols).  Under DMGL_AUTO the
	 bit is unset and decorations are stripped.
   */

    /* Automatic mode omits return, variable and member type prefixes,
      matching ordinary Itanium symbol names.  Unlike Itanium function
      templates, MSVC function templates also omit their return type here.
      Explicit MSVC mode preserves these prefixes unless RET_DROP is set.
      OF_NoReturnType would also drop the return types of the function
      types in the parameters, so only the symbol's own one is removed.
      For ?hairyfunc5@@YAHP6AP6AHJ@ZPEAD@Z@Z:
	LLVM: hairyfunc5(int (__cdecl * (__cdecl *)(char *))
	GDB:  hairyfunc5(int (* (*)(char *))(long))  */
    if ((flags & DMGL_MSVC) == 0 || (flags & DMGL_RET_DROP) != 0)
      {
	OF = OutputFlags (OF | OF_NoVariableType | OF_NoMemberType);
	if (auto *F = dynamic_cast<FunctionSymbolNode *> (ast))
	  {
	    if (F->Signature != nullptr)
	      F->Signature->ReturnType = nullptr;
	  }
	else
	  OF = OutputFlags (OF | OF_NoReturnType);
      }

  // Drop function parameters.
  if ((flags & DMGL_PARAMS) == 0)
    {
      auto *F = dynamic_cast<FunctionSymbolNode *> (ast);
      if (F)
	{
	  auto *S = static_cast<FunctionSignatureNode *> (F->Signature);
	  if (S)
	    {
	      S->FunctionClass = FC_NoParameterList;
	    }
	}
    }

  /* MSVC mangles a parameter's top-level const or volatile, which is not
     part of the function's type; Itanium names leave it out.  For
     ?f@@YAXQEAD@Z:
       LLVM: f(char *const)
       GDB:  f(char *)  */
  if ((flags & DMGL_MSVC) == 0)
    if (auto *F = dynamic_cast<FunctionSymbolNode *> (ast))
      if (F->Signature != nullptr && F->Signature->Params != nullptr)
	for (size_t i = 0; i < F->Signature->Params->Count; ++i)
	  {
	    Node *param = F->Signature->Params->Nodes[i];
	    if (auto *P = dynamic_cast<TypeNode *> (param))
	      P->Quals = Qualifiers (P->Quals & ~(Q_Const | Q_Volatile));
	  }

  char *Buf;
  llvm::itanium_demangle::OutputBuffer outBuf;
  ast->output (outBuf, OF);
  outBuf += '\0';
  Buf = outBuf.getBuffer ();

  /* Skip stripping special symbols if DMGL_MSVC is set.  */
  if ((flags & DMGL_MSVC) == 0)
    {
      std::string text
	= msvc_replace_int64 (msvc_respell_local_scopes
			      (mangled_name, flags,
			       msvc_respell_operators_and_structors
				 (ast, OF, Buf)).c_str ());
      /* A compiler-added suffix such as ".cold.1", which LLVM ignores,
	 is kept as the Itanium demangler keeps it.  For
	 ?split_fn@@YAHH@Z.cold.1:
	   LLVM: split_fn(int)
	   GDB:  split_fn(int) [clone .cold.1]  */
      if (!Name.empty () && Name[0] == '.')
	text += " [clone " + std::string (Name) + "]";
      std::free (Buf);
      Buf = msvc_xstrdup (text);
      if (Buf == nullptr)
	return nullptr;
      strip_msvc_keywords_inplace (Buf);
      msvc_replace_empty_params (Buf);
      msvc_replace_anonymous_namespace (Buf);
    }

  return Buf;
}

SymbolNode *
llvm_demangle_symbol_node (const char *mangled_name, void **ctx)
{
  if (ctx == nullptr)
    return nullptr;

  auto demangler = std::make_unique<Demangler> ();
  std::string_view Name{ mangled_name ? mangled_name : "" };
  SymbolNode *ast = demangler->parse (Name);
  if (demangler->Error || ast == nullptr)
    {
      *ctx = nullptr;
      return nullptr;
    }

  *ctx = demangler.release ();
  return ast;
}

static void
msvc_demangler_free (void *ctx)
{
  delete static_cast<Demangler *> (ctx);
}

static std::string
msvc_scope_from_qualified_name (const QualifiedNameNode *name)
{
  if (name == nullptr || name->Components == nullptr
      || name->Components->Count == 0)
    return std::string ();

  const NodeArrayNode *components = name->Components;

  if (components->Count == 1)
    {
      auto *only = components->Nodes[0];
      if (only != nullptr && only->kind () == NodeKind::StructorIdentifier)
	{
	  auto *structor = static_cast<const StructorIdentifierNode *> (only);
	  if (structor->Class != nullptr)
	    return structor->Class->toString ();
	}

      return std::string ();
    }

  std::string result;
  for (size_t i = 0; i + 1 < components->Count; ++i)
    {
      const Node *component = components->Nodes[i];
      if (component == nullptr)
	return std::string ();

      if (i != 0)
	result += "::";
      result += component->toString ();
    }
  return result;
}

/* GDB introduces these calls for no reason; it already
   uses gdb_demangle flags to control demangling so it seems
   natural to add the flags there to achieve the needed.  */
char *
llvm_msvc_class_name_from_physname (const char *physname)
{
  if (physname == nullptr)
    return nullptr;

  void *ctx = nullptr;
  SymbolNode *sym = llvm_demangle_symbol_node (physname, &ctx);
  if (sym == nullptr || ctx == nullptr || sym->Name == nullptr)
    return nullptr;

  demangler_context_ptr ctx_holder (static_cast<Demangler *> (ctx),
				    demangler_context_deleter (
				      msvc_demangler_free));

  std::string scope = msvc_scope_from_qualified_name (sym->Name);
  if (scope.empty ())
    return nullptr;

  char *result = static_cast<char *> (std::malloc (scope.size () + 1));
  if (result == nullptr)
    return nullptr;

  std::memcpy (result, scope.c_str (), scope.size () + 1);
  msvc_replace_anonymous_namespace (result);
  return result;
}

/* Set *CTOR_KIND or *DTOR_KIND when MANGLED names a constructor or
   destructor, as is_ctor_or_dtor does for Itanium names.  MSVC spells
   the complete-object destructor ??_D (only for classes with virtual
   bases), the base-object destructor ??1 and the deleting destructors
   ??_G (scalar) and ??_E (vector).  ??0 is the only constructor.  */

void
llvm_msvc_structor_kind (const char *mangled,
			 enum gnu_v3_ctor_kinds *ctor_kind,
			 enum gnu_v3_dtor_kinds *dtor_kind)
{
  *ctor_kind = (enum gnu_v3_ctor_kinds) 0;
  *dtor_kind = (enum gnu_v3_dtor_kinds) 0;

  if (mangled == nullptr)
    return;

  void *ctx = nullptr;
  SymbolNode *sym = llvm_demangle_symbol_node (mangled, &ctx);
  if (sym == nullptr || ctx == nullptr)
    return;

  demangler_context_ptr ctx_holder (static_cast<Demangler *> (ctx),
				    demangler_context_deleter (
				      msvc_demangler_free));

  if (sym->kind () != NodeKind::FunctionSymbol || sym->Name == nullptr)
    return;

  IdentifierNode *id = sym->Name->getUnqualifiedIdentifier ();
  if (id == nullptr)
    return;

  if (id->kind () == NodeKind::StructorIdentifier)
    {
      if (static_cast<StructorIdentifierNode *> (id)->IsDestructor)
	*dtor_kind = gnu_v3_base_object_dtor;
      else
	*ctor_kind = gnu_v3_complete_object_ctor;
    }
  else if (id->kind () == NodeKind::IntrinsicFunctionIdentifier)
    switch (static_cast<IntrinsicFunctionIdentifierNode *> (id)->Operator)
      {
      case IntrinsicFunctionKind::VbaseDtor:
	*dtor_kind = gnu_v3_complete_object_dtor;
	break;
      case IntrinsicFunctionKind::ScalarDelDtor:
      case IntrinsicFunctionKind::VecDelDtor:
	*dtor_kind = gnu_v3_deleting_dtor;
	break;
      default:
	break;
      }
}

char *
llvm_msvc_method_name_from_physname (const char *physname)
{
  if (physname == nullptr)
    return nullptr;

  void *ctx = nullptr;
  SymbolNode *sym = llvm_demangle_symbol_node (physname, &ctx);
  if (sym == nullptr || ctx == nullptr || sym->Name == nullptr)
    return nullptr;

  demangler_context_ptr ctx_holder (static_cast<Demangler *> (ctx),
				    demangler_context_deleter (
				      msvc_demangler_free));

  auto *id = sym->Name->getUnqualifiedIdentifier ();
  if (id == nullptr)
    return nullptr;

  std::string scope = id->toString ();
  if (scope.empty ())
    return nullptr;

  char *result = static_cast<char *> (std::malloc (scope.size () + 1));
  if (result == nullptr)
    return nullptr;

  std::memcpy (result, scope.c_str (), scope.size () + 1);
  return result;
}
