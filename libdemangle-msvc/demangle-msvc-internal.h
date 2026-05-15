/* Internal interface to libdemangle-msvc's LLVM-based MSVC demangler.
   Copyright (C) 2026 Advanced Micro Devices, Inc.

   Licensed under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.

   This header is private to libdemangle-msvc.  It exposes the C++
   entry points and helper types used by the wrapper that implements
   the public C interface in <demangle-msvc.h>.  External consumers
   (gdb, bfd, binutils) must only include <demangle-msvc.h>.  */

#ifndef LIBDEMANGLE_MSVC_INTERNAL_H
#define LIBDEMANGLE_MSVC_INTERNAL_H

#include <memory>
#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/MicrosoftDemangle.h"

char *llvm_msvc_demangle (const char *mangled_name, int c_demangle_flags);
char *llvm_msvc_class_name_from_physname (const char *physname);
char *llvm_msvc_method_name_from_physname (const char *physname);

std::string llvmNodeKindToString (llvm::ms_demangle::NodeKind K);

llvm::ms_demangle::SymbolNode *
llvm_demangle_symbol_node (const char *mangled_name, void **ctx);

using demangler_ctx_deleter = void (*) (void *);

struct demangler_context_deleter
{
  demangler_context_deleter () = default;
  explicit demangler_context_deleter (demangler_ctx_deleter fn) : fn (fn)
  {
  }

  void
  operator() (llvm::ms_demangle::Demangler *ptr) const
  {
    if (ptr && fn)
      fn (ptr);
  }

private:
  demangler_ctx_deleter fn = nullptr;
};

using demangler_context_ptr
  = std::unique_ptr<llvm::ms_demangle::Demangler, demangler_context_deleter>;

#endif /* LIBDEMANGLE_MSVC_INTERNAL_H */
