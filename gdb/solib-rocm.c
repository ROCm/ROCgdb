/* Handle ROCm Code Objects for GDB, the GNU Debugger.

   Copyright (C) 2019 Free Software Foundation, Inc.
   Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.

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

#include "defs.h"

#include "arch-utils.h"
#include "elf-bfd.h"
#include "gdbcore.h"
#include "inferior.h"
#include "objfiles.h"
#include "observable.h"
#include "rocm-tdep.h"
#include "solib-svr4.h"
#include "solib.h"
#include "solist.h"
#include "symfile.h"

#include <functional>
#include <string>

#define ROCM_DSO_NAME_PREFIX "AMDGPU shared object [loaded from memory "

#define ROCM_DSO_NAME_SUFFIX "]"

/* ROCm-specific inferior data.  */

struct solib_info
{
  /* List of code objects loaded into the inferior.  */
  struct so_list *solib_list;
};

/* Per-inferior data key.  */
static const struct inferior_data *rocm_solib_data;

struct target_so_ops rocm_solib_ops;

/* Free the solib linked list.  */

static void
rocm_free_solib_list (struct solib_info *info)
{
  while (info->solib_list != NULL)
    {
      struct so_list *next = info->solib_list->next;

      free_so (info->solib_list);
      info->solib_list = next;
    }

  info->solib_list = NULL;
}

/* Fetch the solib_info data for the current inferior.  */

static struct solib_info *
get_solib_info (void)
{
  struct inferior *inf = current_inferior ();
  struct solib_info *info;

  info = (struct solib_info *)inferior_data (inf, rocm_solib_data);
  if (info == NULL)
    {
      info = XCNEW (struct solib_info);
      set_inferior_data (inf, rocm_solib_data, info);
    }

  return info;
}

/* Relocate section addresses.  */

static void
rocm_solib_relocate_section_addresses (struct so_list *so,
                                       struct target_section *sec)
{
  if (!rocm_is_amdgcn_gdbarch (gdbarch_from_bfd (so->abfd)))
    {
      svr4_so_ops.relocate_section_addresses (so, sec);
      return;
    }

  lm_info_svr4 *li = (lm_info_svr4 *)so->lm_info;
  sec->addr = sec->addr + li->l_addr;
  sec->endaddr = sec->endaddr + li->l_addr;
}

/* Make a deep copy of the solib linked list.  */

static struct so_list *
rocm_solib_copy_list (const struct so_list *src)
{
  struct so_list *dst = NULL;
  struct so_list **link = &dst;

  while (src != NULL)
    {
      struct so_list *newobj;

      newobj = XNEW (struct so_list);
      memcpy (newobj, src, sizeof (struct so_list));

      lm_info_svr4 *src_li = (lm_info_svr4 *)src->lm_info;
      newobj->lm_info = new lm_info_svr4 (*src_li);

      newobj->next = NULL;
      *link = newobj;
      link = &newobj->next;

      src = src->next;
    }

  return dst;
}

/* Build a list of `struct so_list' objects describing the shared
   objects currently loaded in the inferior.  */

static struct so_list *
rocm_solib_current_sos (void)
{
  /* First, retrieve the host-side shared library list.  */
  struct so_list *head = svr4_so_ops.current_sos ();

  /* Then, the device-side shared library list.  */
  struct so_list *list = get_solib_info ()->solib_list;

  if (!list)
    return head;

  list = rocm_solib_copy_list (list);

  if (!head)
    return list;

  /* Append our libraries to the end of the list.  */
  struct so_list *tail;
  for (tail = head; tail->next; tail = tail->next)
    /* Nothing.  */;
  tail->next = list;

  return head;
}

struct target_elf_image
{
  /* The base address of the ELF file image in the inferior's memory.  */
  CORE_ADDR base_addr;

  /* The size of the ELF file image.  */
  ULONGEST size;
};

static void *
rocm_bfd_iovec_open (bfd *nbfd, void *open_closure)
{
  return open_closure;
}

static int
rocm_bfd_iovec_close (bfd *nbfd, void *stream)
{
  xfree (stream);
  return 0;
}

static file_ptr
rocm_bfd_iovec_pread (bfd *abfd, void *stream, void *buf, file_ptr nbytes,
                      file_ptr offset)
{
  CORE_ADDR addr = ((target_elf_image *)stream)->base_addr;

  if (target_read_memory (addr + offset, (gdb_byte *)buf, nbytes) != 0)
    {
      bfd_set_error (bfd_error_invalid_operation);
      return -1;
    }

  return nbytes;
}

static int
rocm_bfd_iovec_stat (bfd *abfd, void *stream, struct stat *sb)
{
  memset (sb, '\0', sizeof (struct stat));
  sb->st_size = ((struct target_elf_image *)stream)->size;
  return 0;
}

static gdb_bfd_ref_ptr
rocm_solib_bfd_open (const char *pathname)
{
  struct target_elf_image *open_closure;
  CORE_ADDR addr, end;

  /* Handle regular SVR4 libraries.  */
  if (strstr (pathname, ROCM_DSO_NAME_PREFIX) != pathname)
    return svr4_so_ops.bfd_open (pathname);

  /* Decode the start and end addresses.  */
  if (sscanf (pathname + sizeof (ROCM_DSO_NAME_PREFIX) - 1, "0x%lx..0x%lx",
              &addr, &end)
      != 2)
    internal_error (__FILE__, __LINE__, "ROCm-GDB Error: bad DSO name: `%s'",
                    pathname);

  open_closure = XNEW (struct target_elf_image);
  open_closure->base_addr = addr;
  open_closure->size = end - addr;

  gdb_bfd_ref_ptr abfd (gdb_bfd_openr_iovec (
      pathname, "elf64-amdgcn", rocm_bfd_iovec_open, open_closure,
      rocm_bfd_iovec_pread, rocm_bfd_iovec_close, rocm_bfd_iovec_stat));

  /* Check bfd format.  */
  if (!bfd_check_format (abfd.get (), bfd_object))
    error (_ ("`%s': not in executable format: %s"),
           bfd_get_filename (abfd.get ()), bfd_errmsg (bfd_get_error ()));

  unsigned char osabi = elf_elfheader (abfd)->e_ident[EI_OSABI];
  unsigned char osabiversion = elf_elfheader (abfd)->e_ident[EI_ABIVERSION];

  /* Make a check if the code object in the elf is V3. ROCM-gdb has
     support only for V3.  */
  if (osabi != ELFOSABI_AMDGPU_HSA)
    error (_ ("`%s': ELF file OS ABI invalid (%d)."),
           bfd_get_filename (abfd.get ()), osabi);

  if (osabi == ELFOSABI_AMDGPU_HSA && osabiversion < 1)
    error (_ ("`%s': ELF file ABI version (%d) is not supported."),
           bfd_get_filename (abfd.get ()), osabiversion);

  return abfd;
}

static void
rocm_solib_create_inferior_hook (int from_tty)
{
  rocm_free_solib_list (get_solib_info ());

  svr4_so_ops.solib_create_inferior_hook (from_tty);
}

static void
rocm_update_solib_list ()
{
  amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id ();
  solib_info *info = get_solib_info ();
  amd_dbgapi_status_t status;

  rocm_free_solib_list (info);
  struct so_list **link = &info->solib_list;

  amd_dbgapi_code_object_id_t *code_object_list;
  size_t count;

  if ((status = amd_dbgapi_code_object_list (process_id, &count,
                                             &code_object_list, nullptr))
      != AMD_DBGAPI_STATUS_SUCCESS)
    {
      warning (_ ("amd_dbgapi_code_object_list failed (%d)"), status);
      return;
    }

  for (size_t i = 0; i < count; ++i)
    {
      struct so_list *so = XCNEW (struct so_list);
      lm_info_svr4 *li = new lm_info_svr4;
      so->lm_info = li;

      char *uri_bytes;

      if (amd_dbgapi_code_object_get_info (
              process_id, code_object_list[i],
              AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS, sizeof (li->l_addr),
              &li->l_addr)
              != AMD_DBGAPI_STATUS_SUCCESS
          || amd_dbgapi_code_object_get_info (
                 process_id, code_object_list[i],
                 AMD_DBGAPI_CODE_OBJECT_INFO_URI_NAME, sizeof (uri_bytes),
                 &uri_bytes)
                 != AMD_DBGAPI_STATUS_SUCCESS)
        continue;

      /* FIXME: We need to properly decode the URI.  */

      std::string uri (uri_bytes);
      xfree (uri_bytes);

      size_t address_pos = uri.find ("://");
      if (address_pos == std::string::npos)
        continue;
      address_pos += 3;

      size_t fragment_pos = uri.find ('#', address_pos);
      if (address_pos == std::string::npos)
        continue;

      std::string fragment = uri.substr (fragment_pos);

      /* Decode the offset and size.  */
      amd_dbgapi_global_address_t mem_addr;
      amd_dbgapi_size_t mem_size;

      if (sscanf (fragment.c_str (), "#offset=0x%lx&size=0x%lx", &mem_addr,
                  &mem_size)
          != 2)
        internal_error (__FILE__, __LINE__,
                        "ROCm-GDB Error: bad DSO name: `%s'", uri.c_str ());

      xsnprintf (so->so_name, sizeof so->so_name,
                 ROCM_DSO_NAME_PREFIX "%s..%s" ROCM_DSO_NAME_SUFFIX,
                 hex_string (mem_addr), hex_string (mem_addr + mem_size));
      strcpy (so->so_original_name, so->so_name);

      so->next = nullptr;
      *link = so;
      link = &so->next;
    }

  xfree (code_object_list);

  /* Force GDB to reload the solibs.  */
  clear_program_space_solib_cache (current_inferior ()->pspace);

  /* Switch terminal for any messages produced by
     breakpoint_re_set.  */
  target_terminal::ours_for_output ();

  solib_add (NULL, 0, auto_solib_add);

  /* Switch it back.  */
  target_terminal::inferior ();
}

static void
rocm_solib_dbgapi_activated ()
{
  if (rocm_solib_ops.current_sos == NULL)
    {
      /* Override what we need to */
      rocm_solib_ops = svr4_so_ops;
      rocm_solib_ops.current_sos = rocm_solib_current_sos;
      rocm_solib_ops.solib_create_inferior_hook
          = rocm_solib_create_inferior_hook;
      rocm_solib_ops.bfd_open = rocm_solib_bfd_open;
      rocm_solib_ops.relocate_section_addresses
          = rocm_solib_relocate_section_addresses;
    }

  /* Engage the ROCm so_ops.  */
  set_solib_ops (target_gdbarch (), &rocm_solib_ops);
}

static void
rocm_solib_dbgapi_deactivated ()
{
  /* Disengage the ROCm so_ops.  */
  set_solib_ops (target_gdbarch (), &svr4_so_ops);
}

/* Handles the cleanup of the per-inferior solib_info data.  */

static void
rocm_solib_inferior_cleanup (struct inferior *inf, void *arg)
{
  struct solib_info *info = static_cast<struct solib_info *> (arg);

  rocm_free_solib_list (info);

  xfree (info);
}

static void
rocm_solib_target_inferior_created (struct target_ops *target, int from_tty)
{
  rocm_free_solib_list (get_solib_info ());
  rocm_update_solib_list ();
}

/* -Wmissing-prototypes */
extern initialize_file_ftype _initialize_rocm_solib;

void
_initialize_rocm_solib (void)
{
  /* Set a per-inferior rocm_inferior_info.  */
  rocm_solib_data = register_inferior_data_with_cleanup (
      NULL, rocm_solib_inferior_cleanup);

  /* Install our observers.  */
  amd_dbgapi_activated.attach (rocm_solib_dbgapi_activated);
  amd_dbgapi_deactivated.attach (rocm_solib_dbgapi_deactivated);
  amd_dbgapi_code_object_list_updated.attach (rocm_update_solib_list);

  /* FIXME: remove this when we can clear the solist in
     rocm_solib_create_inferior_hook.  */
  gdb::observers::inferior_created.attach (rocm_solib_target_inferior_created);
}
