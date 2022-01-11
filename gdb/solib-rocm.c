/* Handle ROCm Code Objects for GDB, the GNU Debugger.

   Copyright (C) 2019-2022 Free Software Foundation, Inc.
   Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include "gdb/fileio.h"
#include "gdbcore.h"
#include "inferior.h"
#include "objfiles.h"
#include "observable.h"
#include "rocm-tdep.h"
#include "solib-svr4.h"
#include "solib.h"
#include "solist.h"
#include "symfile.h"
#include "gdbsupport/fileio.h"

#include <functional>
#include <string>
#include <unordered_map>

/* ROCm-specific inferior data.  */

struct solib_info
{
  /* List of code objects loaded into the inferior.  */
  struct so_list *solib_list;
};

/* Per-inferior data key.  */
static const struct inferior_key<solib_info> rocm_solib_data;

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
get_solib_info (inferior *inf = nullptr)
{
  if (!inf)
    inf = current_inferior ();

  struct solib_info *info = rocm_solib_data.get (inf);
  if (info == NULL)
    info = rocm_solib_data.emplace (inf);

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

  lm_info_svr4 *li = (lm_info_svr4 *) so->lm_info;
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

      lm_info_svr4 *src_li = (lm_info_svr4 *) src->lm_info;
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

namespace {

/* Interface to interact with a rocm code object stream.  */

struct rocm_code_object_stream
{
  DISABLE_COPY_AND_ASSIGN (rocm_code_object_stream);

  /* Copy SIZE bytes from the underlying objfile storage starting at OFFSET
     into the user provided buffer BUF.  Return the number of bytes actually
     copied (might be inferior to SIZE if the end of the stream is reached).
   */
  virtual file_ptr read (void *buf, file_ptr size, file_ptr offset) = 0;

  /* Retrieve file information in SB.  Return 0 on success.  On failure,
     set the bfd appropriate error number (using bfd_set_error) and return -1.
   */
  int stat (struct stat *sb);

  virtual ~rocm_code_object_stream () = default;

protected:
  rocm_code_object_stream () = default;

  /* Return the size of the object file, or -1 if the size cannot be
     determined.

     This is a helper function for stat.  */
  virtual LONGEST size () = 0;
};

int
rocm_code_object_stream::stat (struct stat *sb)
{
  const LONGEST size = this->size ();
  if (size == -1)
    return -1;

  memset (sb, '\0', sizeof (struct stat));
  sb->st_size = size;
  return 0;
}

/* Interface to a rocm object stream which is embedded in an ELF file
   accessible to the debuggee.  */

struct rocm_code_object_stream_file final : rocm_code_object_stream
{
  DISABLE_COPY_AND_ASSIGN (rocm_code_object_stream_file);

  rocm_code_object_stream_file (int fd, ULONGEST offset, ULONGEST size);

  file_ptr read (void *buf, file_ptr size, file_ptr offset) override;

  LONGEST size () override;

  ~rocm_code_object_stream_file () override;

protected:

  /* The target file descriptor for this stream.  */
  int m_fd;

  /* The offset of the ELF file image in the target file.  */
  ULONGEST m_offset;

  /* The size of the ELF file image.  The value 0 means that it was
     unspecified in the URI descriptor.  */
  ULONGEST m_size;
};

rocm_code_object_stream_file::rocm_code_object_stream_file
  (int fd, ULONGEST offset, ULONGEST size)
     : m_fd { fd }, m_offset { offset }, m_size { size }
{
}

file_ptr
rocm_code_object_stream_file::read (void *buf, file_ptr size,
				    file_ptr offset)
{
  int target_errno;
  file_ptr nbytes = 0;
  while (size > 0)
    {
      QUIT;

      file_ptr bytes_read
	= target_fileio_pread (m_fd, static_cast<gdb_byte *> (buf) + nbytes,
			       size, m_offset + offset + nbytes,
			       &target_errno);

      if (bytes_read == 0)
	break;

      if (bytes_read < 0)
	{
	  errno = fileio_errno_to_host (target_errno);
	  bfd_set_error (bfd_error_system_call);
	  return -1;
	}

      nbytes += bytes_read;
      size -= bytes_read;
    }

  return nbytes;
}

LONGEST
rocm_code_object_stream_file::size ()
{
  if (m_size == 0)
    {
      int target_errno;
      struct stat stat;
      if (target_fileio_fstat (m_fd, &stat, &target_errno) < 0)
	{
	  errno = fileio_errno_to_host (target_errno);
	  bfd_set_error (bfd_error_system_call);
	  return -1;
	}

      /* Check that the offset is valid.  */
      if (m_offset >= stat.st_size)
	{
	  bfd_set_error (bfd_error_bad_value);
	  return -1;
	}

      m_size = stat.st_size - m_offset;
    }

  return m_size;
}

rocm_code_object_stream_file::~rocm_code_object_stream_file ()
{
  int target_errno;
  target_fileio_close (m_fd, &target_errno);
}

/* Interface to a code object which lives in the inferior's memory.  */

struct rocm_code_object_stream_memory final : public rocm_code_object_stream
{
  DISABLE_COPY_AND_ASSIGN (rocm_code_object_stream_memory);

  rocm_code_object_stream_memory (ULONGEST offset, ULONGEST size);

  file_ptr read (void *buf, file_ptr size, file_ptr offset) override;

protected:

  /* The offset of the ELF file image in the target memory.  */
  ULONGEST m_offset;

  /* The size of the ELF file image in target memory.  */
  ULONGEST m_size;

  LONGEST size () override
  {
    return m_size;
  }
};

rocm_code_object_stream_memory::rocm_code_object_stream_memory
  (ULONGEST offset, ULONGEST size)
     : m_offset { offset }, m_size { size }
{
}

file_ptr
rocm_code_object_stream_memory::read (void *buf, file_ptr size,
				      file_ptr offset)
{
  if (target_read_memory (m_offset + offset, (gdb_byte *) buf, size)
      != 0)
    {
      bfd_set_error (bfd_error_invalid_operation);
      return -1;
    }
  return size;
}

} // anonymous namespace

static void *
rocm_bfd_iovec_open (bfd *abfd, void *inferior)
{
  std::string uri (bfd_get_filename (abfd));

  std::string protocol_delim ("://");
  size_t protocol_end = uri.find (protocol_delim);
  std::string protocol = uri.substr (0, protocol_end);
  protocol_end += protocol_delim.length ();

  std::transform (protocol.begin (), protocol.end (), protocol.begin (),
		  [] (unsigned char c) { return std::tolower (c); });

  std::string path;
  size_t path_end = uri.find_first_of ("#?", protocol_end);
  if (path_end != std::string::npos)
    path = uri.substr (protocol_end, path_end++ - protocol_end);
  else
    path = uri.substr (protocol_end);

  /* %-decode the string.  */
  std::string decoded_path;
  decoded_path.reserve (path.length ());
  for (size_t i = 0; i < path.length (); ++i)
    if (path[i] == '%' && std::isxdigit (path[i + 1])
	&& std::isxdigit (path[i + 2]))
      {
	decoded_path += std::stoi (path.substr (i + 1, 2), 0, 16);
	i += 2;
      }
    else
      decoded_path += path[i];

  /* Tokenize the query/fragment.  */
  std::vector<std::string> tokens;
  size_t pos, last = path_end;
  while ((pos = uri.find ('&', last)) != std::string::npos)
    {
      tokens.emplace_back (uri.substr (last, pos - last));
      last = pos + 1;
    }
  if (last != std::string::npos)
    tokens.emplace_back (uri.substr (last));

  /* Create a tag-value map from the tokenized query/fragment.  */
  std::unordered_map<std::string, std::string> params;
  std::for_each (tokens.begin (), tokens.end (),
		 [&] (std::string &token)
		 {
		   size_t delim = token.find ('=');
		   if (delim != std::string::npos)
		     params.emplace (token.substr (0, delim),
				     token.substr (delim + 1));
		 });

  try
    {
      ULONGEST offset = 0;
      ULONGEST size = 0;

      auto offset_it = params.find ("offset");
      if (offset_it != params.end ())
	offset = std::stoul (offset_it->second, nullptr, 0);

      auto size_it = params.find ("size");
      if (size_it != params.end ())
	if (!(size = std::stoul (size_it->second, nullptr, 0)))
	  {
	    bfd_set_error (bfd_error_bad_value);
	    return nullptr;
	  }

      if (protocol == "file")
	{
	  int target_errno;
	  int fd
	    = target_fileio_open (static_cast<struct inferior *> (inferior),
				  decoded_path.c_str (), FILEIO_O_RDONLY,
				  false, 0, &target_errno);

	  if (fd == -1)
	    {
	      /* FIXME: Should we set errno?  Move fileio_errno_to_host from
		 gdb_bfd.c to fileio.cc  */
	      /* errno = fileio_errno_to_host (target_errno); */
	      bfd_set_error (bfd_error_system_call);
	      return nullptr;
	    }

	  return new rocm_code_object_stream_file (fd, offset, size);
	}
      else if (protocol == "memory")
	{
	  pid_t pid = std::stoul (path);
	  if (pid != static_cast<struct inferior *> (inferior)->pid)
	    {
	      warning (_ ("`%s': code object is from another inferior"),
		       uri.c_str ());
	      bfd_set_error (bfd_error_bad_value);
	      return nullptr;
	    }

	  return new rocm_code_object_stream_memory (offset, size);
	}
    }
  catch (...)
    {
      bfd_set_error (bfd_error_bad_value);
      return nullptr;
    }

  warning (_ ("`%s': protocol not supported: %s"), uri.c_str (),
	   protocol.c_str ());
  bfd_set_error (bfd_error_bad_value);
  return nullptr;
}

static int
rocm_bfd_iovec_close (bfd *nbfd, void *data)
{
  delete static_cast<rocm_code_object_stream *> (data);

  return 0;
}

static file_ptr
rocm_bfd_iovec_pread (bfd *abfd, void *data, void *buf, file_ptr size,
		      file_ptr offset)
{
  return static_cast<rocm_code_object_stream *> (data)->read (buf, size,
							      offset);
}

static int
rocm_bfd_iovec_stat (bfd *abfd, void *data, struct stat *sb)
{
  return static_cast<rocm_code_object_stream *> (data)->stat (sb);
}

static gdb_bfd_ref_ptr
rocm_solib_bfd_open (const char *pathname)
{
  /* Handle regular files with SVR4 open.  */
  if (!strstr (pathname, "://"))
    return svr4_so_ops.bfd_open (pathname);

  gdb_bfd_ref_ptr abfd (
    gdb_bfd_openr_iovec (pathname, "elf64-amdgcn", rocm_bfd_iovec_open,
			 current_inferior (), rocm_bfd_iovec_pread,
			 rocm_bfd_iovec_close, rocm_bfd_iovec_stat));

  if (abfd == nullptr)
    error (_ ("Could not open `%s' as an executable file: %s"), pathname,
	   bfd_errmsg (bfd_get_error ()));

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
  if (process_id.handle == AMD_DBGAPI_PROCESS_NONE.handle)
    return;

  solib_info *info = get_solib_info ();
  amd_dbgapi_status_t status;

  rocm_free_solib_list (info);
  struct so_list **link = &info->solib_list;

  amd_dbgapi_code_object_id_t *code_object_list;
  size_t count;

  if ((status
       = amd_dbgapi_process_code_object_list (process_id, &count,
					      &code_object_list, nullptr))
      != AMD_DBGAPI_STATUS_SUCCESS)
    {
      warning (_ ("amd_dbgapi_code_object_list failed (%d)"), status);
      return;
    }

  for (size_t i = 0; i < count; ++i)
    {
      CORE_ADDR l_addr;
      char *uri_bytes;

      if (amd_dbgapi_code_object_get_info (
	    code_object_list[i], AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS,
	    sizeof (l_addr), &l_addr)
	    != AMD_DBGAPI_STATUS_SUCCESS
	  || amd_dbgapi_code_object_get_info (
	       code_object_list[i], AMD_DBGAPI_CODE_OBJECT_INFO_URI_NAME,
	       sizeof (uri_bytes), &uri_bytes)
	       != AMD_DBGAPI_STATUS_SUCCESS)
	continue;

      struct so_list *so = XCNEW (struct so_list);
      lm_info_svr4 *li = new lm_info_svr4;
      li->l_addr = l_addr;
      so->lm_info = li;

      strncpy (so->so_name, uri_bytes, sizeof (so->so_name));
      so->so_name[sizeof (so->so_name) - 1] = '\0';
      xfree (uri_bytes);

      /* Make so_original_name unique so that code objects with the same URI
	 but different load addresses are seen by gdb core as different shared
	 objects.  */
      xsnprintf (so->so_original_name, sizeof (so->so_original_name),
		 "code_object_%ld", code_object_list[i].handle);

      so->next = nullptr;
      *link = so;
      link = &so->next;
    }

  xfree (code_object_list);

  /* Force GDB to reload the solibs.  */
  current_inferior ()->pspace->clear_solib_cache ();

  /* Switch terminal for any messages produced by
     breakpoint_re_set.  */
  target_terminal::ours_for_output ();

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

      /* Engage the ROCm so_ops.  */
      set_solib_ops (target_gdbarch (), &rocm_solib_ops);
    }

  solib_add (NULL, 0, auto_solib_add);

  /* Switch it back.  */
  target_terminal::inferior ();
}

static void
rocm_solib_target_inferior_created (inferior *inf)
{
  rocm_free_solib_list (get_solib_info (inf));
  rocm_update_solib_list ();
}

/* -Wmissing-prototypes */
extern initialize_file_ftype _initialize_rocm_solib;

void
_initialize_rocm_solib ()
{
  /* Install our observers.  */
  amd_dbgapi_code_object_list_updated.attach (rocm_update_solib_list,
					      "solib-rocm");

  /* FIXME: remove this when we can clear the solist in
     rocm_solib_create_inferior_hook.  */
  gdb::observers::inferior_created.attach (rocm_solib_target_inferior_created,
					   "solib-rocm");
}
