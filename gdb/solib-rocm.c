/* Handle ROCm Code Objects for GDB, the GNU Debugger.

   Copyright (C) 2019-2026 Free Software Foundation, Inc.

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


#include "amd-dbgapi-target.h"
#include "amdgpu-tdep.h"
#include "elf-bfd.h"
#include "elf/amdgpu.h"
#include "event-top.h"
#include "gdbsupport/fileio.h"
#include "inferior.h"
#include "linux-tdep.h"
#include "observable.h"
#include "solib.h"
#include "solib-svr4.h"
#include "symfile.h"
#include "filesystem.h"

namespace {

/* Per inferior cache of opened file descriptors.  */
struct rocm_solib_fd_cache
{
  explicit rocm_solib_fd_cache (inferior *inf) : m_inferior (inf) {}
  DISABLE_COPY_AND_ASSIGN (rocm_solib_fd_cache);

  /* The open method returns an object of this type.

     On destruction, automatically call close to indicate the caller no longer
     uses the fd.  */
  struct cached_target_fd
  {
    cached_target_fd (target_fd fd, rocm_solib_fd_cache &cache)
      : m_fd (fd),
	m_cache (cache)
    {
    }

    cached_target_fd (cached_target_fd &&other)
      : m_fd (other.m_fd),
	m_cache (other.m_cache)
    {
      other.m_fd = target_fd::INVALID;
    }

    ~cached_target_fd ()
    {
      if (m_fd == target_fd::INVALID)
	return;

      fileio_error target_errno;
      if (m_cache.close (m_fd, &target_errno) != 0)
	warning (_("Failed to close solib: %s"),
		 strerror (fileio_error_to_host (target_errno)));
    }

    DISABLE_COPY_AND_ASSIGN (cached_target_fd);

    target_fd fd () const { return m_fd; }

  private:
    target_fd m_fd;
    rocm_solib_fd_cache &m_cache;
  };

  /* Return a read-only file descriptor to FILENAME and increment the
     associated reference count.

     Open the file FILENAME if it is not already opened, reuse the existing file
     descriptor otherwise.

     On error, return a cached_target_fd with target_fd::INVALID and set
     *TARGET_ERRNO.  */
  cached_target_fd open (const std::string &filename,
			 fileio_error *target_errno);

private:
  /* Decrement the reference count to FD and close FD if the reference count
     reaches 0.

     On success, return 0.  On error, return -1 and set TARGET_ERRNO.  */
  int close (target_fd fd, fileio_error *target_errno);

  struct refcnt_fd
  {
    refcnt_fd (target_fd fd, int refcnt) : fd (fd), refcnt (refcnt) {}

    DISABLE_COPY_AND_ASSIGN (refcnt_fd);

    refcnt_fd (refcnt_fd &&) = default;
    refcnt_fd &operator=(refcnt_fd &&other) = default;

    target_fd fd = target_fd::INVALID;
    int refcnt = 0;
  };

  inferior *m_inferior;
  gdb::unordered_string_map<refcnt_fd> m_cache;
};

rocm_solib_fd_cache::cached_target_fd
rocm_solib_fd_cache::open (const std::string &filename,
			   fileio_error *target_errno)
{
  auto it = m_cache.find (filename);
  if (it == m_cache.end ())
    {
      /* The file is not yet opened on the target.  */
      target_fd fd
	= target_fileio_open (m_inferior, filename.c_str (), FILEIO_O_RDONLY,
			      0, false, target_errno);
      if (fd != target_fd::INVALID)
	m_cache.emplace (std::piecewise_construct,
			 std::forward_as_tuple (filename),
			 std::forward_as_tuple (fd, 1));
      return cached_target_fd { fd, *this };
    }
  else
    {
      /* The file is already opened.  Increment the refcnt and return the
	 already opened FD.  */
      it->second.refcnt++;
      gdb_assert (it->second.fd != target_fd::INVALID);
      return cached_target_fd { it->second.fd, *this };
    }
}

int
rocm_solib_fd_cache::close (target_fd fd, fileio_error *target_errno)
{
  using cache_val = gdb::unordered_string_map<refcnt_fd>::value_type;
  auto it
    = std::find_if (m_cache.begin (), m_cache.end (),
		    [fd](const cache_val &s) { return s.second.fd == fd; });

  gdb_assert (it != m_cache.end ());

  it->second.refcnt--;
  if (it->second.refcnt == 0)
    {
      int ret = target_fileio_close (it->second.fd, target_errno);
      m_cache.erase (it);
      return ret;
    }
  else
    {
      /* Keep the FD open for the other users, return success.  */
      return 0;
    }
}

} /* Anonymous namespace.  */

/* ROCm-specific inferior data.  */

struct rocm_so
{
  rocm_so (const char *name, std::string unique_name, lm_info_svr4_up lm_info)
    : name (name),
      unique_name (std::move (unique_name)),
      lm_info (std::move (lm_info))
  {}

  std::string name, unique_name;
  lm_info_svr4_up lm_info;
};

/* solib_ops for ROCm systems.  */

struct rocm_solib_ops : public solib_ops
{
  /* HOST_OPS is the host solib_ops that rocm_solib_ops hijacks / wraps,
     in order to provide support for ROCm code objects.  */
  explicit rocm_solib_ops (inferior *inf, solib_ops_up host_ops)
    : solib_ops (inf->pspace), m_host_ops (std::move (host_ops)),
      m_fd_cache (inf)
  {
    gdb_assert (m_host_ops != nullptr);
    gdb_assert (dynamic_cast<rocm_solib_ops *> (m_host_ops.get ()) == nullptr);
  }

  /* Release the host solib_ops.  */
  solib_ops_up release_host_ops ()
  { return std::move (m_host_ops); }

  /* The methods implemented by rocm_solib_ops.  */
  owning_intrusive_list<solib> current_sos () override;
  void create_inferior_hook (int from_tty) override;
  gdb_bfd_ref_ptr bfd_open (const char *pathname) override;
  void relocate_section_addresses (solib &so, target_section *) const override;
  void handle_event () override;

  /* This needs to be public because it is called from
     rocm_solib_target_inferior_created.  */
  void update_solib_list ();

  /* Implement the following methods just to forward the calls to the host
     solib_ops.  We currently need to implement all the methods that
     svr4_solib_ops implements.  */
  void clear_so (const solib &so) const override
  { return m_host_ops->clear_so (so); }

  void clear_solib (program_space *pspace) const override
  { return m_host_ops->clear_solib (pspace); }

  bool open_symbol_file_object (int from_tty) const override
  { return m_host_ops->open_symbol_file_object (from_tty); }

  bool in_dynsym_resolve_code (CORE_ADDR pc) const override
  { return m_host_ops->in_dynsym_resolve_code (pc); }

  bool same (const solib &gdb, const solib &inferior) const override
  { return m_host_ops->same (gdb, inferior); }

  bool keep_data_in_core (CORE_ADDR vaddr, unsigned long size) const override
  { return m_host_ops->keep_data_in_core (vaddr, size); }

  void update_breakpoints () const override
  { return m_host_ops->update_breakpoints (); }

  std::optional<CORE_ADDR> find_solib_addr (solib &so) const override
  { return m_host_ops->find_solib_addr (so); }

  bool supports_namespaces () const override
  { return m_host_ops->supports_namespaces (); }

  int find_solib_ns (const solib &so) const override
  { return m_host_ops->find_solib_ns (so); }

  int num_active_namespaces () const override
  { return m_host_ops->num_active_namespaces (); }

  std::vector<const solib *> get_solibs_in_ns (int nsid) const override
  { return m_host_ops->get_solibs_in_ns (nsid); }

  void iterate_over_objfiles_in_search_order
    (iterate_over_objfiles_in_search_order_cb_ftype cb,
     objfile *current_objfile) const override
  {
    return m_host_ops->iterate_over_objfiles_in_search_order
      (cb, current_objfile);
  }

private:
  owning_intrusive_list<solib>
  solibs_from_rocm_sos (const std::vector<rocm_so> &sos);
  gdb_bfd_iovec_base *bfd_iovec_open (bfd *abfd, inferior *inferior);

  solib_ops_up m_host_ops;

  /* List of code objects loaded into the inferior.  */
  std::vector<rocm_so> m_solib_list;

  /* Cache of opened FD in the inferior.  */
  rocm_solib_fd_cache m_fd_cache;
};

/* Relocate section addresses.  */

void
rocm_solib_ops::relocate_section_addresses (solib &so,
					    struct target_section *sec) const
{
  if (!is_amdgpu_arch (so.abfd.get ()))
    {
      m_host_ops->relocate_section_addresses (so, sec);
      return;
    }

  auto *li = gdb::checked_static_cast<lm_info_svr4 *> (so.lm_info.get ());
  sec->addr = sec->addr + li->l_addr;
  sec->endaddr = sec->endaddr + li->l_addr;
}

void
rocm_solib_ops::handle_event ()
{
  /* Since we sit on top of a host solib_ops, we might get called following an
     event concerning host libraries.  We must therefore forward the call.  If
     the event was for a ROCm code object, it will be a no-op.  On the other hand
     if the event was for host libraries, rocm_update_solib_list will be
     essentially be a no-op (it will reload the same code object list as was
     previously loaded).  */
  m_host_ops->handle_event ();

  this->update_solib_list ();
}

/* Create solib objects from rocm_so objects in SOS.  */

owning_intrusive_list<solib>
rocm_solib_ops::solibs_from_rocm_sos (const std::vector<rocm_so> &sos)
{
  owning_intrusive_list<solib> dst;

  for (const rocm_so &so : sos)
    dst.emplace_back (std::make_unique<lm_info_svr4> (*so.lm_info),
		      so.unique_name, so.name, *this);

  return dst;
}

/* Build a list of `struct solib' objects describing the shared
   objects currently loaded in the inferior.  */

owning_intrusive_list<solib>
rocm_solib_ops::current_sos ()
{
  /* First, retrieve the host-side shared library list.  */
  owning_intrusive_list<solib> sos = m_host_ops->current_sos ();

  /* Then, the device-side shared library list.  */
  if (m_solib_list.empty ())
    return sos;

  owning_intrusive_list<solib> dev_solibs = solibs_from_rocm_sos (m_solib_list);

  if (sos.empty ())
    return dev_solibs;

  /* Append our libraries to the end of the list.  */
  sos.splice (std::move (dev_solibs));

  return sos;
}

namespace {

/* Interface to interact with a ROCm code object stream.  */

struct rocm_code_object_stream : public gdb_bfd_iovec_base
{
  DISABLE_COPY_AND_ASSIGN (rocm_code_object_stream);

  int stat (bfd *abfd, struct stat *sb) final override;

  ~rocm_code_object_stream () override = default;

protected:
  rocm_code_object_stream () = default;

  /* Return the size of the object file, or -1 if the size cannot be
     determined.

     This is a helper function for stat.  */
  virtual LONGEST size () = 0;
};

int
rocm_code_object_stream::stat (bfd *, struct stat *sb)
{
  const LONGEST size = this->size ();
  if (size == -1)
    return -1;

  memset (sb, '\0', sizeof (struct stat));
  sb->st_size = size;
  return 0;
}

/* Interface to a ROCm object stream which is embedded in an ELF file
   accessible to the debugger.  */

struct rocm_code_object_stream_file final : rocm_code_object_stream
{
  DISABLE_COPY_AND_ASSIGN (rocm_code_object_stream_file);

  rocm_code_object_stream_file (rocm_solib_fd_cache::cached_target_fd fd,
				ULONGEST offset, ULONGEST size);

  file_ptr read (bfd *abfd, void *buf, file_ptr size,
		 file_ptr offset) override;

  LONGEST size () override;

protected:
  /* The target file descriptor for this stream.  */
  rocm_solib_fd_cache::cached_target_fd m_fd;

  /* The offset of the ELF file image in the target file.  */
  ULONGEST m_offset;

  /* The size of the ELF file image.  The value 0 means that it was
     unspecified in the URI descriptor.  */
  ULONGEST m_size;
};

rocm_code_object_stream_file::rocm_code_object_stream_file
  (rocm_solib_fd_cache::cached_target_fd fd, ULONGEST offset, ULONGEST size)
  : m_fd (std::move (fd)), m_offset (offset), m_size (size)
{
}

file_ptr
rocm_code_object_stream_file::read (bfd *, void *buf, file_ptr size,
				    file_ptr offset)
{
  fileio_error target_errno;
  file_ptr nbytes = 0;
  while (size > 0)
    {
      QUIT;

      file_ptr bytes_read
	= target_fileio_pread (m_fd.fd (),
			       static_cast<gdb_byte *> (buf) + nbytes, size,
			       m_offset + offset + nbytes, &target_errno);

      if (bytes_read == 0)
	break;

      if (bytes_read < 0)
	{
	  errno = fileio_error_to_host (target_errno);
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
      fileio_error target_errno;
      struct stat stat;
      if (target_fileio_fstat (m_fd.fd (), &stat, &target_errno) < 0)
	{
	  errno = fileio_error_to_host (target_errno);
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

/* Interface to a code object which lives in the inferior's memory.  */

struct rocm_code_object_stream_memory final : public rocm_code_object_stream
{
  DISABLE_COPY_AND_ASSIGN (rocm_code_object_stream_memory);

  rocm_code_object_stream_memory (gdb::byte_vector buffer);

  file_ptr read (bfd *abfd, void *buf, file_ptr size,
		 file_ptr offset) override;

protected:

  /* Snapshot of the original ELF image taken during load.  This is done to
     support the situation where an inferior uses an in-memory image, and
     releases or reuses this memory before GDB is done using it.  */
  gdb::byte_vector m_objfile_image;

  LONGEST size () override
  {
    return m_objfile_image.size ();
  }
};

rocm_code_object_stream_memory::rocm_code_object_stream_memory
  (gdb::byte_vector buffer)
  : m_objfile_image (std::move (buffer))
{
}

file_ptr
rocm_code_object_stream_memory::read (bfd *, void *buf, file_ptr size,
				      file_ptr offset)
{
  if (size > m_objfile_image.size () - offset)
    size = m_objfile_image.size () - offset;

  memcpy (buf, m_objfile_image.data () + offset, size);
  return size;
}

} /* anonymous namespace */

gdb_bfd_iovec_base *
rocm_solib_ops::bfd_iovec_open (bfd *abfd, inferior *inferior)
{
  std::string_view uri (bfd_get_filename (abfd));
  std::string_view protocol_delim = "://";
  size_t protocol_end = uri.find (protocol_delim);
  std::string protocol (uri.substr (0, protocol_end));
  protocol_end += protocol_delim.length ();

  std::transform (protocol.begin (), protocol.end (), protocol.begin (),
		  [] (unsigned char c) { return c_tolower (c); });

  std::string_view path;
  size_t path_end = uri.find_first_of ("#?", protocol_end);
  if (path_end != std::string::npos)
    path = uri.substr (protocol_end, path_end++ - protocol_end);
  else
    path = uri.substr (protocol_end);

  /* %-decode the string.  */
  std::string decoded_path;
  decoded_path.reserve (path.length ());
  for (size_t i = 0; i < path.length (); ++i)
    if (path[i] == '%'
	&& i < path.length () - 2
	&& c_isxdigit (path[i + 1])
	&& c_isxdigit (path[i + 2]))
      {
	std::string_view hex_digits = path.substr (i + 1, 2);
	decoded_path += std::stoi (std::string (hex_digits), 0, 16);
	i += 2;
      }
    else
      decoded_path += path[i];

  /* Tokenize the query/fragment.  */
  std::vector<std::string_view> tokens;
  size_t pos, last = path_end;
  while ((pos = uri.find ('&', last)) != std::string::npos)
    {
      tokens.emplace_back (uri.substr (last, pos - last));
      last = pos + 1;
    }

  if (last != std::string::npos)
    tokens.emplace_back (uri.substr (last));

  /* Create a tag-value map from the tokenized query/fragment.  */
  gdb::unordered_map<std::string_view, std::string_view,
		     gdb::string_view_hash> params;
  for (std::string_view token : tokens)
    {
      size_t delim = token.find ('=');
      if (delim != std::string::npos)
	{
	  std::string_view tag = token.substr (0, delim);
	  std::string_view val = token.substr (delim + 1);
	  params.emplace (tag, val);
	}
    }

  try
    {
      ULONGEST offset = 0;
      ULONGEST size = 0;

      auto try_strtoulst = [] (std::string_view v)
	{
	  errno = 0;
	  ULONGEST value = strtoulst (v.data (), nullptr, 0);
	  if (errno != 0)
	    {
	      /* The actual message doesn't matter, the exception is caught
		 below, transformed in a BFD error, and the message is lost.  */
	      error (_("Failed to parse integer."));
	    }

	  return value;
	};

      auto offset_it = params.find ("offset");
      if (offset_it != params.end ())
	offset = try_strtoulst (offset_it->second);

      auto size_it = params.find ("size");
      if (size_it != params.end ())
	{
	  size = try_strtoulst (size_it->second);
	  if (size == 0)
	    error (_("Invalid size value"));
	}

      if (protocol == "file")
	{
	  /* A Windows absolute file path is encoded in a file: URI with a
	     leading "/" before the drive letter: "file:///C:/Users/foo/bar".
	     See grammar in RFC 8089 Section 2, the path-absolute production
	     requires the leading "/".  'path-absolute' is defined by RFC 3986
	     Section 3.3.  After decoding, decoded_path would be
	     "/C:/Users/foo/bar", which is not a valid Windows path.  Drop the
	     leading "/" as a normalization step.  */
	  if ((effective_target_file_system_kind ()
	       == file_system_kind_dos_based)
	      && decoded_path.length () >= 4
	      && decoded_path[0] == '/'
	      && c_isalpha (decoded_path[1])
	      && decoded_path[2] == ':'
	      && decoded_path[3] == '/')
	    decoded_path.erase (0, 1);

	  fileio_error target_errno;
	  rocm_solib_fd_cache::cached_target_fd fd
	    = m_fd_cache.open (decoded_path, &target_errno);

	  if (fd.fd () == target_fd::INVALID)
	    {
	      errno = fileio_error_to_host (target_errno);
	      bfd_set_error (bfd_error_system_call);
	      return nullptr;
	    }

	  return new rocm_code_object_stream_file (std::move (fd), offset,
						   size);
	}

      if (protocol == "memory")
	{
	  ULONGEST pid = try_strtoulst (path);
	  if (pid != inferior->pid)
	    {
	      warning (_("`%s': code object is from another inferior"),
		       std::string (uri).c_str ());
	      bfd_set_error (bfd_error_bad_value);
	      return nullptr;
	    }

	  gdb::byte_vector buffer (size);
	  if (target_read_memory (offset, buffer.data (), size) != 0)
	    {
	      warning (_("Failed to copy the code object from the inferior"));
	      bfd_set_error (bfd_error_bad_value);
	      return nullptr;
	    }

	  return new rocm_code_object_stream_memory (std::move (buffer));
	}

      warning (_("`%s': protocol not supported: %s"),
	       std::string (uri).c_str (), protocol.c_str ());
      bfd_set_error (bfd_error_bad_value);
      return nullptr;
    }
  catch (const gdb_exception_quit &ex)
    {
      set_quit_flag ();
      bfd_set_error (bfd_error_bad_value);
      return nullptr;
    }
  catch (const gdb_exception &ex)
    {
      bfd_set_error (bfd_error_bad_value);
      return nullptr;
    }
}

gdb_bfd_ref_ptr
rocm_solib_ops::bfd_open (const char *pathname)
{
  /* Handle regular files with SVR4 open.  */
  if (strstr (pathname, "://") == nullptr)
    return m_host_ops->bfd_open (pathname);

  auto open = [this] (bfd *nbfd)
  {
    return this->bfd_iovec_open (nbfd, current_inferior ());
  };

  gdb_bfd_ref_ptr abfd = gdb_bfd_openr_iovec (pathname, "elf64-amdgcn", open);

  if (abfd == nullptr)
    error (_("Could not open `%s' as an executable file: %s"), pathname,
	   bfd_errmsg (bfd_get_error ()));

  /* Check bfd format.  */
  if (!gdb_bfd_check_format (abfd.get (), bfd_object))
    error (_("`%s': not in executable format: %s"),
	   bfd_get_filename (abfd.get ()), bfd_errmsg (bfd_get_error ()));

  unsigned char osabi = elf_elfheader (abfd)->e_ident[EI_OSABI];
  unsigned char osabiversion = elf_elfheader (abfd)->e_ident[EI_ABIVERSION];

  /* Check that the code object is using the HSA OS ABI.  */
  if (osabi != ELFOSABI_AMDGPU_HSA)
    error (_("`%s': ELF file OS ABI is not supported (%d)."),
	   bfd_get_filename (abfd.get ()), osabi);

  /* We support HSA code objects V3 and greater.  */
  if (osabiversion < ELFABIVERSION_AMDGPU_HSA_V3)
    error (_("`%s': ELF file HSA OS ABI version is not supported (%d)."),
	   bfd_get_filename (abfd.get ()), osabiversion);

  /* For GDB to be able to use this solib, the exact AMDGPU processor type
     must be supported by both BFD and the amd-dbgapi library.  */
  const unsigned char gfx_arch
    = elf_elfheader (abfd)->e_flags & EF_AMDGPU_MACH ;
  const bfd_arch_info_type *bfd_arch_info
    = bfd_lookup_arch (bfd_arch_amdgcn, gfx_arch);

  amd_dbgapi_architecture_id_t architecture_id;
  amd_dbgapi_status_t dbgapi_query_arch
    = amd_dbgapi_get_architecture (gfx_arch, &architecture_id);

  if (dbgapi_query_arch != AMD_DBGAPI_STATUS_SUCCESS
      || bfd_arch_info ==  nullptr)
    {
      if (dbgapi_query_arch != AMD_DBGAPI_STATUS_SUCCESS
	  && bfd_arch_info ==  nullptr)
	{
	  /* Neither of the libraries knows about this arch, so we cannot
	     provide a human readable name for it.  */
	  error (_("'%s': AMDGCN architecture %#02x is not supported."),
		 bfd_get_filename (abfd.get ()), gfx_arch);
	}
      else if (dbgapi_query_arch != AMD_DBGAPI_STATUS_SUCCESS)
	{
	  gdb_assert (bfd_arch_info != nullptr);
	  error (_("'%s': AMDGCN architecture %s not supported by "
		   "amd-dbgapi."),
		 bfd_get_filename (abfd.get ()),
		 bfd_arch_info->printable_name);
	}
      else
	{
	  gdb_assert (dbgapi_query_arch == AMD_DBGAPI_STATUS_SUCCESS);
	  char *arch_name;
	  if (amd_dbgapi_architecture_get_info
	      (architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_NAME,
	       sizeof (arch_name), &arch_name) != AMD_DBGAPI_STATUS_SUCCESS)
	    error (_("amd_dbgapi_architecture_get_info call failed for arch "
		     "%#02x."), gfx_arch);
	  gdb::unique_xmalloc_ptr<char> arch_name_cleaner (arch_name);

	  error (_("'%s': AMDGCN architecture %s not supported."),
		 bfd_get_filename (abfd.get ()),
		 arch_name);
	}
    }

  gdb_assert (is_amdgpu_arch (abfd.get ()));

  return abfd;
}

void
rocm_solib_ops::create_inferior_hook (int from_tty)
{
  m_solib_list.clear ();
  m_host_ops->create_inferior_hook (from_tty);
}

void
rocm_solib_ops::update_solib_list ()
{
  inferior *inf = current_inferior ();

  amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id (inf);
  if (process_id.handle == AMD_DBGAPI_PROCESS_NONE.handle)
    return;

  m_solib_list.clear ();

  amd_dbgapi_code_object_id_t *code_object_list;
  size_t count;

  amd_dbgapi_status_t status
    = amd_dbgapi_process_code_object_list (process_id, &count,
					   &code_object_list, nullptr);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    {
      warning (_("amd_dbgapi_process_code_object_list failed (%s)"),
	       get_status_string (status));
      return;
    }

  gdb::unique_xmalloc_ptr<amd_dbgapi_code_object_id_t> code_object_list_holder
    (code_object_list);

  for (size_t i = 0; i < count; ++i)
    {
      CORE_ADDR l_addr;
      char *uri_bytes;

      status = amd_dbgapi_code_object_get_info
	(code_object_list[i], AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS,
	 sizeof (l_addr), &l_addr);
      if (status != AMD_DBGAPI_STATUS_SUCCESS)
	continue;

      status = amd_dbgapi_code_object_get_info
	(code_object_list[i], AMD_DBGAPI_CODE_OBJECT_INFO_URI_NAME,
	 sizeof (uri_bytes), &uri_bytes);
      if (status != AMD_DBGAPI_STATUS_SUCCESS)
	continue;

      gdb::unique_xmalloc_ptr<char> uri_bytes_holder (uri_bytes);

      /* Pass a dummy debug base.  */
      lm_info_svr4_up li = std::make_unique<lm_info_svr4> (-1);
      li->l_addr = l_addr;

      /* Generate a unique name so that code objects with the same URI but
	 different load addresses are seen by gdb core as different shared
	 objects.  */
      std::string unique_name
	= string_printf ("code_object_%ld", code_object_list[i].handle);

      m_solib_list.emplace_back (uri_bytes, std::move (unique_name),
				 std::move (li));
    }
}

static void
rocm_solib_target_inferior_created (inferior *inf)
{
  /* A vfork child shares its pspace with its parent, do not touch anything.  */
  if (inf->vfork_parent != nullptr)
    return;

  auto prev_ops = inf->pspace->release_solib_ops ();
  auto rocm_ops = std::make_unique<rocm_solib_ops> (inf, std::move (prev_ops));
  inf->pspace->set_solib_ops (std::move (rocm_ops));

  gdb::checked_static_cast<rocm_solib_ops *>
    (inf->pspace->solib_ops ())->update_solib_list ();

  /* Force GDB to reload the solibs.  */
  inf->pspace->clear_solib_cache ();
  solib_add (nullptr, 0, auto_solib_add);
}

static void
rocm_solib_target_inferior_execd (inferior *exec_inf, inferior *follow_inf)
{
  /* Engage the ROCm so_ops, but only if dbgapi is attached to the inferior
     (avoiding remote inferiors and core file debugging).  */
  if (get_amd_dbgapi_process_id (follow_inf) == AMD_DBGAPI_PROCESS_NONE)
    return;

  auto pspace = follow_inf->pspace;
  auto prev_ops = pspace->release_solib_ops ();
  auto rocm_ops
    = std::make_unique<rocm_solib_ops> (follow_inf, std::move (prev_ops));
  pspace->set_solib_ops (std::move (rocm_ops));
}

static void
rocm_solib_target_inferior_forked (inferior *parent_inf, inferior *child_inf,
				   target_waitkind fork_kind,
				   bool detach_on_fork, bool follow_child)
{
  if (detach_on_fork && follow_child && fork_kind == TARGET_WAITKIND_FORKED)
    {
      /* In this particular configuration, infrun's follow_fork_inferior
	 function moves the parent pspace to the child directly.  Remove the
	 existing rocm_solib_ops from the child and restore the host solib_ops,
	 to make it look like a brand new pspace.

	 Remove solibs created by the rocm_solib_ops, since they contain back
	 references to the rocm_solib_ops (namely, to the fd cache).  */
      auto &solibs = child_inf->pspace->solibs ();
      auto rocm_ops
	= gdb::checked_static_cast<rocm_solib_ops *>
	    (child_inf->pspace->solib_ops ());

      for (auto solibs_it = solibs.begin (); solibs_it != solibs.end ();)
	{
	  if (&solibs_it->ops () != rocm_ops)
	    {
	      ++solibs_it;
	      continue;
	    }

	  solibs_it = remove_solib (child_inf->pspace, solibs_it);
	}

      auto rocm_ops_holder = child_inf->pspace->release_solib_ops ();
      child_inf->pspace->set_solib_ops (rocm_ops->release_host_ops ());
    }
}

INIT_GDB_FILE (rocm_solib)
{
  /* The dependency on the amd-dbgapi exists because solib-rocm's
     inferior_created observer needs amd-dbgapi to have attached the process,
     which happens in amd_dbgapi_target's inferior_created observer.  */
  gdb::observers::inferior_created.attach
    (rocm_solib_target_inferior_created,
     "solib-rocm",
     { &get_amd_dbgapi_target_inferior_created_observer_token () });

  gdb::observers::inferior_execd.attach
    (rocm_solib_target_inferior_execd, "solib-rocm",
     { &get_amd_dbgapi_target_inferior_execd_observer_token () });

  gdb::observers::inferior_forked.attach
    (rocm_solib_target_inferior_forked, "solib-rocm");
}
