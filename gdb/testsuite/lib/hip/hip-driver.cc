/* Kernel entry point for testing against AMDGCN devices, using HIP.

   Copyright (C) 2021 Free Software Foundation, Inc.
   Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime.h>

/* Pull in the gdb_hip_test_main declarations.  */
#include "hip-test-main.h"

#include <unistd.h>
#include <cstddef>
#include <stdlib.h>

/* The device's malloc buffer, and the pointer to the current position
   where we'll return memory from.  We allocate this buffer from host
   code instead of using static __device__ buffer because we don't
   want any symbol attached to the buffer, as some testcases print
   heap addresses not expecting to find any symbol.  One example is
   gdb.base/find.exp.  */
static __device__ unsigned char *malloc_buf;
static __device__ unsigned char *malloc_head;

/* The size of the device malloc buffer, in bytes.  */
static constexpr size_t malloc_buf_size = 0x10000;

/* For getenv.  */
static __device__ char **d_environ;

/* The kernel's exit code.  */
static __device__ int *exit_code;

/* The test kernel's entry point.  Call the testcase's main, which has
   been renamed to "gdb_hip_test_main" by the hip-test-wrapper.cc
   wrapper.  */

static __global__ void
kernel (unsigned char *malloc_buffer,
	int *res, int argc, char **argv, char **envp)
{
  /* Store global pointers.  */
  malloc_buf = malloc_buffer;
  malloc_head = malloc_buf;
  d_environ = envp;
  exit_code = res;

  /* See comments in hip-test-main.h.  gdb_hip_test_main is weak -- we
     call the version that is defined by the testcase.  */
  if (auto &m = static_cast<int (&)(int, char **, char **)>(gdb_hip_test_main))
    *res = m (argc, argv, envp);

  else if (auto &m = static_cast<int (&)(int, char **)>(gdb_hip_test_main))
    *res = m (argc, argv);

  else if (auto &m = static_cast<int (&)(int, const char **)>(gdb_hip_test_main))
    *res = m (argc, (const char **) argv);

  else if (auto &m = static_cast<int (&)(int, char *const *)>(gdb_hip_test_main))
    *res = m (argc, argv);

  else if (auto &m = static_cast<int (&)()>(gdb_hip_test_main))
    *res = m ();

  else
    {
      printf ("hip-driver.cc: error: gdb_hip_test_main not defined!\n");
      *res = 1;
    }
}

#define CHECK(cmd)					\
  do							\
    {							\
      hipError_t error = cmd;				\
      if (error != hipSuccess)				\
	{						\
	  fprintf(stderr, "error: '%s'(%d) at %s:%d\n",	\
		  hipGetErrorString (error), error,	\
		  __FILE__, __LINE__);			\
	  exit (EXIT_FAILURE);				\
	}						\
    } while (0)

/* Clone an array of pointers in ARGV style to the device.  Return the
   pointer to the device array.  */

static char **
copy_argv_device (int argc, char **argv)
{
  char **argv_d;
  CHECK (hipMalloc (&argv_d, sizeof (char *) * argc));

  for (int i = 0; i < argc; i++)
    {
      char *arg_d;
      if (argv[i] != nullptr)
	{
	  size_t size = strlen (argv[i]) + 1;
	  CHECK (hipMalloc (&arg_d, size));
	  CHECK (hipMemcpy (arg_d, argv[i], size, hipMemcpyHostToDevice));
	}
      else
	arg_d = nullptr;
      CHECK (hipMemcpy (&argv_d[i], &arg_d, sizeof (char *), hipMemcpyHostToDevice));
    }
  return argv_d;
}

int
main (int argc, char **argv, char **envp)
{
  int deviceId;
  CHECK (hipGetDevice (&deviceId));
  hipDeviceProp_t props;
  CHECK (hipGetDeviceProperties (&props, deviceId));
  printf("info: running on device #%d %s\n", deviceId, props.name);

  printf("info: copy Host2Device\n");

  unsigned char *malloc_buf_d;
  CHECK (hipMalloc (&malloc_buf_d, 0x10000));

  int exitcode_h;
  int *exitcode_d;
  CHECK (hipMalloc (&exitcode_d, sizeof (int)));

  char **argv_d = copy_argv_device (argc, argv);

  /* Count ENVP entries.  */
  int envc;
  for (envc = 0; envp[envc] != nullptr; envc++)
    ;

  /* One extra for the NULL terminator.  */
  char **envp_d = copy_argv_device (envc + 1, envp);

  printf ("info: launch kernel\n");
  const unsigned blocks = 1;
  const unsigned threadsPerBlock = 1;

  hipLaunchKernelGGL (kernel, dim3 (blocks), dim3 (threadsPerBlock), 0, 0,
		      malloc_buf_d, exitcode_d, argc, argv_d, envp_d);

  printf("info: copy Device2Host\n");
  CHECK (hipMemcpy (&exitcode_h, exitcode_d, sizeof (int), hipMemcpyDeviceToHost));

  /* Wait until kernel finishes.  */
  hipDeviceSynchronize ();

  return exitcode_h;
}

/* Implement some standard functions used by tests, that are missing
   on the device.  */

int __device__
atoi (const char *nptr)
{
  int res = 0;
  for (int i = 0; nptr[i] != '\0'; i++)
    res = res * 10 + nptr[i] - '0';

  return res;
}

int __device__
puts (const char *s)
{
  printf ("%s", s);
  return 0;
}

static __device__ uint64_t
align_up (uint64_t v, int n)
{
  return (v + n - 1) & -n;
}

/* A very simple and stupid malloc implementation that is not thread
   safe and never frees any memory.  This is sufficient for our needs
   since we only run single-threaded programs.  */

void *__device__
malloc (size_t size)
{
  size = align_up (size, alignof (std::max_align_t));

  assert (malloc_head + size > malloc_head
	  && malloc_head + size < malloc_buf + malloc_buf_size);

  void *ret = malloc_head;
  malloc_head += size;
  return ret;
}

void *__device__
free (void *ptr)
{
  return 0;
}

void * __device__
calloc (size_t nmemb, size_t size)
{
  void *res = malloc (nmemb * size);
  return memset (res, 0, nmemb * size);
}

size_t __device__
strlen (const char *s)
{
  const char *end = s;
  while (*end != '\0')
    end++;
  return end - s;
}

char * __device__
strdup (const char *s)
{
  size_t size = strlen (s) + 1;
  char *ns = (char *) malloc (size);
  memcpy (ns, s, size);
  return ns;
}

char * __device__
strcpy (char *dest, const char *src)
{
  size_t size = strlen (src) + 1;
  memcpy (dest, src, size);
  return dest;
}

int __device__
strncmp (const char *s1, const char *s2, size_t n)
{
  /* Need to do the final subtraction in unsigned.  */
  auto *us1 = (unsigned char *) s1;
  auto *us2 = (unsigned char *) s2;

  size_t i = 0;
  for (;;)
    {
      if (i == n)
	return 0;
      if (us1[i] == '\0' || us1[i] != us2[i])
	return us1[i] - us2[i];
      i++;
    }
}

void __device__
exit (int code)
{
  __threadfence ();

  /* Store the exit code in the global.  The host side uses this as
     ultimate return code.  */
  *exit_code = code;

  /* End the kernel program.  */
  asm ("s_endpgm");

  /* This is an __attribute__((noreturn)) function.  Need this other
     Clang warns.  */
  __builtin_unreachable () ;
}

/* CLOCKS_PER_SEC is far from real.  Multiply by a factor that gives a
   close enough result in practice.  Tested with a simple program that
   calls sleep(10), and then doing "time ./a.out" until a close enough
   number came out.  This will obviously need tuning on different
   systems...  */
static constexpr uint64_t clocks_factor = 2000;

static void __device__
sleep_clocks (uint64_t clocks)
{
  uint64_t start = clock64 ();
  for (;;)
    {
      uint64_t now = clock64 ();
      uint64_t elapsed = (now > start
			  ? now - start
			  : now + (0xffffffffffffffff - start));
      if (elapsed >= clocks)
	return;
    }
}

extern "C" int __device__
usleep (useconds_t usec)
{
  uint64_t clocks = clocks_factor * CLOCKS_PER_SEC * usec / 1000000;

  sleep_clocks (clocks);
  return 0;
}

extern "C" unsigned int __device__
sleep (unsigned int seconds)
{
  uint64_t clocks = clocks_factor * CLOCKS_PER_SEC * seconds;

  sleep_clocks (clocks);
  return 0;
}

char * __device__
getenv (const char *name)
{
  size_t len = strlen (name);
  for (char **p = d_environ; *p != nullptr; ++p)
    if (strncmp (*p, name, len) == 0)
      {
	char *c = *p + len;
	if (*c == '=')
	  return c + 1;
      }
  return nullptr;
}
