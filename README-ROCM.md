AMD ROCm Debugger (ROCgdb)
==========================

Introduction
------------

The AMD ROCm Debugger (ROCgdb) is the AMD source-level debugger for Linux and
Windows, based on the GNU Debugger (GDB).  It enables heterogeneous debugging on
the AMD ROCm platform comprising of an x86-based host architecture along with
commercially available AMD GPU architectures supported by the AMD Debugger API
Library (ROCdbgapi).  The AMD Debugger API Library (ROCdbgapi) is included with
the AMD ROCm release.

For more information about AMD ROCm, see:

- https://docs.amd.com/

You can use the standard GDB commands for both CPU and GPU code debugging.  For
more information about ROCgdb, refer to the *ROCgdb User Guide* which is
installed under the main ROCm installation directory on Linux (typically
``/opt/rocm/``) or HIP SDK installation directory on Windows (typically
``C:/Program Files/AMD/ROCm/``), in the following locations:

- ``.../share/info/rocgdb/gdb.info`` as a texinfo file (Linux only)
- ``.../share/doc/rocgdb/rocgdb.pdf`` as a PDF file
- ``.../share/html/rocgdb/index.html`` as an HTML file

You can refer to the following chapters in the *ROCgdb User Guide* for more
specific information about debugging heterogeneous programs on AMD ROCm:

- *Debugging Heterogeneous Programs* provides general information about
  debugging heterogeneous programs.  It presents features and commands that are
  not currently implemented but provisionally planned for future versions.
- *Configuration-Specific Information > Architectures > AMD GPU* provides
  specific information about debugging heterogeneous programs on AMD ROCm with
  supported AMD GPU chips.  This section also lists the implementation status
  and known issues of the current version.

For more information about the GNU Debugger (GDB), refer to the ``README`` file
in this folder or check the GNU Debugger (GDB) web site at:

- http://www.gnu.org/software/gdb

Build the AMD ROCm Debugger
---------------------------

ROCgdb can be built on Linux for Ubuntu 20.04, Ubuntu 22.04, Centos 8.1, RHEL
8.1, RHEL 9.1, and SLES 15 Service Pack 1.  ROCgdb can be built on Windows 11
using an MSYS2 environment.

Building ROCgdb has the following prerequisites:

1. A C++17 compiler such as GCC 9 or Clang 5.

2. AMD Debugger API Library (ROCdbgapi).  On Linux, this can be installed as
   part of the AMD ROCm release by the ``rocm-dbgapi`` package.  On Windows, it
   must be built from source.

3. For Ubuntu 20.04 and Ubuntu 22.04 the following adds the needed packages:

   ````shell
   apt install bison flex gcc make ncurses-dev texinfo g++ zlib1g-dev \
     libexpat-dev python3-dev liblzma-dev libgmp-dev libmpfr-dev
   ````

4. For CentOS 8.1, RHEL 8.1 and RHEL 9.1 the following adds the needed
   packages:

   ````shell
   yum install -y epel-release centos-release-scl bison flex gcc make \
     texinfo texinfo-tex gcc-c++ zlib-devel expat-devel python3-devel \
     xz-devel gmp-devel mpfr-devel ncurses-devel
   ````

5. For SLES 15 Service Pack 1 the following adds the needed packages:

   ````shell
   zypper in bison flex gcc make texinfo gcc-c++ zlib-devel libexpat-devel \
     python3-devel xz-devel gmp-devel mpfr-devel ncurses-devel
   ````

6. For Windows under MSYS2 the following adds the needed packages:

   ````shell
   pacman -S bison flex make \
     mingw-w64-ucrt-x86_64-gcc \
     mingw-w64-ucrt-x86_64-gmp \
     mingw-w64-ucrt-x86_64-mpfr \
     mingw-w64-ucrt-x86_64-gettext-runtime \
     mingw-w64-ucrt-x86_64-expat \
     mingw-w64-ucrt-x86_64-libiconv \
     mingw-w64-ucrt-x86_64-ncurses \
     mingw-w64-ucrt-x86_64-python \
     mingw-w64-ucrt-x86_64-xxhash \
     mingw-w64-ucrt-x86_64-xz \
     mingw-w64-ucrt-x86_64-zlib \
     mingw-w64-ucrt-x86_64-zstd
     texinfo texinfo-tex \
     mingw-w64-x86_64-texlive-plain-generic \
     mingw-w64-x86_64-texlive-latex-recommended
   ````

An example command-line to build ROCgdb on Linux is:

````shell
cd rocgdb
mkdir build
cd build
../configure --program-prefix=roc \
  --enable-64-bit-bfd --enable-targets="x86_64-linux-gnu,amdgcn-amd-amdhsa" \
  --disable-ld --disable-gas --disable-gdbserver --disable-sim --enable-tui \
  --disable-gdbtk --disable-gprofng --disable-shared --with-expat \
  --with-system-zlib --without-guile --without-babeltrace --with-lzma \
  --with-python=python3
make
````

An example command-line to build ROCgdb on Windows under MSYS2 is:

````shell
cd rocgdb
mkdir build
cd build
../configure --program-prefix=roc \
  --host=x86_64-w64-mingw32 \
  --target=x86_64-w64-mingw32 \
  --enable-64-bit-bfd --enable-targets="amdgcn-amd-amdhsa" \
  --disable-binutils --disable-gas --disable-gdbserver --disable-gold \
  --disable-gprof --disable-gprofng --disable-ld --disable-libctf \
  --disable-sim --disable-gdbtk --disable-shared \
  --disable-source-highlight --with-system-zlib \
  --without-guile --with-python=python3 \
  --without-babeltrace --with-lzma --enable-tui \
  CFLAGS="-D_WIN32_WINNT=0xA00" \
  CXXFLAGS="-D_WIN32_WINNT=0xA00"
make
````

If the AMD Debugger API Library (ROCdbgapi) is not installed in the system
default location, specify ``PKG_CONFIG_PATH`` so ``pkg-config`` can gather the
correct build configuration.  For example, if ROCdbgapi is installed in
``/opt/rocm-$ROCM_VERSION`` (the default for ROCm packages on Linux), use
``PKG_CONFIG_PATH=/opt/rocm-$ROCM_VERSION/share/pkgconfig``.

If the system's dynamic linker is not configured to locate ROCdbgapi where it is
installed, ROCgdb can be configured and built using
``LDFLAGS="-Wl,-rpath=/opt/rocm-$ROCM_VERSION/lib"``.  Alternatively,
``LD_LIBRARY_PATH`` can be used at runtime to indicate where ROCdbgapi is
installed.

The built ROCgdb executable will be placed in:

- ``build/gdb/gdb``

The texinfo *User Manual* will be placed in:

- ``build/gdb/doc/gdb.info``

To install ROCgdb:

````shell
make install
````

The installed ROCgdb will be placed in:

- ``<prefix>/bin/rocgdb``

To execute ROCgdb, the ROCdbgapi library and its dependent Comgr library must
be available.

On Linux, these can be installed as part of the AMD ROCm release by the
``rocm-dbgapi`` package:

- ``librocm-dbgapi.so``
- ``libamd_comgr.so``

On Windows, Comgr is installed as part of the HIP SDK release:

- ``C:\Windows\System32\amd_comgr_*.dll``

ROCdbgapi must be built from source; it is not available as a pre-installed DLL,
because the ROCgdb included in the HIP SDK installation is statically linked
against ROCdbgapi.

The PDF *User Manual* can be generated with:

````shell
make pdf
````

The generated PDF will be placed in:

- ``build/gdb/doc/gdb.pdf``

Disclaimer
----------

The information contained herein is for informational purposes only and is
subject to change without notice.  While every precaution has been taken in the
preparation of this document, it may contain technical inaccuracies, omissions
and typographical errors, and AMD is under no obligation to update or otherwise
correct this information.  Advanced Micro Devices, Inc. makes no
representations or warranties with respect to the accuracy or completeness of
the contents of this document, and assumes no liability of any kind, including
the implied warranties of noninfringement, merchantability or fitness for
particular purposes, with respect to the operation or use of AMD hardware,
software or other products described herein.  No license, including implied or
arising by estoppel, to any intellectual property rights is granted by this
document.  Terms and limitations applicable to the purchase or use of AMD’s
products are as set forth in a signed agreement between the parties or in AMD’s
Standard Terms and Conditions of Sale.

AMD®, the AMD Arrow logo, ROCm® and combinations thereof are trademarks of
Advanced Micro Devices, Inc.  Linux® is the registered trademark of Linus
Torvalds in the U.S. and other countries.  RedHat® and the Shadowman logo are
registered trademarks of Red Hat, Inc. www.redhat.com in the U.S. and other
countries.  SUSE® is a registered trademark of SUSE LLC in the United Stated
and other countries.  Ubuntu® and the Ubuntu logo are registered trademarks of
Canonical Ltd.  Other product names used in this publication are for
identification purposes only and may be trademarks of their respective
companies.

Copyright (c) 2019-2025 Advanced Micro Devices, Inc.  All rights reserved.
