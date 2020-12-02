AMD ROCm Debugger (ROCgdb)
==========================

Introduction
------------

The AMD ROCm Debugger (ROCgdb) is the AMD source-level debugger for Linux,
based on the GNU Debugger (GDB).  It enables heterogeneous debugging on the AMD
ROCm platform comprising of an x86-based host architecture along with
commercially available AMD GPU architectures supported by the AMD Debugger API
Library (ROCdbgapi).  The AMD Debugger API Library (ROCdbgapi) is included with
the AMD ROCm release.

The current AMD ROCm Debugger (ROCgdb) is an initial prototype that focuses on
source line debugging.  Symbolic variable debugging capabilities are not
currently supported.

For more information about AMD ROCm, see:

- https://rocmdocs.amd.com/

You can use the standard GDB commands for both CPU and GPU code debugging.  For
more information about ROCgdb, refer to the *ROCgdb User Guide* which is
installed at:

- ``/opt/rocm/share/info/gdb.info`` as a texinfo file
- ``/opt/rocm/share/doc/gdb/gdb.pdf`` as a PDF file

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

ROCgdb can be built on Ubuntu 18.04, Ubuntu 20.04, Centos 8.1, RHEL 8.1, and SLES
15 Service Pack 1.

Building ROCgdb has the following prerequisites:

1. A C++11 compiler such as GCC 4.8 or Clang 3.3.

2. AMD Debugger API Library (ROCdbgapi) which can be installed as part of the
   AMD ROCm release by the ``rocm-dbgapi`` package.

3. For Ubuntu 18.04 and Ubuntu 20.04 the following adds the needed packages:

   ````shell
   apt install bison flex gcc make ncurses-dev texinfo g++ zlib1g-dev \
     libexpat-dev python3-dev liblzma-dev libbabeltrace-dev \
     libbabeltrace-ctf-dev
   ````

4. For CentOS 8.1 and RHEL 8.1 the following adds the needed packages:

   ````shell
   yum install -y epel-release centos-release-scl bison flex gcc make \
     texinfo texinfo-tex gcc-c++ zlib-devel expat-devel python3-devel \
     xz-devel libbabeltrace-devel ncurses-devel
   wget http://repo.okay.com.mx/centos/8/x86_64/release/libbabeltrace-devel-1.5.4-2.el8.x86_64.rpm \
   && rpm -ivh --nodeps libbabeltrace-devel-1.5.4-2.el8.x86_64.rpm
   ````

5. For SLES 15 Service Pack 1 the following adds the needed packages:

   ````shell
   zypper in bison flex gcc make texinfo gcc-c++ zlib-devel libexpat-devel \
     python3-devel xz-devel babeltrace-devel ncurses-devel
   ````

An example command-line to build ROCgdb on Linux is:

````shell
cd rocgdb
mkdir build
cd build
../configure --program-prefix=roc \
  --enable-64-bit-bfd --enable-targets="x86_64-linux-gnu,amdgcn-amd-amdhsa" \
  --disable-ld --disable-gas --disable-gdbserver --disable-sim --enable-tui \
  --disable-gdbtk --disable-shared --with-expat --with-system-zlib \
  --without-guile --with-babeltrace --with-lzma --with-python=python3
make
````

Specify ``--with-rocm-dbgapi=PATH`` if the the AMD Debugger API Library
(ROCdbgapi) is not installed in its default location.  The ``configure`` script
looks in ``PATH/include`` and ``PATH/lib``. The default value for ``PATH`` is
``/opt/rocm``.

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

To execute ROCgdb, the ROCdbgapi library and its dependent ROCcomgr library must
be installed.  These can be installed as part of the AMD ROCm release by the
``rocm-dbgapi`` package:

- ``librocm-dbgapi.so.0``
- ``libamd_comgr.so.1``

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

Copyright (c) 2019-2021 Advanced Micro Devices, Inc.  All rights reserved.
