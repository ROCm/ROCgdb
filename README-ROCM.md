ROCm Debugger (ROCgdb)
======================

This is ROCgdb, the ROCm source-level debugger for Linux, based on GDB, the GNU
source-level debugger. It includes support for heterogenous debugging on the
ROCm platform of an x86-based host architecture together with the AMD
commercially available GPU architectures supported by the AMD Debugger API which
is included with the ROCm release as the ROCdbgapi library.

Standard GDB commands can be used for both CPU and GPU code debugging. For more
information about ROCgdb, please refer to the *ROCgdb User Manual* which is
installed as both texinfo files and as a single PDF file in ``/opt/rocm/doc``.
In particular, refer to the *Debugging Heterogeneous Programs* chapter for
general information, and the *AMD GPU* subsection of the *Architectures* section
in the *Configuration-Specific Information* chapter. The *AMD GPU* subsection
also provides the list of features not currently implemented, and known current
restrictions.

For more information about ROCm, please refer to:

- https://github.com/RadeonOpenCompute/ROCm

For more information about GDB, please refer to the README file in this folder
or check the GDB home page at:

- http://www.gnu.org/software/gdb

Build the ROCm Debugger
-----------------------

ROCgdb can be built on Ubuntu 16.04, Ubuntu 18.04, and Centos 7.6.

Building ROCgdb has the following prerequisites:

1. A C++11 compiler such as GCC 4.8 or Clang 3.3.

2. AMD Debugger API Library (ROCdbgapi) which can be installed as part of the
   ROCm release by the ``rocm-dbgapi`` package.

3. For Ubuntu 16.04 and Ubuntu 18.04 the following adds the needed packages:

   ````shell
   apt install bison flex gcc make ncurses-dev texinfo g++ \
     zlib1g-dev libexpat-dev libpython2.7-dev python2.7-minimal liblzma-dev \
     libbabeltrace-dev libbabeltrace-ctf-dev
   ````

4. For Centos 7.6 the following adds the needed packages:

   ````shell
   yum install -y epel-release centos-release-scl
   yum install -y bison flex gcc make texinfo gcc-c++ \
     zlib-devel expat-devel python-devel xz-devel \
     libbabeltrace-devel ncurses-devel
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
  --without-guile --with-babeltrace --with-lzma --with-python
make
````

Specify ``--with-rocm-dbgapi=PATH`` if the the AMD Debugger API Library
(ROCdbgapi) is not installed in its default location. The ``configure`` script
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
be installed. These can be installed as part of the ROCm release by the
``rocm-dbgapi`` package:

- ``librocm-dbgapi.so.0``
- ``libamd_comgr.so.1``

The PDF *User Manual* can be generated with:

````shell
make pdf
````

The generated PDF will be placed in:

- ``build/gdb/doc/gdb.pdf``
