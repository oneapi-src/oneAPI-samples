CMake + Intel&reg; oneAPI Examples
==================================

The CMake projects in this code repository demonstrate use of CMake to build
simple programs with CMake for various common scenarios. Examples are divided
-into directories for C, C++, and Fortran. SYCL examples are included at the
top level. SYCL can be implemented entirely in C++, but people looking for SYCL
examples are more likely to beless interested in pure C, C++, or Fortran examples.

Each language specific directory are contains at least a simple "Hello, World"
example, and OpenMP example.

The examples in this directory are structured as a collection of independent
projects, rather as a large single project. This way, any example can be copied
into a separate subdirectory, compiled, run, and used as the basis for a new project.

The top level CMakeLists.txt includes projects in all the sub-directories.  To
build all of the examples, create a build directory and generate the project as
usual.  For example in Linux,

    $ mkdir build
    $ cd build
    $ CC=icx CXX=icpx FC=ifx cmake ..
    $ cmake --build . -j


CMake Minimum Required Version
------------------------------

CMake version 3.20.0 first added support for the Intel oneAPI C, C++, and
Fortran compilers.  Since 3.20.0 a improvements have been made to the initial
support in CMake.  The latest released CMake is likely to work best with
oneAPI compilers.

Older versions of may also be suitable depending on a project's needs.  The following
table summarizes which kinds of tasks work in each version of CMake.

