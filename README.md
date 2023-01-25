# Data Parallel C++ Book Source Samples

This repository accompanies [*Data Parallel C++: Mastering DPC++ for Programming of Heterogeneous Systems using C++ and SYCL*](https://www.apress.com/9781484255735) by James Reinders, Ben Ashbaugh, James Brodman, Michael Kinsner, John Pennycook, Xinmin Tian (Apress, 2020).

[comment]: #cover
![Cover image](9781484255735.jpg)

## Purpose of this branch (main)

This branch (main) contains source code derived from what was published with the First Edition of the DPC++ book, but modified to be compatible with the SYCL 2020 specification which was released by The Khronos Group after book publication.  The original book source was primarily based on the older SYCL 1.2.1 specification, and many enhancements and changes were added in the later SYCL 2020 specification.  More importantly, most current toolchains which support SYCL are based on SYCL 2020, so this main branch is intended to be compatible with recent compiler and toolchain releases.

The Second Edition of the DPC++ book, likely to release in 2023, will be based on the updated code examples in this main branch.

## Overview

Many of the samples in the book are snips from the more complete files in this repository.  The full files contain supporting code, such as header inclusions, which are not shown in every listing within the book.  The complete listings are intended to compile and be modifiable for experimentation.

> :warning: Samples in this repository are updated to align with the most recent changes to the language and
toolchains, and are more current than captured in the book text due to lag between finalization and actual
publication of a print book.  If experimenting with the code samples, start with the versions in this
repository.  DPC++ and SYCL are evolving to be more powerful and easier to use, and updates to the sample code
in this repository are a good sign of forward progress!

Download the files as a zip using the green button, or clone the repository to your machine using Git.

## How to Build the Samples

> :warning: The samples in this repository are intended to compile with the open source project toolchain linked below, or
with the 2023.0 release or newer of the DPC++ toolchain.  If you have an older toolchain installed, you may encounter
compilation errors due to evolution of the features and extensions.

To build and use these examples, you will need an installed DPC++ toolchain.  For one such toolchain, please visit:

https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html

Alternatively, much of the toolchain can be built directly from:

https://github.com/intel/llvm

Some of the Chapter 18 examples require an installation of oneDPL, which is available from:

https://github.com/oneapi-src/oneDPL


To build the samples:

1. Setup oneAPI environment variables:

    On Windows:

    ```sh
    \path\to\inteloneapi\setvars.bat
    ```

    On Linux:

    ```sh
    source /path/to/inteloneapi/setvars.sh
    ```

2. Create build files using CMake, specifying the DPC++ toolchain.  For example:

    ```sh
    mkdir build && cd build
    cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=../dpcpp_toolchain.cmake ..
    ```

    NOTE: If you do not have Ninja installed, you can use another Makefile generator such as 'Unix Makefiles'.
    
    NOTE: If you do not have oneDPL installed, you can disable compilation of those tests with the option `NODPL`

    ```sh
    cmake -G Ninja -DNODPL=1 -DCMAKE_TOOLCHAIN_FILE=../dpcpp_toolchain.cmake ..
    ```

3. Build with the generated build files:

    ```sh
    ninja install
    ```
