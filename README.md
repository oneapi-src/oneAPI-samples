# Data Parallel C++ Book Source Samples

This repository accompanies [*Data Parallel C++: Mastering DPC++ for Programming of Heterogeneous Systems using C++ and SYCL*](https://www.apress.com/9781484255735) by James Reinders, Ben Ashbaugh, James Brodman, Michael Kinsner, John Pennycook, Xinmin Tian (Apress, 2020).

[comment]: #cover
![Cover image](9781484255735.jpg)

Many of the samples in the book are snips from the more complete files in this repository.  The full files contain supporting code, such as header inclusions, which are not shown in every listing within the book.  The complete listings are intended to compile and be modifiable for experimentation.

Download the files as a zip using the green button, or clone the repository to your machine using Git.

## How to Build the Samples

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

    NOTE: If you do not have oneDPL installed, you can disable compilation of those tests with the option `NODPL`

    ```sh
    cmake -G Ninja -DNODPL=1 -DCMAKE_TOOLCHAIN_FILE=../dpcpp_toolchain.cmake ..
    ```

3. Build with the generated build files:

    ```sh
    ninja install
    ```
