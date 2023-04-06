# XeHE - Intel GPU-accelerated Privacy Protecting Computing project


# Project structure
`cmake/`   - cmake functionality (such as findSYCL, etc.)

`doc/` - documentation

`external/` - directory where third-party codes live (various licenses and copyrights), NTT, SEAL, Catch2, cuFHE, cuHE, etc.

`src/` - our code

`tests/` - our tests, using our own code and/or external libraries

`tools/` - miscellaneous tools, ours or not

`examples/` - playground to try various things

## External source codes
For testing and comparison only:

`SEAL` - commit f70b4dc6ab57ff743fa63177f8a9798913a27c68 (master, tag: v3.6.4). Binaries for examples and tests will be in the `build/external/SEAL/lib/` directory. By default, SEAL tests and examples are build. This is controlled by options in SEAL/CMakeLists.txt `SEAL_BUILD_EXAMPLES` and `SEAL_BUILD_TESTS`.
This version includes 7 examples and many-many tests.

`Catch2` - commit d399a308d (master)

`HEAAN` - commit 48a1ed0d31708e8e45d9423b698968081adabccd (master): rebuild with CMake for compatibility

### Dependencies

#### package dependencies
`NTL` - needed for HEAAN, can be installed with the latest code from `https://www.shoup.net/ntl/`. This library will in turn require GMP package, on Ubuntu 18.04.4 LTS, this can be installed via `sudo apt install libgmp-dev`.

#### toolchain dependencies
install oneAPI - the best way through apt (importing the keys, intel sources, etc). This way the maintaince and later updating will be easier.
oneAPI also needs a reasonably fresh GFX driver (if targetting GPU with DPC++).

#### build software dependencies
`cmake` also needs to be reasonably fresh (3.13+). If not, try to add Kitware sources to apt and update. If you don't have sudo rights, try a variant of
```bash
sudo apt remove cmake && pip install cmake --upgrade
sudo cp /$HOME/$USER/miniconda3/cmake /usr/bin/
```
#### WINDOWS
MSVC19
latest cmake

# Build 

## Linux
Activate oneAPI environment in linux with:
```bash
source /opt/intel/oneapi/setvars.sh
```

This builds all binaries:
```bash
mkdir -p build
cd build
cmake .. && make -j
```
To enable building with DPC++ for GPUs or other accelerators, run instead:
```bash
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DBUILD_WITH_IGPU=ON ..
make -j
```
### Available build options
Default options are:
```cmake
OPTION(BUILD_WITH_SEAL "Building with MS SEAL" ON)
OPTION(BUILD_WITH_HEAAN "Building with HEAAN (Needs NTL and GMP)" ON)
OPTION(BUILD_WITH_IGPU "Building with DPC++ SYCL" OFF)
OPTION(SEAL_USE_INTEL_GPU "Building XeHE as SEAL Backend" OFF)
OPTION(SEAL_USE_INTEL_XEHE "Use Intel XeHE library" ON)
OPTION(VERBOSE_TEST_FLAG "Building with verbose tests (gpu)" OFF)
OPTION(GPU_DEFAULT "Ciphertext is processed on gpu." OFF)
OPTION(SEALTEST_OMP_ENABLED "Building with enabled OMP test" OFF)
```

HEAAN is not built by default. To switch it `on`, turn `cmake -DBUILD_WITH_HEAAN=ON ..`.

Note that, at this time, SEAL targets will be placed in `external/SEAL/native/lib`.

Notes on **GPU_DEFAULT** flag.
The flag is meaningful only when SEAL_USE_INTEL_GPU is ON.
When **GPU_DEFAULT=ON**: Every Ciphertext object participated in CKKS HE ops is processed on GPU *by default*.
When **GPU_DEFAULT=OFF**: To make any Ciphertext object to be processed on gpu it *has to be touched* with ct.gpu() call.


Current build line with GPU back-end:

### Release version
```bash
CXXFLAGS=-isystem\ /opt/intel/oneapi/compiler/2021.3.0/linux/compiler/include/ cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DOpenMP_C_FLAGS="-qopenmp" -DOpenMP_C_LIB_NAMES="libiomp5" -DOpenMP_CXX_FLAGS="-qopenmp" -DOpenMP_CXX_LIB_NAMES="libiomp5" -DOpenMP_libiomp5_LIBRARY=/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin/libiomp5.so -DBUILD_WITH_IGPU=ON -DSEAL_USE_INTEL_GPU=ON -DSEAL_BUILD_TESTS=ON -DSEAL_BUILD_EXAMPLES=ON -DGPU_DEFAULT=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_HEAAN=OFF -DSEAL_USE_INTEL_LATTICE=OFF -DCMAKE_BUILD_TYPE=Release ..
```

### Debug version
```bash
CXXFLAGS=-isystem\ /opt/intel/oneapi/compiler/2021.3.0/linux/compiler/include/ cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DOpenMP_C_FLAGS="-qopenmp" -DOpenMP_C_LIB_NAMES="libiomp5" -DOpenMP_CXX_FLAGS="-qopenmp" -DOpenMP_CXX_LIB_NAMES="libiomp5" -DOpenMP_libiomp5_LIBRARY=/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin/libiomp5.so -DBUILD_WITH_IGPU=ON -DSEAL_USE_INTEL_GPU=ON -DSEAL_BUILD_TESTS=ON -DSEAL_BUILD_EXAMPLES=ON -DGPU_DEFAULT=ON -DCMAKE_BUILD_TYPE=Debug -DBUILD_WITH_HEAAN=OFF -DSEAL_USE_INTEL_LATTICE=OFF -DCMAKE_BUILD_TYPE=Debug ..
```

To remove GPU back-end
```bash
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_WITH_IGPU=OFF -DSEAL_USE_INTEL_GPU=OFF -DSEAL_USE_INTEL_XEHE=OFF -DSEAL_BUILD_TESTS=ON -DSEAL_BUILD_EXAMPLES=ON ..
```

### Intel-specific proxy environment issue in Linux

When cloning the repo, sometimes it fails with "bad certificate". One workaround:
```bash
https_proxy="" git clone htpps://...
```

## WINDOWS (MSVC 19)


### Intel-specific proxy environment
Set env variables

HTTP_PROXY:  http://proxy-chain.intel.com:911
HTTPS_PROXY: http://proxy-chain.intel.com:912


Install latest cmake
Install latest windows driver 
Install latest one API

Add oneAPI libraries path to the system PATH env varible:

ONE_API_FOLDER='C:\Program Files (x86)\Intel\oneAPI`
PATH='%ONE_API_FOLDER%\compiler\latest\windows\redist\intel64_win\compiler;%ONE_API_FOLDER%\compiler\latest\windows\bin;%PATH%`


Check environment variables names and values (see above)

Delete build directory

```
mkdir build
cd build
```

### GPU
```
cmake -A x64 -G"Visual Studio 16 2019" -DBUILD_WITH_IGPU=ON -DSEAL_USE_INTEL_GPU=ON -DSEAL_USE_INTEL_XEHE=ON -DGPU_DEFAULT=OFF -DCMAKE_BUILD_TYPE=Debug -DSEAL_BUILD_TESTS=ON -DSEAL_USE_INTEL_LATTICE=OFF -DBUILD_WITH_HEAAN=OFF  ..
```

### CPU only
```
cmake -A x64 -G"Visual Studio 16 2019" -DBUILD_WITH_IGPU=OFF -DSEAL_USE_INTEL_GPU=OFF -DSEAL_USE_INTEL_XEHE=OFF -DCMAKE_BUILD_TYPE=Debug -DSEAL_BUILD_TESTS=ON -DSEAL_USE_INTEL_LATTICE=OFF -DBUILD_WITH_HEAAN=OFF ..
```

click XeHE.sln - start MSVC-19
build all

Add existing project XeHE_app.vcxproj from XeHE_app directory
Make it startup project
Ctl F5

Should work





## Run on Linux
### Tests
* `./tests/tests ` -- expected output 'All passed'
* `./tests/tests [gpu]` -- run tests tagged with `gpu`, expected output 'All passed'
* `./tests/tests "Basic add test with uintarithmod"` -- run a specific test, expected output 'All passed'

#### To run SEAL tests and examples (while being in `build` dir):
* List all available SEAL tests:
`./external/SEAL/bin/sealtest --gtest_list_tests`
* To run a specific subset of SEAL tests:
`./external/SEAL/bin/sealtest --gtest_filter=UInt*`
* To run SEAL example apps:
`./external/SEAL/bin/sealexamples`

### Benchmarking tests
* `./tests/tests [gpu][perf] --benchmark-samples 1000` - run basic build-in benchmarks with 1000 samples per basic kernel

# Additional Features

## Inline Assembly
* To use exisitng inline assembly implementations on XeHE, define the flag `XeHE_INLINE_ASM` in `src/include/util/defines.h`
* To view/add inline assembly strings, go to `src/include/util/inline_kernels.hpp`
* Please visit this wiki [page](https://gitlab.devtools.intel.com/alyashev/XeHE/-/wikis/Main/vISA) for more details on inlining vISA asm in XeHE.


# Acknowledgement

Authors: Alexey Titov, Alexander Lyashevsky, Yiqin Qiu

Copyright (c) 2020, Intel Corporation

All rights reserved.
