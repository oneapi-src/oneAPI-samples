# `Guided Matrix Multiplication Bad Buffers` Sample

The `Guided Matrix Multiplication Bad Buffers` sample demonstrates how to use several tools in the Intel® oneAPI Base Toolkit (Base Kit) to triage incorrect use of the SYCL language.

The sample is a simple program that multiplies together two large matrices and verifies the results.

| Area                  | Description
|:---                   |:---
| What you will learn   | A method to determine the root cause problems from passing bad buffers through the SYCL runtime.
| Time to complete      | 50 minutes

>**Note**: For comprehensive instructions on the Intel® Distribution for GDB* and writing SYCL code, see the *[Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/overview.html)*. (Use search or the table of contents to find relevant information quickly.)

## Purpose

The two samples in this tutorial show examples of how to debug issues arising from passing bad buffers through the SYCL runtime, one when using SYCL buffers and one when using a direct reference to device memory.

In one case, we will know that there is a problem due to a crash. In the other case, we will get bad results.

The sample includes different versions of a simple matrix multiplication program.

| File                          | Description
|:---                           |:---
| `a1_matrix_mul_zero_buff.cpp` | This example shows the crash you get when a zero-element buffer is passed to a SYCL `submit`  lambda function.
| `a2_matrix_mul.cpp`           | A working version of the matrix multiply code that uses SYCL buffers and accessors.
| `b1_matrix_mul_null_usm.cpp`  | This example shows you the bad results you get when a null pointer to device memory is passed to a SYCL `submit` lambda function.
| `b2_matrix_mul_usm.cpp`       | A working version of the matrix multiply code that uses explicit pointers to host and device memory rather than SYCL buffers and accessors.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 24.04 LTS
| Hardware                | GEN9 or newer
| Software                | Intel® oneAPI DPC++/C++ Compiler 2025.1 <br> Intel® Distribution for GDB* 2025.1 <br> Unified Tracing and Profiling Tool 2.1.2, which is available from the [following Github repository](https://github.com/intel/pti-gpu/tree/master/tools/unitrace).
| Intel GPU Driver | Intel® General-Purpose GPU Rolling Release driver 2507.12 or later from https://dgpu-docs.intel.com/releases/releases.html

## Key Implementation Details

The basic SYCL* standards implemented in the code include the use of the following:

- SYCL* local buffers and accessors (declare local memory buffers and accessors to be accessed and managed by each workgroup)
- Explicit memory operations using Unified Shared Memory (USM)
- SYCL* kernels (including parallel_for function and explicit memory copies)
- SYCL* queues

## Set Environment Variables

When working with the command-line interface (CLI), configure the oneAPI toolkit environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries and tools are ready for development.

## Build the `Guided Matrix Multiplication Bad Buffers` Programs

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/use-the-setvars-and-oneapi-vars-scripts-with-linux.html)*.

### Use Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension *Environment Configurator for Intel® Software Development Tools*.

 2. Download a sample using the extension *Code Sample Browser for Intel® Software Development Tools*.

 3. Open a terminal in VS Code (`Terminal > New Terminal`).

 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the *[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-vs-code/current/overview.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the programs.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```
3. Run the programs.
   ```
   make run_all
   ```
   >**Note**: The application by default uses the Level Zero runtime and will run without errors.  We will do a deeper investigation of the application, in particular with the OpenCL™ runtime, to expose problems that could also manifest in Level Zero.

   For the zero buffer version only, enter the following:
   ```
   make run_a1_zero_buff
   ```
   For the working buffers version only, enter the following:
   ```
   make run_a2
   ```
   For the USM null pointer version only, enter the following:
   ```
   make run_b1_null_usm
   ```
   For the working USM version only, enter the following:
   ```
   make run_b2_usm
   ```
4. Clean the program. (Optional)
   ```
   make clean
   ```


#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html)* for more information on using the utility.


## Guided Debugging

These instructions assume you have installed the Intel® Distribution for GDB* and have a basic working knowledge of GDB.

### Setting up to Debug on the GPU
To learn how setup and use Intel® Distribution for GDB*, see the *[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/docs/distribution-for-gdb/get-started-guide-linux/current/overview.html)*.  Additional setup instructions you should follow are at *[GDB-PVC debugger](https://dgpu-docs.intel.com/system-user-guides/DNP-Max-1100-userguide/DNP-Max-1100-userguide.html#gdb-pvc-debugger)* and *[Configuring Kernel Boot Parameters](https://dgpu-docs.intel.com/driver/configuring-kernel-boot-parameters.html)*.

Documentation on using the debugger in a variety of situations can be found at *[Debug Examples in Linux](https://www.intel.com/content/www/us/en/docs/distribution-for-gdb/tutorial-debugging-dpcpp-linux/current/overview.html)*

>**Note**: SYCL applications will use the oneAPI Level Zero runtime by default. oneAPI Level Zero provides a low-level, direct-to-metal interface for the devices in a oneAPI platform. For more information see the *[Level Zero Specification Documentation - Introduction](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/INTRO.html)* and *[Intel® oneAPI Level Zero](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/intel-oneapi-level-zero.html)*.

### Getting the Tracing and Profiling Tool

At a step in this tutorial, the instructions require a utility that was not installed with the Intel® oneAPI Base Toolkit (Base Kit).

To complete the steps in the following section, you must download the [Unified Tracing and Profiling Tool](https://github.com/intel/pti-gpu/tree/master/tools/unitrace) code from GitHub and build the utility. The build instructions are included in the README in the GitHub repository.  This build will go much more smoothly if you first install the latest drivers from [the Intel GPU driver download site](https://dgpu-docs.intel.com/driver/overview.html), especially the development packages (only available in the Data Center GPU driver install ).  Once you have built the utility, you invoke it on the command line in front of your program (similar to using GDB).

### Guided Instructions for Zero Buffer

In `a1_matrix_mul_zero_buff`, a zero-element buffer is passed to a SYCL submit `lambda` function. **This will cause the application to crash.**

1. Run the program without the debugger.
   ```
   ./a1_matrix_mul_zero_buff
   ```
   On an Intel(R) Data Center GPU Max, the program should crash almost immediately with segmentation faults as shown below.
   ```
   Device: Intel(R) Data Center GPU Max 1550
   Problem size: c(150,600) = a(150,300) * b(300,600)
   Segmentation fault from GPU at 0x0, ctx_id: 1 (CCS) type: 0 (NotPresent), level: 3 (PML4), access: 1 (Write), banned: 0, aborting.
   Segmentation fault from GPU at 0x0, ctx_id: 1 (CCS) type: 0 (NotPresent), level: 3 (PML4), access: 1 (Write), banned: 0, aborting.
   Abort was called at 274 line in file:
   ./shared/source/os_interface/linux/drm_neo.cpp
   Aborted (core dumped)
   ```
   These error messages tells us that we wrote to an address on a memory page that we did not allocate on the GPU (generating an unexpected page fault)

   On an Intel(R) Graphics GPU, the crash will look something like this:
   ```
   Device: Intel(R) Graphics [0xe20b]
   Problem size: c(150,600) = a(150,300) * b(300,600)
   terminate called after throwing an instance of 'sycl::_V1::exception'
   what():  Native API failed. Native API returns: 20 (UR_RESULT_ERROR_DEVICE_LOST)
   Aborted (core dumped)
   ```

2. Start the debugger to watch the application failure and find out where it failed.   Since the message indicates that the failure was on the GPU, we need to enable GPU debugging by doing [some setup on your system](#setting-up-to-debug-on-the-gpu).

   However, in this case, let's see if we can catch the stack dump by running on the CPU where it is easier to pull data from the failing kernel.
   ```
   ONEAPI_DEVICE_SELECTOR=opencl:cpu gdb-oneapi ./a1_matrix_mul_zero_buff
   ```
   > **Note:** this will only work if the `sycl-ls` command shows OpenCL
   > devices for the graphics card, such as like this:

   ```
   $ sycl-ls
   [opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz OpenCL 3.0 (Build 0) [2024.18.6.0.02_160000]
   [opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1550 OpenCL 3.0 NEO  [24.22.29735.27]
   [opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1550 OpenCL 3.0 NEO  [24.22.29735.27]
   [opencl:cpu][opencl:3] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz OpenCL 3.0 (Build 0) [2023.16.7.0.21_160000]
   [opencl:fpga][opencl:4] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.7.0.21_160000]
   [level_zero:gpu][level_zero:0] Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1550 1.3 [1.3.29735]
   [level_zero:gpu][level_zero:1] Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1550 1.3 [1.3.29735]
   ```

   If you are missing `[opencl:gpu]` devices you may have to add the necessary libraries to your device path by setting the appropriate path in `DRIVERLOC` and then running the following four commands (for Ubuntu - adapt for other OSes):

   ```
   export DRIVERLOC=/usr/lib/x86_64-linux-gnu
   export OCL_ICD_FILENAMES=$OCL_ICD_FILENAMES:$DRIVERLOC/intel-opencl/libigdrcl.so
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DRIVERLOC
   export PATH=$PATH:/opt/intel/oneapi:$DRIVERLOC
   ```

3. You should get the prompt `(gdb)`.

4. From the debugger, run the program. The program will fail.
    ```
    (gdb) run
    ```
    The program will stop with a segmentation fault that will look something like this

    ```
    Problem size: c(150,600) = a(150,300) * b(300,600)

   Thread 1 "a1_matrix_mul_z" received signal SIGSEGV, Segmentation fault.
   0x00007ffff4b0cf72 in main::{lambda(auto:1&)#2}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(auto:1)#1}::operator()<sycl::_V1::item<2, true> >(sycl::_V1::item<2, true>) const (this=0x7fffffff9ff0,
      index=sycl::item range ..., offset ... = ...) at a1_matrix_mul_zero_buff.cpp:93
   93              a[index] = 1.0f;
   (gdb)
    ```

5. Run a `backtrace` to get a summary showing the approximate location triggering the assert.
   ```
   (gdb) backtrace
   ```

   The output might look similar to the following if the crash happens on Thread 1.
   ```
   #0  0x00007ffff4b0cf72 in main::{lambda(auto:1&)#2}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(auto:1)#1}::operator()<sycl::_V1::item<2, true> >(sycl::_V1::item<2, true>) const (this=0x7fffffff9ff0,
    index=sycl::item range ..., offset ... = ...) at a1_matrix_mul_zero_buff.cpp:93
   #1  0x00007ffff4b0e8f5 in _ZTSZZ4mainENKUlRT_E0_clIN4sycl3_V17handlerEEEDaS0_EUlS_E_ (_arg_a=sycl::id = ...,
      _arg_a=sycl::id = ..., _arg_a=sycl::id = ..., _arg_a=sycl::id = ...)
      at /opt/intel/oneapi/compiler/2025.1/bin/compiler/../../include/sycl/handler.hpp:1457
   #2  0x00007ffff29a1d4f in non-virtual thunk to Intel::OpenCL::DeviceBackend::Kernel::RunGroup(void const*, unsigned long const*, void*) const () from /opt/intel/oneapi/compiler/2025.1/lib/libintelocl.so
   #3  0x00007ffff28940af in Intel::OpenCL::CPUDevice::NDRange::ExecuteIteration(unsigned long, unsigned long, unsigned long, void*) () from /opt/intel/oneapi/compiler/2025.1/lib/libintelocl.so
   #4  0x00007ffff28a20b3 in tbb::detail::d1::start_for<Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit>, TaskLoopBody2D<Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit> >, tbb::detail::d1::auto_partitioner const>::run_body(Intel::OpenCL::TaskExecutor::BlockedRangeByDefaultTBB2d<Intel::OpenCL::TaskExecutor::NoProportionalSplit>&) ()
      from /opt/intel/oneapi/compiler/2025.1/lib/libintelocl.so
   :
   #39 0x000000000040bda5 in std::shared_ptr<sycl::_V1::detail::buffer_impl>::~shared_ptr (this=0x7fffffffb7e8)
      at /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/bits/shared_ptr.h:175
   #40 0x000000000040b2b5 in sycl::_V1::detail::buffer_plain::~buffer_plain (this=0x7fffffffb7e8)
      at /opt/intel/oneapi/compiler/2025.1/bin/compiler/../../include/sycl/buffer.hpp:87
   #41 0x00000000004095a2 in sycl::_V1::buffer<float, 2, sycl::_V1::detail::aligned_allocator<float>, void>::~buffer (
      this=0x7fffffffb7e8) at /opt/intel/oneapi/compiler/2025.1/bin/compiler/../../include/sycl/buffer.hpp:485
   #42 0x0000000000403cef in main ()
      at a1_matrix_mul_zero_buff.cpp:123
   ```

   Note that the stack originates from `a1_matrix_mul_zero_buff.cpp:123` (if you look in the source you will see that this is the end of the block containing all the offload code), but we crash in `a1_matrix_mul_zero_buff.cpp:93`.   This is a side-effect of the fact that SYCL does not wait for the submitted kernel to complete (unless you tell it to), so the main thread has made it all the way past all the offload statements and is waiting for them to complete.

   If the crash happens on a thread other than the first thread, you'll have a shorter that that starts from a "clone":

   ```
   #0  0x00007ffff7886fa2 in main::{lambda(auto:1&)#2}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(auto:1)#1}::operator()<sycl::_V1::item<2, true> >(sycl::_V1::item<2, true>) const (this=0x7fffd5fff450,
    index=sycl::item range ..., offset ... = ...) at a1_matrix_mul_zero_buff.cpp:93
   #1  0x00007ffff78888f5 in _ZTSZZ4mainENKUlRT_E0_clIN4sycl3_V17handlerEEEDaS0_EUlS_E_ (_arg_a=sycl::id = ...,
      _arg_a=sycl::id = ..., _arg_a=sycl::id = ..., _arg_a=sycl::id = ...)
      at /opt/intel/oneapi/compiler/2025.1/bin/compiler/../../include/sycl/handler.hpp:1457
   #2  0x00007ffff2025d4f in non-virtual thunk to Intel::OpenCL::DeviceBackend::Kernel::RunGroup(void const*, unsigned long const*, void*) const () from /opt/intel/oneapi/compiler/2025.1/lib/libintelocl.so
   #3  0x00007ffff1f180af in Intel::OpenCL::CPUDevice::NDRange::ExecuteIteration(unsigned long, unsigned long, unsigned long, void*) () from /opt/intel/oneapi/compiler/2025.1/lib/libintelocl.so
   :
   #21 0x00007ffff72a1e2e in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
   #22 0x00007ffff7333a4c in __GI___clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78
   ```
    This stack might look a little odd due to the fact we are seeing one thread out of many launched to execute the kernels.   We ran the debugger on the host OpenCL CPU driver, where the parallel processing is implemented using Intel(R) oneAPI Threading Building Blocks.

6. Examine the code at the crash
   ```
   (gdb) list
   88            accessor a(a_buf, h, write_only);
   89
   90            // Execute kernel.
   91            h.parallel_for(range(M, N), [=](auto index) {
   92              // Each element of matrix a is 1.
   93              a[index] = 1.0f;
   94            });
   95          });
   96
   97          // Submit command group to queue to multiply matrices: c = a * b
   ```
   The error at line 93 suggests there is some problem with the accessor `a` or the buffer it points to

7. Inspect `a` in the debugger using the `print` command.  You will see something like the following:
   ```
   (gdb) p a
   $1 = sycl::accessor write range {0, 0}
   ```
   Or in the more verbose manner
   ```
   (gdb) print /r a
   ```
   >**Note:** `/r` disables the *pretty printer* for the SYCL `buffer` class. You can see all available pretty printers using `info pretty-printer` at the `gdb` prompt.

   You might notice that this buffer has a size 0 by 0 elements (the `AccessRange` and `MemRange` are for a `common_array` of size 0 by 0 elements). Since it has zero size, this buffer is the problem.

   ```
   $2 = {<sycl::_V1::detail::accessor_common<float, 2, (sycl::_V1::access::mode)1025, (sycl::_V1::access::target)2014, (sycl::_V1::access::placeholder)0, sycl::_V1::ext::oneapi::accessor_property_list<> >> = {
      static AS = sycl::_V1::access::address_space::global_space, static IsHostBuf = false, static IsHostTask = false,
      static IsPlaceH = <optimized out>, static IsGlobalBuf = true, static IsConstantBuf = false,
      static IsAccessAnyWrite = true, static IsAccessReadOnly = false, static IsConst = false,
      static IsAccessReadWrite = <optimized out>,
      static IsAccessAtomic = <optimized out>}, <sycl::_V1::detail::OwnerLessBase<sycl::_V1::accessor<float, 2, (sycl::_V1::access::mode)1025, (sycl::_V1::access::target)2014, (sycl::_V1::access::placeholder)0, sycl::_V1::ext::oneapi::accessor_property_list<> > >> = {<No data fields>}, static AdjustedDim = 2, static IsAccessAnyWrite = true,
   static IsAccessReadOnly = false, static IsConstantBuf = false, static IsGlobalBuf = true, static IsHostBuf = false,
   static IsPlaceH = <optimized out>, static IsConst = false, static IsHostTask = false, impl = {
      Offset = {<sycl::_V1::detail::array<2>> = {common_array = {0, 0}}, static dimensions = <optimized out>},
      AccessRange = {<sycl::_V1::detail::array<2>> = {common_array = {0, 0}}, static dimensions = <optimized out>},
      MemRange = {<sycl::_V1::detail::array<2>> = {common_array = {0, 0}}, static dimensions = <optimized out>}}, {
      MData = 0x0}}
   ```

8. Now look at the `index` variable, which represents the iteration space that we will traverse to set all elements of the array `a` to an initial value of `1.0`.   You will see something like this:
   ```
   (gdb) p index
   $3 = sycl::item range {150, 300} = {131, 0}
   (gdb) p /r index
   $4 = {static dimensions = <optimized out>, MImpl = {MExtent = {<sycl::_V1::detail::array<2>> = {common_array = {150,
            300}}, static dimensions = <optimized out>}, MIndex = {<sycl::_V1::detail::array<2>> = {common_array = {131,
            0}}, static dimensions = <optimized out>}, MOffset = {<sycl::_V1::detail::array<2>> = {common_array = {0,
            0}}, static dimensions = <optimized out>}}}
   ```
   Clearly there is a mismatch here!   'a' has no space reserved for it, yet we will be iterating over 150 by 300 elements (and updating element 120 by 144 in this thread), which is clearly an error.

9. To further root-cause the error, we will need to restart the program and look at the values of `a_buf` and `b_buf`, which are not in scope in any of our stack frames.   We'll set some breakpoints where they are used:

   ```
   (gdb) b 79
   Breakpoint 1 at 0x40481b: a1_matrix_mul_zero_buff.cpp:79. (6 locations)
   (gdb) b 91
   Breakpoint 2 at 0x40602b: a1_matrix_mul_zero_buff.cpp:91. (6 locations)
   (gdb) r
   The program being debugged has been started already.
   Start it from the beginning? (y or n) y
   Starting program: ./a1_matrix_mul_zero_buff
   Registering SYCL extensions for gdb
   [Thread debugging using libthread_db enabled]
   :
   Problem size: c(150,600) = a(150,300) * b(300,600)

   Thread 1 "a1_matrix_mul_z" hit Breakpoint 1.1, main::{lambda(auto:1&)#1}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const (this=0x7fffffffb550, h=sycl::handler& = {...})
      at /nfs/site/home/cwcongdo/oneAPI-samples-mine/Tools/ApplicationDebugger/guided_matrix_mult_BadBuffers/src/a1_matrix_mul_zero_buff.cpp:79
   79            h.parallel_for(range(N, P), [=](auto index) {
   (gdb)
   ```
   Now look at the dimensions of `b` and `b_buf`:

   ```
   (gdb) p b
   $5 = sycl::accessor write range {300, 600, 1} = {Cannot access memory at address 0x0

   (gdb) p b_buf
   $6 = sycl::buffer& range {300, 600} = {Cannot access memory at address 0x0
   ```
   Continuing after deleting the first breakpoint, let's look at `a` and `a_buf`
   ```
   (gdb) d 1
   (gdb) c
   Continuing.

   Thread 1 "a1_matrix_mul_z" hit Breakpoint 2.1, main::{lambda(auto:1&)#2}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const (this=0x7fffffffb550, h=sycl::handler& = {...})
      at /nfs/site/home/cwcongdo/oneAPI-samples-mine/Tools/ApplicationDebugger/guided_matrix_mult_BadBuffers/src/a1_matrix_mul_zero_buff.cpp:91
   91            h.parallel_for(range(M, N), [=](auto index) {
   (gdb) p a
   $7 = sycl::accessor write range {0, 0, 1}

   (gdb) p a_buf
   $8 = sycl::buffer& range {0, 0}
   (gdb)
   ```
   As you can clearly see,  `a_buf` has size 0 by 0 elements, while `b_buf` has a size 300 by 600 elements.

   Looking at the source again, you'll see that this originated when the buffers were created around line 61 and 62:
   ```
    buffer<float, 2> a_buf(range(0, 0));
    buffer<float, 2> b_buf(range(N, P));
   ```

   In real code the values to the ranges may be passed into the function from outside, so you will need to inspect those as well as the code where they are calculated.  For example, you would need to find the values of `M`, `N`, and `P` to make sure that the resulting buffer sizes are non-zero in these buffer definitions:
   ```
    buffer<float, 2> a_buf(range(M, N));
    buffer<float, 2> b_buf(range(N, P));
   ```

### Guided Instructions for Null Device Pointer

In `b1_matrix_mul_null_usm.cpp` a bad (in this case, null) pointer that is supposed to represent unallocated memory on the device is inadvertently used in a kernel.  This example uses unified shared memory rather than SYCL buffers like the previous example.

1. Run the program on the GPU using Level Zero.
   ```
   ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./b1_matrix_mul_null_usm
   ```
   This run produces troublesome output.
   ```
   Device max work group size: 1024
   Problem size: c(150,600) = a(150,300) * b(300,600)
   Segmentation fault from GPU at 0x13000, ctx_id: 1 (CCS) type: 0 (NotPresent), level: 3 (PML4), access: 0 (Read), banned: 0, aborting.
   Segmentation fault from GPU at 0x13000, ctx_id: 1 (CCS) type: 0 (NotPresent), level: 3 (PML4), access: 0 (Read), banned: 0, aborting.
   Abort was called at 269 line in file:
   ./shared/source/os_interface/linux/drm_neo.cpp
   Aborted (core dumped)
   ```

2. If your machine shows an OpenCL™ gpu device when you run `sycl-ls`, check the output if we run on the GPU again but using OpenCL.
   ```
   ONEAPI_DEVICE_SELECTOR=opencl:gpu ./b1_matrix_mul_null_usm
   ```

   The results should be the same as the Level Zero output.

   > **Note:** this will only work if the `sycl-ls` command shows OpenCL
   > devices for the graphics card, such as like this:

   ```
   $ sycl-ls
   [opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz OpenCL 3.0 (Build 0) [2024.18.6.0.02_160000]
   [opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1550 OpenCL 3.0 NEO  [24.22.29735.27]
   [opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1550 OpenCL 3.0 NEO  [24.22.29735.27]
   [opencl:cpu][opencl:3] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz OpenCL 3.0 (Build 0) [2023.16.7.0.21_160000]
   [opencl:fpga][opencl:4] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.7.0.21_160000]
   [level_zero:gpu][level_zero:0] Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1550 1.3 [1.3.29735]
   [level_zero:gpu][level_zero:1] Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1550 1.3 [1.3.29735]
   ```

   If you are missing `[opencl:gpu]` devices you may have to add the necessary libraries to your device path by setting the appropriate path in `DRIVERLOC` and then running the following four commands (for Ubuntu - adapt for other OSes):

   ```
   export DRIVERLOC=/usr/lib/x86_64-linux-gnu
   export OCL_ICD_FILENAMES=$OCL_ICD_FILENAMES:$DRIVERLOC/intel-opencl/libigdrcl.so
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DRIVERLOC
   export PATH=$PATH:/opt/intel/oneapi:$DRIVERLOC
   ```

3. Check the output we get by bypassing the GPU entirely and using the OpenCL driver for CPU.
   ```
   ONEAPI_DEVICE_SELECTOR=opencl:cpu ./b1_matrix_mul_null_usm
   ```
   ```
   Problem size: c(150,600) = a(150,300) * b(300,600)
   Segmentation fault (core dumped)
   ```

#### Attempting to Understand What Is Happening

Why did we try with multiple backends?   If one had shown correct or incorrect results, and one had crashed, we might be facing a race condition that only occasionally manifests as something that goes terribly wrong.  Or one of the backbends might have a bug.  But here all three crash, so it's likely the program is doing something illegal to memory.  The host CPU is a particularly good place to test for illegal memory accesses, because the CPU never allows pointers with an address within a few kilobytes of address 0x0, while this may be legally allocated memory on the GPU.

Another reason to try different backends is that debugging support may differ between different GPU drivers and/or different GPU models.  Debugging the program using the OpenCL™ CPU driver gets around these issues.

Let's see what caused the problem by running in the debugger:

1. Start the debugger using OpenCL™ on the CPU.
   ```
   ONEAPI_DEVICE_SELECTOR=opencl:cpu gdb-oneapi ./b1_matrix_mul_null_usm
   ```
2. You should get the prompt `(gdb)`.

3. From the debugger, run the program.
   ```
   (gdb) run
   ```
   This will launch the application, and quickly generate an error.

   ```
   Problem size: c(150,600) = a(150,300) * b(300,600)

   Thread 2.1616 received signal SIGSEGV, Segmentation fault
   Warning: The location reported for the signal may be inaccurate.
   [Switching to thread 2.1616:0 (ZE 0.25.1.7 lane 0)]
   0x00008000ffca6a70 in main::{lambda(auto:1&)#2}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(auto:1)#1}::operator()<sycl::_V1::item<2, true> >(sycl::_V1::item<2, true>) const (this=0xff00000000e6ba90,
      index=sycl::item range ..., offset ... = ...) at b1_matrix_mul_null_usm.cpp:122
   122               sum += dev_a[a_index] * dev_b[b_index];
   (gdb)
   ```

4. Run the backtrace to see where we run into a problem.
   ```
   (gdb) backtrace
   ```
   You should see output similar to the following.
   ```
   #0  0x00007ffff4a75f60 in main::{lambda(auto:1&)#2}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(auto:1)#1}::operator()<sycl::_V1::item<2, true> >(sycl::_V1::item<2, true>) const (this=0x7fff3d5ff438,
      index=sycl::item range ..., offset ... = ...) at b1_matrix_mul_null_usm.cpp:122
   #1  0x00007ffff4a76490 in _ZTSZZ4mainENKUlRT_E0_clIN4sycl3_V17handlerEEEDaS0_EUlS_E_ (_arg_width_a=300,
      _arg_dev_a=0x0, _arg_dev_b=0x7fffe5250000, _arg_dev_c=0x7fffe5d20000)
      at /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../include/sycl/handler.hpp:1639
   #2  0x00007ffff2ac223f in non-virtual thunk to Intel::OpenCL::DeviceBackend::Kernel::RunGroup(void const*, unsigned long const*, void*) const () from /opt/intel/oneapi/compiler/2025.0/lib/libintelocl.so
   :
   #21 0x00007ffff769caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
   #22 0x00007ffff7729c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78
   (gdb)
   ```

5. We got lucky in that the frame where we crashed is in our code.   Let's examine the code in a little more detail:
   ```
   (gdb) list
   ```
   You should see the code around the line reporting the problem.

   ```
   117
   118             // Compute the result of one element of c
   119             for (int i = 0; i < width_a; i++) {
   120               auto a_index = row * width_a + i;
   121               auto b_index = i * P + col;
   122               sum += dev_a[a_index] * dev_b[b_index];
   123             }
   124
   125             auto idx = row * P + col;
   126             dev_c[idx] = sum;

   ```

6. Let's check the pointers in use at line 122.
   1. Check the first variable.
      ```
      (gdb) print dev_a
      ```
      This should display output similar to the following.
      ```
      $1 = (float *) 0x0
      ```
   2. Check the second variable.
      ```
      (gdb) print dev_b
      ```
      This should display output similar to the following.
      ```
      $2 = (float *)  0xff003ffffff00000
      ```
   This located the problem:  one of the array pointers is null (zero) while the other has a (hopefully) valid value. If both pointers were valid, we'd want to check the index values being used to access the arrays, and the memory values at those locations (and at the start of the array).

#### Understanding What Is Happening

Early on, operating system designers realized that it was common for developers to write bugs in which they accidentally try to de-reference null pointers (access them like an array).  To make it easier to find these errors the operating system designers implemented logic that made it illegal for any program to access the first two memory pages (so from address 0x0 to around 0x2000).  They didn't go further to explicitly validate the address of any memory accessed by a program (because it is too expensive), but this range of illegal addresses was a cheap check that caught a huge number of bugs.

GPUs typically don't have a lot of memory, so they can't afford to set aside a large range of illegal addresses.  So both Level Zero and OpenCL passed on the pointer to device memory (in our case intentionally zero) assuming it was valid.   From there we had three possible outcomes:

1.  If the pointer was correct and pointing to allocated memory, the program would have completed correctly.
2.  If the pointer was incorrect but was pointing to memory allocated for something else, the kernel would have accessed random memory values in calculating the sum on line 122, and returned incorrect results as a consequence.
3.  If the pointer was incorrect and pointing to memory that was never allocated, the kernel will crash on either the GPU or CPU.

What would we have seen if `dev_a` had contained just a random pointer value?  If you were lucky, the address returned when you print `dev_a` would look non-null and very different from the one returned when you printed `dev_b`, but that's not a certainty.  Or it would have pointed to memory not owned by your process and caused a crash.

#### Other Debug Techniques

If you are trying to debug bad values on the GPU, you will be very challenged to figure out whether or not this is due to bad inputs. You *might* be able to catch the bad pointer if you breakpoint at or just before the line there you crash. You also *might* see something odd if you capture the arguments sent to API calls with `unitrace`.

You need to build `unitrace` before you can use it. See the instructions at [Unified Tracing and Profiling Tool](#getting-the-tracing-and-profiling-tool). Once you have built the utility, you can invoke it before your program (similar to GDB).

1. Run the `unitrace` tool. (we include `-c` when invoking `unitrace` to enable call logging of API calls.)

   >**Note**: You must modify the command shown below to include the path to where you installed the `unitrace` utility.
   ```
   [path]/unitrace -c ./b1_matrix_mul_null_usm > b1_matrix_mul_null_usm_unitrace_out.txt 2>&1
   ```
   While reviewing the output, you might see something like the following excerpt near the bottom of the output.
   ```
   :
   >>>> [1487508411066857] zeKernelCreate: hModule = 44797904 desc = 140735939087520 {ZE_STRUCTURE_TYPE_KERNEL_DESC(0x1d) 0 0 "_ZTSZZ4mainENKUlRT_E0_clIN4sycl3_V17handlerEEEDaS0_EUlS_E_"} phKernel = 140735939087512 (hKernel = 14798282318754847232)
   <<<< [1487508411092885] zeKernelCreate [21566 ns] hKernel = 44799912 -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1487508411096140] zeDeviceGetSubDevices: hDevice = 39507224 pCount = 140735939087500 (Count = 0) phSubdevices = 0
   <<<< [1487508411098094] zeDeviceGetSubDevices [395 ns] Count = 0 -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1487508411099724] zeDeviceGetSubDevices: hDevice = 39507224 pCount = 140735939087500 (Count = 0) phSubdevices = 0
   <<<< [1487508411100759] zeDeviceGetSubDevices [22 ns] Count = 0 -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1487508411104804] zeKernelSetIndirectAccess: hKernel = 44799912 flags = 7
   <<<< [1487508411106715] zeKernelSetIndirectAccess [431 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1487508411113472] zeKernelGetProperties: hKernel = 44799912 pKernelProperties = 45076936
   <<<< [1487508411122357] zeKernelGetProperties [7693 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1487508411125804] zeKernelSetArgumentValue: hKernel = 44799912 argIndex = 0 argSize = 4 pArgValue = 44677968
   <<<< [1487508411127704] zeKernelSetArgumentValue [265 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1487508411130788] zeKernelSetArgumentValue: hKernel = 44799912 argIndex = 1 argSize = 8 pArgValue = 0
   <<<< [1487508411132260] zeKernelSetArgumentValue [528 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1487508411133457] zeKernelSetArgumentValue: hKernel = 44799912 argIndex = 2 argSize = 8 pArgValue = 140735939087912
   <<<< [1487508411135573] zeKernelSetArgumentValue [1131 ns] -> ZE_RESULT_SUCCESS(0x0)
   >>>> [1487508411136604] zeKernelSetArgumentValue: hKernel = 44799912 argIndex = 3 argSize = 8 pArgValue = 140735939087912
   <<<< [1487508411137932] zeKernelSetArgumentValue [459 ns] -> ZE_RESULT_SUCCESS(0x0)
   :
   ```
   Notice how all the kernel arguments have a non-zero value except one (`argIndex = 1`) when they are set up. We have nothing to help us map the kernel arguments created by the SYCL runtime and passed to Level Zero to the arguments in the user program (the value may be OK). However, if you see a kernel argument that looks out of place (note that the values at all other argument indexes are very similar), you might want to be suspicious.

   Other than this clue, there are no other hints that the bad output from the program is due to a bad input argument rather than a race condition or other algorithmic error.  These sorts of issues can be partically tricky to diagnose when the device pointers are initialized in a different part of the program, or in a 3rd-party library.

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
