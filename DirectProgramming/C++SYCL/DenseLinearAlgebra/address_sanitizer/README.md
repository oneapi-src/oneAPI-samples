# `Address Sanitizer` Sample

The `Address Sanitizer` sample demonstrates how to use the AddressSanitizer (ASan) memory detector with the SYCL library.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to use the Address Sanitizer tool
| Time to complete       | 10 minutes
| Category               | Memory Management


## Purpose

The `Address Sanitizer` sample illustrates how to use Address Sanitizer to manage memory errors with the SYCL library. Each of the examples shows a different error and how to initialize it.

All samples can be run on a CPU or a PVC GPU.

>**Note**: The gfx-driver needed to run ASan is version 1.3.28986 for a level zero GPU, 2024.18.3.0.head.prerelease for a CPU and 24.11.028986 for a GPU

The sample includes nine different mini samples that showcase the usage of ASan.

## Prerequisites

| Optimized for          | Description
|:---                    |:---
| OS                     | Ubuntu* 20.04 (or newer)
| Hardware               | GEN9 (or newer)
| Software               | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details

The basic SYCL implementation explained in the code includes:

- out-of-bounds
- use-after-free
- misalign-access
- double-free
- bad-free
- bad-context


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Address Sanitizer` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:

1. Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
2. Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
3. Open a terminal in VS Code (**Terminal > New Terminal**).
4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
*[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```
3. Run the program.
   ```
   make run_array_reduction
   ```   
6. Clean the project. (Optional)
   ```
   make clean
   ```

> **Note**: List of all samples and command to run them.
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>| File Name                                      | Run command
>|:---                                            |:---
>|`array_reduction.cpp`  			  | make run_array_reduction
>|`bad_free.cpp`          			  | make run_bad_free
>|`device_global.cpp`				  | make run_device_global
>|`group_local.cpp`				  | make run_group_local
>|`local_stencil.cpp`				  | make run_local_stencil
>|`map.cpp`					  | make run_map
>|`matmul_broadcast.cpp`		 	  | make run_matmul_broadcast
>|`misalign-long.cpp`			 	  | make run_misalign-long
>|`nd_range_reduction.cpp`			  | make run_nd_range_reduction

## Example Output

The following output is for the arrayreduction.cpp sample.
```
Histogram:
bin[0]: 4
bin[1]: 4
bin[2]: 4
bin[3]: 4
SUCCESS

====ERROR: DeviceSanitizer: bad-free on address 0x250d040
  #0 ./array_reduction() [0x405249]
  #1 /lib/x86_64-linux-gnu/libc.so.6(+0x29d90) [0x735d35229d90]
  #2 /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80) [0x735d35229e40]
  #3 ./array_reduction() [0x4049b5]

0x250d040 may be allocated on Host Memory
Segmentation fault (core dumped)
make[3]: *** [src/CMakeFiles/run_array_reduction.dir/build.make:70: src/CMakeFiles/run_array_reduction] Error 139
make[2]: *** [CMakeFiles/Makefile2:392: src/CMakeFiles/run_array_reduction.dir/all] Error 2
make[1]: *** [CMakeFiles/Makefile2:399: src/CMakeFiles/run_array_reduction.dir/rule] Error 2
make: *** [Makefile:254: run_array_reduction] Error 2
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
