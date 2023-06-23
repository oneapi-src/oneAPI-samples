# `array-transform` Sample

This is a small SYCL*-compliant code sample for exercising application debugging using
Intel&reg; Distribution for GDB*.  It is highly recommended that you go
through this sample *after* you familiarize yourself with the basics of
SYCL programming, and *before* you start using the debugger.


| Area                | Description
|---------------------|--------------
| What you will learn | Essential debugger features for effective debugging on CPU, GPU (Linux only), and FPGA emulator
| Time to complete    | 20 minutes for CPU or FPGA emulator; 30 minutes for GPU

This sample accompanies 
[Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://software.intel.com/en-us/get-started-with-debugging-dpcpp)
of the application debugger.

## Purpose

The `array-transform` sample is a SYCL-conforming application with a small
computation kernel that is designed to illustrate key debugger
features such as breakpoint definition, thread switching,
scheduler-locking and SIMD lane views.  The sample is intended
for exercising the debugger, not for performance benchmarking.

The debugger supports debugging kernels that run on the CPU, GPU, or
accelerator devices.  Use the ONEAPI_DEVICE_SELECTOR environment variable
to select device.  The default device is Level Zero GPU device, if available.
For more details on possible values of this variable see [Environment Variables](https://intel.github.io/llvm-docs/EnvironmentVariables.html#oneapi-device-selector).
The selected device is displayed in the output.  Concrete instructions
about how to run the program and example outputs are given further
below.  For complete setup and usage instructions, see [Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://software.intel.com/en-us/get-started-with-debugging-dpcpp)
of the application debugger.

## Prerequisites

| Optimized for                                    | Description
|--------------------------------------------------|--------------
| OS                                               | Linux* Ubuntu* 20.04 to 22.04 <br> CentOS* 8 <br> Fedora* 30 <br> SLES 15 <br> Windows* 10, 11
| Hardware to debug offloaded <br> kernels on GPUs | Intel® Arc(tm) <br> Intel® Data Center GPU Flex Series
| Software                                         | Intel&reg; oneAPI DPC++/C++ Compiler

> **Note** although the sample can be run on all supported by Intel® oneAPI
> Base Toolkit platforms, the GPU debugger can debug only kernels offloaded
> onto devices specified at “Hardware to debug offloaded kernels on GPUs”
> while running with the L0 backend.  When the GPU device is different from
> the listed above, e.g., an integrated graphics device, breakpoints inside
> the kernel won't be hit.  In such case, try to switch the offload to a CPU
> device by using ONEAPI_DEVICE_SELECTOR environment variable.

We recommend to first make sure that the program you intend to debug is running
correctly on CPU and only after that switch the offload to GPU.

## Key Implementation Details

The basic SYCL implementation explained in the code includes device
selection, buffer, accessor, and command groups.  The kernel contains
data access via read/write accessors and a conditional statement to
illustrate (in) active SIMD lanes on a GPU.

## Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).


## Building and Running the `array-transform` Program
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Setup

Preliminary setup steps are needed for the debugger to function.
Please see the setup instructions in the Get Started Guide based on
your OS: 
- [Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/)
- [Get Started with Intel® Distribution for GDB* on Windows* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-windows/)

### Include Files

The include folder is located at
`%ONEAPI_ROOT%\dev-utilities\latest\include` on your development
system.

### Running Samples In DevCloud

If running a sample in the Intel DevCloud, remember that you must
specify the compute node (CPU, GPU, FPGA) and whether to run in
batch or interactive mode.  For the array transform sample, a node
with GPU and an interactive shell is recommended.

```
$ qsub -I -l nodes=1:gpu:ppn=2
```

For more information, see the Intel® oneAPI
Base Toolkit Get Started Guide
(https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

### Auto-Attach

The debugger has a feature called _auto-attach_ that automatically
starts and connects an instance of `gdbserver-gt` so that kernels
offloaded to the GPU can be debugged conveniently.  Auto-attach is
by default enabled.  To turn this feature off, if desired (e.g., if
interested in debugging CPU or FPGA-emu only), do:
```
$ export INTELGT_AUTO_ATTACH_DISABLE=1
```

To turn the feature back on:
```
$ unset INTELGT_AUTO_ATTACH_DISABLE
```

### On a Linux* System

Perform the following steps:

1.  Build the program using the following `cmake` commands.
    ```
    $ cd array-transform
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    ```
    > Note: The cmake configuration enforces the `Debug` build type.

2.  Run the program:
    ```
    $ ./array-transform
    ```
    > Note: to specify a device type to offload the kernel, use
    > the `ONEAPI_DEVICE_SELECTOR` environment variable.
    > E.g.  to restrict the offload only to CPU devices use:
    ```
    $ ONEAPI_DEVICE_SELECTOR=*:cpu ./array-transform
    ```

3.  Start a debugging session:
    ```
    $ gdb-oneapi array-transform
    ```

4.  Clean the program using:
    ```
    $ make clean
    ```

By default, CMake configures the build for Just-in-Time (JIT)
compilation of the kernel.  However, it also offers an option for
*Ahead-of-Time* (AoT) compilation.  To compile the kernel
ahead-of-time for a specific device, set the `SYCL_COMPILE_TARGET`
option to the desired device during configuration.  For CPU, use the
`cpu` value; for FPGA-emu, use the `fpga-emu` value.  Other values are
assumed to be for GPU and are passed directly to the GPU AoT
compiler.

For example, to do AoT compilation for a specific GPU device ID:

```
$ cmake .. -DSYCL_COMPILE_TARGET=<device id>
```
where the `<device id>` must be replaced with the actual device ID in the hex format.
Use `sycl-ls` command to list available devices on your target machine:

```
$ sycl-ls
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Graphics [0x56c1] 1.3 [1.3.0]
[ext_oneapi_level_zero:gpu:1] Intel(R) Level-Zero, Intel(R) Graphics [0x56c1] 1.3 [1.3.0]
[ext_oneapi_level_zero:gpu:2] Intel(R) Level-Zero, Intel(R) Graphics [0x56c1] 1.3 [1.3.0]
[ext_oneapi_level_zero:gpu:3] Intel(R) Level-Zero, Intel(R) Graphics [0x56c1] 1.3 [1.3.0]
```

In the above example, the device ID is `0x56c1`.

> *Hint:* Run `ocloc compile --help` to see all available GPU device options.

> **Note**: AoT compilation is particularly helpful in larger
> applications where compiling with debug information takes
> considerably longer time.

For instructions about starting and using the debugger, please
see [Get Started with Intel® Distribution for GDB* on Linux* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-linux/).


If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

### On a Windows* System Using Visual Studio* Version 2019 or Newer

#### Command line using MSBuild

* `set CL_CONFIG_USE_NATIVE_DEBUGGER=1`
* `MSBuild array-transform.sln /t:Rebuild /p:Configuration="debug"`

#### Visual Studio IDE

1. Right-click on the solution files and open via either Visual Studio 2019
   or in 2022.

2. Select Menu "Build > Build Solution" to build the selected configuration.

3. Select Menu "Debug > Start Debugging" to run the program.

4. The solution file is configured to set `ONEAPI_DEVICE_SELECTOR=*:cpu`
   for Local Windows Debugging.  That setting causes the program to offload
   on a CPU device.  To select a different device, go to the project's
   "Configuration Properties > Debugging" and edit the "Environment" field.
   Modify the value of `ONEAPI_DEVICE_SELECTOR` as you need.

For detailed instructions about starting and using the debugger,
please see [Get Started with Intel® Distribution for GDB* on Windows* OS Host](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-debugging-dpcpp-windows/).


### Example Outputs

```
$ ONEAPI_DEVICE_SELECTOR=*:cpu gdb-oneapi -q ./array-transform
Reading symbols from ./array-transform...
(gdb) break 54
Breakpoint 1 at 0x4057b7: file array-transform.cpp, line 54.
(gdb) run
...<snip>...
[SYCL] Using device: [Intel(R) Core(TM) i9-7900X processor] from [Intel(R) OpenCL]
[Switching to Thread 0x7fffe3bfe700 (LWP 925)]

Thread 4 "array-transform" hit Breakpoint 1, main::{lambda(auto:1&)#1}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(sycl::_V1::id<1>)#1}::operator()(sycl::_V1::id<1>) const (this=0x7fffc85582c8, index=sycl::_V1::id<1> = {...}) at array-transform.cpp:54
54              int element = in[index];  // breakpoint-here
(gdb)
```

```
$ ONEAPI_DEVICE_SELECTOR=*:fpga gdb-oneapi -q ./array-transform
Reading symbols from ./array-transform...
(gdb) break 54
Breakpoint 1 at 0x4057b7: file array-transform.cpp, line 54.
(gdb) run
...<snip>...
[SYCL] Using device: [Intel(R) FPGA Emulation Device] from [Intel(R) FPGA Emulation Platform for OpenCL(TM) software]
[Switching to Thread 0x7fffe1ffb700 (LWP 2387)]

Thread 6 "array-transform" hit Breakpoint 1, main::{lambda(auto:1&)#1}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(sycl::_V1::id<1>)#1}::operator()(sycl::_V1::id<1>) const (this=0x7fffc08cef48, index=sycl::_V1::id<1> = {...}) at array-transform.cpp:54
54              int element = in[index];  // breakpoint-here
(gdb)
```

```
$ ONEAPI_DEVICE_SELECTOR=level_zero:gpu gdb-oneapi -q ./array-transform
Reading symbols from ./array-transform...
(gdb) break 54
Breakpoint 1 at 0x4057b7: file array-transform.cpp, line 54.
(gdb) run
...<snip>...
intelgt: gdbserver-ze started for process 18496.
...<snip>...
[SYCL] Using device: [Intel(R) Data Center GPU Flex Series 140 [0x56c1]] from [Intel(R) Level-Zero]
[Switching to Thread 1.153 lane 0]

Thread 2.153 hit Breakpoint 1, with SIMD lanes [0-7], main::{lambda(auto:1&)#1}::operator()<sycl::_V1::handler>(sycl::_V1::handler&) const::{lambda(sycl::_V1::id<1>)#1}::operator()(sycl::_V1::id<1>) const (this=0xffffd556ab1898d0, index=...) at array-transform.cpp:54
54              int element = in[index];  // breakpoint-here
(gdb)
```

## Useful Commands

`help <cmd>`
: Print help info about the command `cmd`.

`run [arg1, ... argN]`
: Start the program, optionally with arguments.

`break <filename>:<line>`
: Define a breakpoint at the given source file's specified line.

`info break`
: Show the defined breakpoints.

`delete <N>`
: Remove the `N`th breakpoint.

`watch <exp>`
: Stop when the value of the expression `exp` changes.

`step`, `next`
: Single-step a source line, stepping into/over function calls.

`continue`
: Continue execution.

`print <exp>`
: Print value of expression `exp`.

`backtrace`
: Show the function call stack.

`up`, `down`
: Go one level up/down in the function call stack.

`disassemble`
: Disassemble the current function.

`info args`/`locals`
: Show the arguments/local vars of the current function.

`info reg <regname>`
: Show contents of the specified register.

`info inferiors`
: Display information about the `inferiors`.  For GPU
  offloading, one inferior represents the host process, and
  another (`gdbserver-gt`) represents the kernel.

`info threads <ID>`
: Display information about threads with id `ID`, including their
  active SIMD lanes. Omit id to display all threads.

`thread <thread_id>:<lane>`
: Switch context to the SIMD lane `lane` of the specified thread.
  E.g: `thread 2.6:4`

`thread apply <thread_id>:<lane> <cmd>`
: Apply command `cmd` to the specified lane of the thread.
  E.g.: `thread apply 2.3:* print element` prints
  `element` for each active lane of thread 2.3.
  Useful for inspecting vectorized values.

`x /<format> <addr>`
: Examine the memory at address `addr` according to
  `format`.  E.g.: `x /i $pc` shows the instruction pointed by
  the program counter.  `x /8wd &count` shows eight words in decimal
  format located at the address of `count`.

`info devices`
: List available GPU devices.

`set nonstop on/off`
: Enable/disable the nonstop mode.  This command may **not** be used
  after the program has started.

`set scheduler-locking on/step/off`
: Set the scheduler locking mode.

`maint jit dump <addr> <filename>`
: Save the JIT'ed objfile that contains address `addr` into the file
  `filename`.  Useful for extracting the kernel when running on
  the CPU device.

`info sharedlibrary`
: List the loaded shared libraries.  While debugging the kernel offloaded
  to GPU, use this command to find out the memory range of the kernel binary.

`dump binary memory <filename> <start_addr> <end_addr>`
: Dump the memory range from `start_addr` to `end_addr` into the file
 `filename`.

`cond [-force] <N> <exp>`
: Define the expression `exp` as the condition for breakpoint `N`.
  Use the optional `-force` flag to force the condition to be defined
  even when `exp` is invalid for the current locations of the breakpoint.
  Useful for defining conditions involving JIT-produced artificial variables.
  E.g.: `cond -force 1 __ocl_dbg_gid0 == 19`.

---

\* Intel is a trademark of Intel Corporation or its subsidiaries.  Other
names and brands may be claimed as the property of others.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
