# `jacobi` Sample

The 'jacobi' code sample is a small DPC++ code sample for exercising application
debugging using Intel&reg; Distribution for GDB\*. It is highly recommended that
you go through this sample *after* you familiarize yourself with the
[array-transform](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/ApplicationDebugger/array-transform)
sample and the
[Get Started Guide](https://software.intel.com/en-us/get-started-with-debugging-dpcpp)
for the debugger.

This sample contains two versions of the same program: `jacobi-bugged` and
`jacobi-fixed`. The former works correctly, but in the latter, several bugs were
injected. You can try to find and fix them using the debugger.

| Optimized for       | Description
|---------------------|--------------
| OS                  | Linux Ubuntu 18.04 to 20.04, CentOS* 8, Fedora* 30, SLES 15; Windows* 10
| Hardware            | Kaby Lake with GEN9 (on GPU) or newer (on CPU)
| Software            | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn | Find existing bugs in a program using the debugger
| Time to complete    | 1 hour for CPU or FPGA emulator; 2 hours for GPU

## Purpose

The `jacobi` sample is a DPC++ application that solves a hardcoded linear system
of equations `Ax=b` using the Jacobi iteration. The matrix `A` is an `n x n` sparse
matrix with diagonals `[1 1 5 1 1]`. Vectors `b` and `x` are `n x 1` vectors.
Vector `b` is set in the way that all components of the solution vector `x` are 1.

The program `jacobi-fixed` computes vector `x` correctly. That is, all
components of the resulting vector are close to 1.0. However, `jacobi-bugged`
returns an incorrect result. With the debugger, you can put a breakpoint in and
outside of the kernel, step through the code, and check the values of local
variables. Use these features to find places in `jacobi-bugged` where the
program does not behave as it was intended.

The sample is intended for exercising the debugger, not for performance
benchmarking.

The debugger supports debugging kernels that run on the CPU, GPU, or accelerator
devices. For convenience, the `jacobi` code sample provides the ability to
select the target device by using a command-line argument of `cpu`, `gpu`, or
`accelerator`.

The selected device is displayed in the output. For the overview of the
debugger, please refer to the
[Get Started Guide](https://software.intel.com/en-us/get-started-with-debugging-dpcpp)
of the application debugger.

## Key Implementation Details

The basic DPC++ implementation explained in the code includes device selection,
buffer, accessor, and command groups.

Each iteration of the Jacobi method performs the following computation:

```
x^{k+1} = D^{-1}(b - (A - D)x^k),
```

where `n x n` matrix `D` is a diagonal component of the matrix `A`. In the
sample, this computation is offloaded to the device at each iteration. In the
code, `x^{k+1}` corresponds to the variable `x_k1`, and `x^k` corresponds to
`x_k`. At the end of each iteration, we update `x_k` with the value of `x_k1`.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third-party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building and Running the `jacobi` Project

If you have not already done so, set up your CLI environment by sourcing
the setvars script located in the root of your oneAPI installation.

Linux Sudo: `. /opt/intel/oneapi/setvars.sh`
Linux User: `. ~/intel/oneapi/setvars.sh`
Windows: `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`

### Setup

Preliminary setup steps are needed for the debugger to function. Please see the
setup instructions in the Get Started Guide based on your OS:
[Linux](https://software.intel.com/en-us/get-started-with-debugging-dpcpp-linux),
[Windows](https://software.intel.com/en-us/get-started-with-debugging-dpcpp-windows).


### Include Files

The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on
your development system.


### Running Samples In DevCloud

If running a sample in the Intel DevCloud, remember that you must specify the
compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode.
We recommend running the `jacobi` sample on a node with GPU. In order to have an
interactive debugging session, we recommend using the interactive mode. To get
the setting, after connecting to the login node, type the following command:

```
$ qsub -I -l nodes=1:gpu:ppn=2
```
Within the interactive session on the GPU node, build and run the sample.
For more information, see the Intel® oneAPI Base Toolkit Get Started Guide
(https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

### On a Linux* System

Perform the following steps:

1.  Build the project using the following `cmake` commands.

    ```
    $ cd jacobi
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    ```

    This builds both `jacobi-bugged` and `jacoby-fixed` versions of the program.
    > Note: The cmake configuration enforces the `Debug` build type.

2.  Run the buggy program:

    ```
    $ ./jacobi-bugged <device>
    ```

    > Note: `<device>` is the type of the device to offload the kernel.
    > Use `cpu`, `gpu`, or `accelerator` to select the CPU, GPU, or the
    > FPGA emulator device, respectively.  E.g.:

    ```
    $ ./jacobi-bugged cpu
    ```

3.  Start a debugging session:

    ```
    $ gdb-oneapi --args jacobi-bugged <device>
    ```

4.  Clean the program using:

    ```
    $ make clean
    ```


For instructions about starting and using the debugger, please see the 
[Get Started Guide (Linux)](https://software.intel.com/en-us/get-started-with-debugging-dpcpp-linux).

### On a Windows* System Using Visual Studio* Version 2017 or Newer

#### Command line using MSBuild

* `set CL_CONFIG_USE_NATIVE_DEBUGGER=1`
* `MSBuild lacobi.sln /t:Rebuild /p:Configuration="debug"`

#### Visual Studio IDE

1. Right-click on the solution files and open via either Visual Studio 2017 or
   in 2019.

2. Select Menu "Build > Build Solution" to build the selected configuration.

3. Select Menu "Debug > Start Debugging" to run the program, the default startup
   project is `jacobi-bugged`.

4. The solution file is configured to pass `cpu` as the argument to the program
   while using "Local Windows Debugger", and `gpu` while using "Remote Windows
   Debugger". To select a different device, go to the project's "Configuration
   Properties > Debugging" and set the "Command Arguments" field. Use `gpu` or
   `accelerator` to target the GPU or the FPGA emulator device, respectively.

For detailed instructions about starting and using the debugger, please see the
[Get Started Guide (Windows)](https://software.intel.com/en-us/get-started-with-debugging-dpcpp-windows).

### Example Outputs

#### jacobi-fixed

```
$ ./jacobi-fixed cpu
[SYCL] Using device: [Intel(R) Core(TM) i7-7567U processor] from [Intel(R) OpenCL]
success; result is correct.
```

```
$ ./jacobi-fixed gpu
[SYCL] Using device: [Intel(R) Iris(R) Plus Graphics 650 [0x5927]] from [Intel(R) Level-Zero]
success; result is correct.
```
```
$ ./jacobi-fixed accelerator
[SYCL] Using device: [Intel(R) FPGA Emulation Device] from [Intel(R) FPGA Emulation Platform for OpenCL(TM) software]
success; result is correct.
```

#### jacobi-bugged

```
$ ./jacobi-bugged cpu
[SYCL] Using device: [Intel(R) Core(TM) i7-7567U processor] from [Intel(R) OpenCL]
fail; components of x_k are not close to 1.0
```

```
./jacobi-bugged gpu
[SYCL] Using device: [Intel(R) Iris(R) Plus Graphics 650 [0x5927]] from [Intel(R) Level-Zero]
fail; components of x_k are not close to 1.0
```

```
$ ./jacobi-bugged accelerator
[SYCL] Using device: [Intel(R) FPGA Emulation Device] from [Intel(R) FPGA Emulation Platform for OpenCL(TM) software]
fail; components of x_k are not close to 1.0
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
: Display information about the `inferiors`. For GPU offloading, one inferior
  represents the host process, and another (`gdbserver-gt`) represents the
  kernel.

`info threads <ID>`
: Display information about threads with id `ID`, including their active SIMD
  lanes. Omit id to display all threads.

`thread <thread_id>:<lane>`
: Switch context to the SIMD lane `lane` of the specified thread.
  E.g: `thread 2.6:4`

`thread apply <thread_id>:<lane> <cmd>`
: Apply command `cmd` to the specified lane of the thread.
  E.g.: `thread apply 2.3:* print element` prints `element` for each active lane
  of thread 2.3. Useful for inspecting vectorized values.

`x /<format> <addr>`
: Examine the memory at address `addr` according to `format`.
  E.g.: `x /i $pc` shows the instruction pointed by the program counter.
  `x /8wd &count` shows eight words in decimal format located at the address
  of `count`.

`set nonstop on/off`
: Enable/disable the nonstop mode. This command may **not** be used after the
  program has started.

`set scheduler-locking on/step/off`
: Set the scheduler locking mode.

`maint jit dump <addr> <filename>`
: Save the JIT'ed objfile that contains address `addr` into the file `filename`.
  Useful for extracting the DPC++ kernel when running on the CPU device.

`cond [-force] <N> <exp>`
: Define the expression `exp` as the condition for breakpoint `N`. Use the
  optional `-force` flag to force the condition to be defined even when `exp` is
  invalid for the current locations of the breakpoint. Useful for defining
  conditions involving JIT-produced artificial variables.
  E.g.: `cond -force 1 __ocl_dbg_gid0 == 19`.

---

\* Intel is a trademark of Intel Corporation or its subsidiaries. Other names
and brands may be claimed as the property of others.
