

# Register Map and streaming interfaces
This FPGA tutorial demonstrates how to specify the kernel control interfaces and kernel argument interfaces. The kernel control interfaces can be implemented as register map interfaces or streaming interfaces. Similarly, the kernel argument interfaces can be implemented as register map interfaces or streaming interfaces as well, independent of which interface that the kernel control is using. The register map interface is referring to an interface registered in the kernel agent memory map, where the streaming interface is referring to an interface that signals are implemented in simple conduits.

> **Note**: The register map and streaming interface control feature is only supported in the IP Component Authoring design flow. The IP Component Authoring design flow compiles SYCL source code to sandalone IPs that can be deployed into user's systems. The generated IP is not meant to be ran on FPGA devices directly, therefore there will be no FPGA executables generated in this tutorial.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04, RHEL*/CentOS* 8, SUSE* 15; Windows* 10
| Software                          | Intel® oneAPI DPC++ Compiler <br> Intel® FPGA Add-On for oneAPI Base Toolkit
| What you will learn               | Basics of specifying kernel control interfaces and kernel argument interfaces
| Time to complete                  | 30 minutes

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow.
>
> For using the simulator flow, one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE

## Purpose

Use register map and streaming interface controls to specify how the kernel control handshaking is performed, as well as how the kernel argument data is passed in to the kernel.

### Declaring a register map kernel interface

#### Example Functor

```c++
struct MyIP {
  MyIP() {}
  register_map_interface void operator()() const {
    ...
  }
};
```

#### Example Lambda
```c++
q.single_task([=] register_map_interface {
  ...
})
```

### Declaring a streaming kernel interface

#### Example Functor

```c++
struct MyIP {
  MyIP() {}
  streaming_interface void operator()() const {
    ...
  }
};
```

#### Example Lambda
```c++
q.single_task([=] streaming_interface {
  ...
})
```

### Declaring a register map kernel argument interface

#### Example Functor

```c++
struct MyIP {
  register_map int arg1;
  MyIP(int arg1_) : arg1(arg1_) {}
  register_map_interface void operator()() const {
    ...
  }
};
```

*__Note__:* Register map kernel arguments are not currently supported in kernels with streaming control.

### Declaring a streaming kernel argument interface

```c++
struct MyIP {
  conduit int arg1;
  MyIP(int arg1_) : arg1(arg1_) {}
  register_map_interface void operator()() const {
    ...
  }
};
```

### Default Interfaces
If no annotation is specified for the kernel control, then a register map kernel control interface will be inferred by the compiler. If no annotation is specified for the a kernel argument, then that kernel argument will have the same interface as the kernel control interface. In the Lambda programming model, all kernel arguments will have the same interface as the kernel control interface.

### Testing the Tutorial
In `register_map_and_streaming_interfaces.cpp`, a total of four kernels are declared. Two use the functor programming model, and the other use the lambda programming model. For each programming model, one kernel is declared with the register map kernel control interface and the other kernel is declared with the streaming kernel control interface.

```c++
struct FunctorRegisterMapControlIP {
  register_map ValueT *input;
  ValueT *output;
  conduit size_t n;
  FunctorRegisterMapControlIP(ValueT *in_, ValueT *out_, size_t N_)
      : input(in_), output(out_), n(N_) {}
  register_map_interface void operator()() const {
    for (int i = 0; i < n; i++) {
      output[i] = SomethingComplicated(input[i]);
    }
  }
};
struct FunctorStreamingControlIP {
  conduit ValueT *input;
  conduit ValueT *output;
  size_t n;
  FunctorStreamingControlIP(ValueT *in_, ValueT *out_, size_t N_)
      : input(in_), output(out_), n(N_) {}
  streaming_interface void operator()() const {
    for (int i = 0; i < n; i++) {
      output[i] = SomethingComplicated(input[i]);
    }
  }
};
```

These two functor kernels are invoked in the same way in the host code, by constructing the struct and submitting a `single_task` into the SYCL `queue`.

```c++
template <typename KernelType>
void TestFunctorKernel(sycl::queue& q, ValueT* in, ValueT* out, size_t count) {
  q.single_task(KernelType{in, out, count}).wait();
}
```

The two lambda kernels are annotated directly on the lambda function body and submitted into the SYCL `queue` as a `single_task`.

```c++
void TestLambdaRegisterMapControlKernel(sycl::queue &q, ValueT *in, ValueT *out, size_t count) {
  q.single_task<LambdaRegisterMapControlIP>([=] register_map_interface  {
    for (int i = 0; i < count; i++) {
      out[i] = SomethingComplicated(in[i]);
    }
  }).wait();
}

void TestLambdaStreamingControlKernel(sycl::queue &q, ValueT *in, ValueT *out, size_t count) {
  q.single_task<LambdaStreamingControlIP>([=] streaming_interface  {
    for (int i = 0; i < count; i++) {
      out[i] = SomethingComplicated(in[i]);
    }
  }).wait();
}
```

## Key Concepts
* Basics of declaring kernel control interfaces and kernel argument interfaces

## Building the `register_map_and_streaming_interfaces` Tutorial

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: `. /opt/intel/oneapi/setvars.sh`
>
> Linux User: `. ~/intel/oneapi/setvars.sh`
>
> Windows: `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System

1. Generate the Makefile by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® Arria10® FPGA family, run `cmake` using the following command:

   ```
   cmake ..
   ```
   
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the following command:
   
   ```
   cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
      ```
      make fpga_emu
      ```
   * Generate the optimization report:
     ```
     make report
     ```
   * Compile for FPGA simulator
     ```
     make fpga_sim
     ```

### On a Windows* System

1. Generate the `Makefile` by running `cmake`.
     ```
   mkdir build
   cd build
   ```
   To compile for the Intel® Arria10® FPGA family, run `cmake` using the following command:

   ```
   cmake -G "NMake Makefiles" .. 
   ```
   You can also compile for a custom FPGA platform. Ensure that the board support package is installed on your system. Then run `cmake` using the command:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
   ```

2. Compile the design through the generated `Makefile`. The following build targets are provided, matching the recommended development flow:

   * Compile for emulation (fast compile time, targets emulated FPGA device):
     ```
     nmake fpga_emu
     ```
   * Generate the optimization report:
     ```
     nmake report
     ```
   * Compile for FPGA simulator:
     ```
     nmake fpga_sim
     ```

>**Tip**: If you encounter issues with long paths when compiling under Windows*, you might have to create your ‘build’ directory in a shorter path, for example `c:\samples\build`.  You can then run `cmake` from that directory, and provide `cmake` with the full path to your sample directory.

### Troubleshooting

If an error occurs, get more details by running `make` with
the `VERBOSE=1` argument:
``make VERBOSE=1``
For more comprehensive troubleshooting, use the Diagnostics Utility for
Intel® oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors.
[Learn more](https://software.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

 ### In Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs]([https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-oneapi-dpcpp-fpga-workflow-on-ide.html))

## Examining the Reports

Locate `report.html` in the `register_map_and_streaming_interfaces_report.prj/reports/` directory. Open the report in any of the following web browsers:  Chrome*, Firefox*, Edge*, or Internet Explorer*.

Open the **Views** menu and select **System Viewer**.

In the left-hand pane, select **FunctorRegisterMapControlIP** or **LambdaRegisterMapControlIP** under the System hierarchy.

In the main **System Viewer** pane, the kernel control interface and kernel argument interfaces are shown. They show that the `start`, `busy` and `done` kernel control interfaces are implemented in register map interfaces, the `arg_input` and `arg_output` kernel arguments are implemented in register map interfaces. The `arg_n` kernel argument is implemented in streaming interface in the **FunctorRegisterMapControlIP**, and in register map interface in the **LambdaRegisterMapControlIP**.

Similarly, in the left-hand pane, select **StreamingControlIP** or **LambdaStreamingControlIP** under the System hierarchy.

In the main **System Viewer** pane, the kernel control interface and kernel argument interfaces are shown. They show that the `start`, `done`, `ready_in` and `ready_out` kernel control interfaces are implemented in streaming interfaces, and the `arg_input`, `arg_output` and `arg_n` kernel arguments are all implemented in streaming interfaces.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./register_map_and_streaming_interfaces.fpga_emu     (Linux)
     register_map_and_streaming_interfaces.fpga_emu.exe   (Windows)
     ```
2. Run the sample on the FPGA simulator:
     ```
     ./register_map_and_streaming_interfaces.fpga_sim      (Linux)
     register_map_and_streaming_interfaces.fpga_sim.exe        (Windows)
     ```

### Example of Output

```
Running the kernel with streaming control implemented in the functor programming model
	 Done

Running the kernel with register map control implemented in the functor programming model
	 Done

Running kernel with streaming control implemented in the lambda programming model
	 Done

Running kernel with register map control implemented in the lambda programming model
	 Done

PASSED
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)