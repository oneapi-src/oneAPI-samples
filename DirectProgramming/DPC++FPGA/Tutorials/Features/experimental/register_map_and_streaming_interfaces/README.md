

# Register Map and Streaming Interfaces
This FPGA tutorial demonstrates how to specify the kernel invocation interfaces and kernel argument interfaces, and demonstrates the differences between streaming interfaces that use a ready/valid handshake, and register-mapped interfaces that exist in the kernel's control/status register (CSR).

The kernel invocation interface (namely, the `start` and `done` signals) can be implemented in the kernel's CSR, or using a ready/valid handshake. Similarly, the kernel arguments can be passed through the CSR, or through dedicated conduits. The invocation interface and any argument interfaces are specified independently, so you may choose to implement the invocation interface with a ready/valid handshake, and implement the kernel arguments in the CSR. All argument interfaces that are implemented as conduits will be synchronized to the ready/valid handshake of the kernel invocation interface. This means that it is not possible to configure a kernel with a register-mapped invocation interface and conduit arguments. The following table lists valid kernel argument interface synchronizations.

| Invocation Interface    | Argument Interface    | Argument Interface Synchronization       |
|----------------------|-----------------------|------------------------------------------|
| Streaming            | Streaming             | Synchronized with `start` and `ready_out` |
| Streaming            | Register mapped       | N/A                                      |
| Register mapped      | Streaming             | *No synchronization possible*            |
| Register mapped      | Register mapped       | N/A                                      |

> **Note**: Register mapped kernel arguments are not currently supported in kernels with a streaming invocation interface.

If you would like an argument to have its own dedicated ready/valid handshake, please implement that argument using a [Host Pipe](../hostpipes/).

> **Note**: The register map and streaming interface feature is only supported in the IP Component Authoring design flow. The IP Component Authoring design flow compiles SYCL* source code to standalone IPs that can be deployed into your Intel® Quartus® Prime projects. The generated IP is not meant to run on FPGA devices directly, therefore there will be no FPGA executables generated in this tutorial. Emulator and simulator executables are still generated to allow you to validate your IP. You can run the generated HDL through Intel® Quartus® Prime to generate accurate f<sub>MAX</sub> and area estimates.

***Documentation***:  The [DPC++ FPGA Code Samples Guide](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of DPC++ for FPGA. <br>
The [oneAPI DPC++ FPGA Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) is the reference manual for targeting FPGAs through DPC++. <br>
The [oneAPI Programming Guide](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/) is a general resource for target-independent DPC++ programming.

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Software                          | Intel® oneAPI DPC++/C++ Compiler
| What you will learn               | Basics of specifying kernel invocation interfaces and kernel argument interfaces
| Time to complete                  | 30 minutes

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE

## Purpose

Use register map and streaming interface annotations to specify how the kernel invocation handshaking is performed, as well as how the kernel argument data is passed in to the kernel.

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
  register_map_interface void operator()() const {
    ...
  }
};
```

*__Note__:* Register mapped kernel arguments are not currently supported in kernels with a streaming invocation interface.

### Declaring a streaming kernel argument interface

```c++
struct MyIP {
  conduit int arg1;
  register_map_interface void operator()() const {
    ...
  }
};
```

### Default Interfaces
If no annotation is specified for the kernel invocation interface, then a register map kernel invocation interface will be inferred by the compiler. If no annotation is specified for the a kernel argument, then that kernel argument will have the same interface as the kernel invocation interface. In the Lambda programming model, all kernel arguments will have the same interface as the kernel invocation interface.

### Testing the Tutorial
A total of four sources files are in the `src/` directory, declaring a total of four kernels. Two use the functor programming model, and the other use the lambda programming model. For each programming model, one kernel is declared with the register map kernel invocation interface and the other kernel is declared with the streaming kernel invocation interface.

```c++
struct FunctorRegisterMapIP {
  register_map ValueT *input;
  ValueT *output;
  conduit size_t n;
  register_map_interface void operator()() const {
    for (int i = 0; i < n; i++) {
      output[i] = SomethingComplicated(input[i]);
    }
  }
};
```
```c++
struct FunctorStreamingIP {
  conduit ValueT *input;
  conduit ValueT *output;
  size_t n;
  streaming_interface void operator()() const {
    for (int i = 0; i < n; i++) {
      output[i] = SomethingComplicated(input[i]);
    }
  }
};
```

These two functor kernels are invoked in the same way in the host code, by constructing the struct and submitting a `single_task` into the SYCL `queue`.

```c++
q.single_task(FunctorRegisterMapIP{in, functorRegisterMapOut, count}).wait();
```
```c++
q.single_task(FunctorStreamingIP{in, functorStreamingOut, count}).wait();
```

The two lambda kernels are annotated directly on the lambda function body and submitted into the SYCL `queue` as a `single_task`.

```c++
void TestLambdaRegisterMapKernel(sycl::queue &q, ValueT *in, ValueT *out, size_t count) {
  q.single_task<LambdaRegisterMapIP>([=] register_map_interface  {
    for (int i = 0; i < count; i++) {
      out[i] = SomethingComplicated(in[i]);
    }
  }).wait();
}
```
```c++
void TestLambdaStreamingKernel(sycl::queue &q, ValueT *in, ValueT *out, size_t count) {
  q.single_task<LambdaStreamingIP>([=] streaming_interface  {
    for (int i = 0; i < count; i++) {
      out[i] = SomethingComplicated(in[i]);
    }
  }).wait();
}
```

### Additional Documentation
- [Explore SYCL* Through Intel&reg; FPGA Code Samples](https://software.intel.com/content/www/us/en/develop/articles/explore-dpcpp-through-intel-fpga-code-samples.html) helps you to navigate the samples and build your knowledge of FPGAs and SYCL.
- [FPGA Optimization Guide for Intel&reg; oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide) helps you understand how to target FPGAs using SYCL and Intel&reg; oneAPI Toolkits.
- [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) helps you understand target-independent, SYCL-compliant programming using Intel&reg; oneAPI Toolkits.

## Key Concepts
* Basics of declaring kernel invocation interfaces and kernel argument interfaces

## Building the `register_map_and_streaming_interfaces` Tutorial

> __Note__: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
>
> * For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> * For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows*:
>
> * `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see __Use the setvars Script__ for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

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
   
   You can also compile for a custom FPGA platform. Run `cmake` using the following command:
   
   ```
   cmake .. -DFPGA_DEVICE=<device-family-name>
   ```
   or
   ```
   cmake .. -DFPGA_DEVICE=<device-part-name>
   ```
   For example, to target the default Intel® Stratix10® device:
   ```
   cmake .. -DFPGA_DEVICE=Stratix10
   ```
   To target a specific Intel® Arria10® device:
   ```
   cmake .. -DFPGA_DEVICE=10ax115s2f45i2sges
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
   You can also compile for a custom FPGA platform. Run `cmake` using the following command:
   
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<device-family-name>
   ```
   or
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<device-part-name>
   ```
   For example, to target the default Intel® Stratix10® device:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=Stratix10
   ```
   To target a specific Intel® Arria10® device:
   ```
   cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=10ax115s2f45i2sges
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

Locate `report.html` in the corresponding `<source_file>_report.prj/reports/` directory. Open the report in any of the following web browsers:  Chrome*, Firefox*, Edge*, or Internet Explorer*.

Open the **Views** menu and select **System Viewer**.

In the left-hand pane, select **FunctorRegisterMapIP** or **LambdaRegisterMapIP** under the System hierarchy for the kernels with a register-mapped invocation interface.

In the main **System Viewer** pane, the kernel invocation interfaces and kernel argument interfaces are shown. They show that the `start`, `busy` and `done` kernel invocation interfaces are implemented in register map interfaces, the `arg_input` and `arg_output` kernel arguments are implemented in register map interfaces. The `arg_n` kernel argument is implemented in a streaming interface in the **FunctorRegisterMapIP**, and in a register map interface in the **LambdaRegisterMapIP**.

Similarly, in the left-hand pane, select **FunctorStreamingIP** or **LambdaStreamingIP** under the System hierarchy for the kernels with a streaming invocation interface.

In the main **System Viewer** pane, the kernel invocation interface and kernel argument interfaces are shown. They show that the `start`, `done`, `ready_in` and `ready_out` kernel invocation interfaces are implemented in streaming interfaces, and the `arg_input`, `arg_output` and `arg_n` kernel arguments are all implemented in streaming interfaces.

## Running the Sample

 1. Run the sample on the FPGA emulator (the kernel executes on the CPU):
     ```
     ./register_map_functor_model.fpga_emu        (Linux)
     ./streaming_functor_model.fpga_emu           (Linux)
     ./register_map_lambda_model.fpga_emu         (Linux)
     ./streaming_lambda_model.fpga_emu            (Linux)
     register_map_functor_model.fpga_emu.exe      (Windows)
     streaming_functor_model.fpga_emu.exe         (Windows)
     register_map_lambda_model.fpga_emu.exe       (Windows)
     streaming_lambda_model.fpga_emu.exe          (Windows)
     ```
2. Run the sample on the FPGA simulator:
     ```
     ./register_map_functor_model.fpga_sim        (Linux)
     ./streaming_functor_model.fpga_sim           (Linux)
     ./register_map_lambda_model.fpga_sim         (Linux)
     ./streaming_lambda_model.fpga_sim            (Linux)
     register_map_functor_model.fpga_sim.exe      (Windows)
     streaming_functor_model.fpga_sim.exe         (Windows)
     register_map_lambda_model.fpga_sim.exe       (Windows)
     streaming_lambda_model.fpga_sim.exe          (Windows)
     ```

### Example of Output

```
Running the kernel with register map invocation interface implemented in the functor programming model
	 Done
PASSED
```

```
Running the kernel with streaming invocation interface implemented in the functor programming model
	 Done
PASSED
```

```
Running kernel with register map invocation interface implemented in the lambda programming model
	 Done
PASSED
```

```
Running kernel with streaming invocation interface implemented in the lambda programming model
	 Done
PASSED
```

### Example Simulation Waveform

The diagram below shows the example waveform generated by the simulator that you will see for the kernels with a register-mapped invocation interface. The waveform shows the register-mapped kernel arguments and kernel invocation handshaking signals are passed in through an Avalon agent interface, whose addresses are as specified in the agent memory map header files in the project directory.
![register_map_invocation_interface](assets/register_map_invocation_interface.png)

The diagram below shows the example waveform generated by the simulator that you will see for the kernels with a streaming invocation interface. The waveform shows the streaming kernel arguments and kernel invocation handshaking signals follow the Avalon-ST protocol. The streaming invocation interface consumes the kernel arguments on the clock cycle that the `start` and `ready_out` signals are asserted, and the kernel invocation is finished on the clock cycle that the `done` and `ready_in` signals are asserted.
![streaming_invocation_interface](assets/streaming_invocation_interface.png)

## License
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)