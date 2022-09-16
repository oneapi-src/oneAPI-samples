# `Base: Vector Add` Sample

The `Base: Vector Add` is the equivalent of a ‘Hello, World!’ sample for data parallel programs. Building and running this sample verifies that your development environment is set up correctly, and the sample code demonstrates some core features of SYCL*.

| Area                     | Description
|:---                      |:---
| What you will learn      | How to begin using SYCL* to offload computations to a GPU
| Time to complete         | 15 minutes

## Purpose
The `Base: Vector Add` is a simple program that adds two large vectors of integers and verifies the results. This program uses C++ and SYCL* for Intel® CPU and accelerators.

In this sample, you can learn how to use C++ code to offload computations to a GPU. This includes using Unified Shared Memory (USM) and buffers. USM requires an explicit wait for the asynchronous kernel's computation to complete. Buffers, at the time they go out of scope, bring main memory in sync with device memory implicitly; the explicit wait on the event is not required as a result. This sample provides examples of both implementations for simple side-by-side reviews (the Windows sample only supports USM).

A detailed code walkthrough can be found in the [Explore SYCL* with Samples from
Intel](https://software.intel.com/content/www/us/en/develop/documentation/explore-dpcpp-samples-from-intel/top.html#top_STEP1_VECTOR_ADD)
guide.

> **Note**: For comprehensive information about oneAPI programming, see the [Intel® oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). (Use search or the table of contents to find relevant information quickly.)

## Prerequisites
| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br> Windows* 10
| Hardware                          | Skylake with GEN9 or newer <br>Intel® Programmable Acceleration Card with Intel® Arria&reg; 10 GX FPGA
| Software                          | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
The basic SYCL* implementation explained in the code includes device selector, USM, buffer, accessor, kernel, and command groups.

The code attempts to execute on an available GPU and fallback to the system CPU if a compatible GPU is not detected. If successful, the name of the offload device and a success message is displayed, which indicates your development environment is set up correctly.

In addition, you can target an FPGA device using the build scripts described below. If you do not have FPGA hardware, the sample will run in emulation mode, which includes static optimization reports for design analysis.

### Known Issues
With oneAPI 2021.4 the argument for accessors was changed from `noinit` to `no_init`. The change was derived from a change between the SYCL 2020
provisional spec and that of the 2020Rev3 spec.

If this sample fails to run, do one of the following:
- Update the Intel® oneAPI Base Toolkit to 2021.4 or later.
- Change the `no_init` argument  to `noinit`.

## Setting Environment Variables
When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Base: Vector Add` Sample for GPU and FPGA
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Running Samples in DevCloud 
If running a sample in the Intel DevCloud,
remember that you must specify the compute node (cpu, gpu, fpga_compile, or
fpga_runtime) and whether to run in batch or interactive mode.

For specific instructions, jump to [Run the sample in the DevCloud](#run-on-devcloud)

For more information see the
[Intel® oneAPI Base Toolkit Get Started Guide](https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

### Using Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
3. Open a terminal in VS Code (**Terminal > New Terminal**).
4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux* for CPU and GPU 
1. Build the program.
    ```
    make build_usm
    ```
> **Note**: To build everything, use `make all`.

### On Linux* for FPGA
1.  Build for FPGA emulation using the following commands:
    ```
    make fpga_emu -f Makefile.fpga
    ```
2. Build for FPGA hardware. (Compiling for hardware can take a long
   time.)
    ```
    make hw -f Makefile.fpga
    ```
3. Generate static optimization reports for design analysis. (The path to the
    reports is `vector-add_report.prj/reports/report.html`.)
    ```
    make report -f Makefile.fpga
    ```
### On Windows* for CPU and GPU
1. Open the **Intel oneAPI Command Prompt**.
2. Build the program.
    ```
    nmake -f Makefile.win build_usm
    ```
> **Note**: To build everything, use `nmake -f Makefile.win`

### On Windows for FPGA Emulation Only
> **Note** On Windows*, you can  compile and run on the FPGA 
emulator only. Generating optimization reports and compiling or running on
the FPGA hardware is not supported.

1. Open the **Intel oneAPI Command Prompt**.

2. Build the program.
   ```
   nmake -f Makefile.win.fpga
   ```
### On Windows Using Visual Studio* 2017 or Newer
1. Change to the sample directory.
2. Launch Visual Studio*.
3. Select the menu sequence **File** > **Open** > **Project/Solution**.
4. Select the `vector-add.sln` file.
5. For CPU and GPU, skip to Step 7 (below).
6. For FPGA emulation only, select the configuration **Debug-fpga**, which contains the settings shown in below. Alternatively, confirm the following settings from the **Project Property** dialog.

   a. Select the **DPC++** tab.

   b. **General** > **Perform ahead of time compilation for the FPGA** is set to **Yes**.

   c. **Preprocessor** > **Preprocessor Definitions** contains **FPGA_EMULATOR=1**.

 7. Select **Project** > **Build** menu option to build the selected
   configuration.
 8. Select **Debug** > **Start Without Debugging** menu option to run the
   program.

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the Sample
### On Linux for CPU and GPU
1. Run the program.
    ```
    make run_usm
    ```
> **Note**: To run everything, use `make run`.

### On Linux for FPGA
1.  Run for FPGA emulation.
    ```
    make run_emu -f Makefile.fpga
    ```
2. Run on FPGA hardware. 
    ```
    make run_hw -f Makefile.fpga
    ```
### On Windows for CPU and GPU
1. Open the **Intel oneAPI Command Prompt**.
3. Run the program using:
    ```
    nmake -f Makefile.win run_usm
    ```
> **Note**: To run everything, use `nmake -f Makefile.win run`

## On Windows for FPGA Emulation
1. Open the **Intel oneAPI Command Prompt**.
2. Build the program.
   ```
   nmake -f Makefile.win.fpga run
   ```
### Run the `Base: Vector Add` Sample in Intel® DevCloud (Optional)
When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

### Application Parameters
There is an optional parameter which determines vector size. Default value is `10000`.

## Example Output
```
Running on device:        Intel(R) Gen(R) HD Graphics NEO
Vector size: 10000
[0]: 0 + 0 = 0
[1]: 1 + 1 = 2
[2]: 2 + 2 = 4
...
[9999]: 9999 + 9999 = 19998
Vector add successfully completed on device.
```

### Running the sample in the DevCloud<a name="run-on-devcloud"></a>

#### Build and run

To launch build and run jobs on DevCloud submit scripts to PBS through the qsub utility.
> Note that all parameters are already specified in the build and run scripts.

1. Build the sample on a gpu node.

    ```bash
    qsub build.sh
    ```

2. When the build job completes, there will be a `build.sh.oXXXXXX` file in the directory. After the build job completes, run the sample on a gpu node:

    ```bash
    qsub run.sh
    ```

3. To build and run for FPGA emulator use accordingly the `build_fpga_emu.sh` and `run_fpga_emu.sh` scripts, for FPGA hardware use the `build_fpga.sh` and `run_fpga.sh` scripts.

#### Additional information

1. In order to inspect the job progress, use the qstat utility.

    ```bash
    watch -n 1 qstat -n -1
    ```

    > Note: The watch `-n 1` command is used to run `qstat -n -1` and display its results every second.
2. When a job terminates, a couple of files are written to the disk:

    <script_name>.sh.eXXXX, which is the job stderr

    <script_name>.sh.oXXXX, which is the job stdout

    > Here XXXX is the job ID, which gets printed to the screen after each qsub command.
3. To inspect the output of the sample use cat command.

    ```bash
    cat run.sh.oXXXX
    ```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).