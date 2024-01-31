# `nstreams_device_selection` Sample

The `nstreams_device_selection` sample demonstrates how to use the Intel® oneAPI Base Toolkit (Base Kit) and Intel® oneAPI DPC++ Library (oneDPL) found in the Base Kit to apply device selection policies using a simple application based on nstreams.

For comprehensive instructions, see the [Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/overview.html) and search based on relevant terms noted in the comments.

| Property                          | Description
|:---                               |:---
| What you will learn               | How to offload the computation to specific devices and use policies to different dynamic offload strategies.
| Time to complete                  | 30 minutes

## Purpose

This sample performs a simple element-wise parallel computation on three vectors: `A`, `B` and `C`.  For each element `i`, it computes `A[i] += B[i] + scalar * C[i]`. Additional information can be found on the [Optimizing Memory Bandwidth on Stream Triad](https://www.intel.com/content/www/us/en/developer/articles/technical/optimizing-memory-bandwidth-on-stream-triad.html) page. This sample starts with a simple implementation of device offload using SYCL*. The second version of the code shows how to introduce Dynamic Device Selection and uses device specific policies that can be selected by supplying different arguments when invoking the application.

The sample includes two different versions of the nstreams project:
1. `1_nstreams_sycl.cpp`: basic SYCL implementation; creates a kernel that targets the system's CPU.
2. `2_nstreams_policies.cpp`: version of the sample that includes five policies:
    1. Static CPU
    2. Static GPU
    3. Round Robin policy CPU/GPU
    4. Dynamic Load policy CPU/GPU
    5. Auto Tune policy CPU/GPU

The varying policies are helpful as follows:
1. **Fixed CPU:** This is the simplest implementation. It can be helpful to start implementations using fixed CPU since any debug or troubleshooting will be considerably easier.
2. **Fixed GPU:** This an incremental step that simply designates the offload kernel to run on the GPU, isolating functionality to help triage any problems that may arise when targeting the GPU.
3. **Round Robin:** Assigns the function to the next available device as specified in the "universe". The capability is particularly beneficial in *multi-GPU systems*. Note that performance benefits may not be realized on single GPU platforms but will scale accordingly on multi-GPU systems.
4. **Dynamic Load** selects the device that has the most available capacity at that moment based on the number of unfinished submissions. This can be useful for offloading kernels of varying cost to devices of varying performance.
5. **Auto-tune** performs run-time profile sampling of the performance of the kernel on the available devices before selecting a final device to use. The choice is made based on runtime performance history, so this policy is only useful for kernels that have stable performance.

[Detailed Descriptions of the Policies](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/current/policies.html) are available in the Intel® oneAPI DPC++ Library Developer Guide and Reference.

> NOTE: Given the simplicity of this example, performance benefits may not be gained depending on the available devices.

Dynamic Device Selection support customization to allow frameworks or application developers to define custom logic for making device selections. Complete reference documentation is available in the [oneAPI DPC++ Library Developer Guide](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-2/overview.html).

## Key Implementation Details

The basic SYCL standards implemented in the code include the use of the following:
- Fixed (CPU and GPU) policies.
- Dynamic policies Round Robin, Load, and Auto-tune.
- Basic structure: header, namespace, define universe, setup policies, wrap kernel, and return event. **Note: a return event is required for all Dynamic Device Selection usage.**

## Building the `nstreams_device_selection` Program for CPU and GPU

### Setting Environment Variables
When working with the Command Line Interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI environment by sourcing the `setvars` script located in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> Microsoft Visual Studio:
> - Open a command prompt window and execute `setx SETVARS_CONFIG " "`. This only needs to be set once and will automatically execute the `setvars` script every time Visual Studio is launched.
>
>For more information on environment variables, see "Use the setvars Script" for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

If you wish to fine tune the list of components and the version of those components, use
a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-vs-code/current/overview.html).

### On Linux*
Perform the following steps:
1. Build the program using the following `cmake` commands.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

2. Run the program.
   ```
   $ make run_all
   ```
   > **Note**: by default, only CPU devices are run.  Use ``sycl-ls`` to see available devices on your target system.

   Manually envoking the application requires supplying a vector length. 1000 is used in the examples below.

   For the basic SYCL implementation:
   ```
   $  ./1_nstreams_sycl 1000
   ```

    For Dynamic Device Selection, usage:  ./2_nstreams_policies 1000 <policy>. For example, Fixed Resource Policy (CPU):
    ```
   $  ./2_nstreams_policies 1000 1
    ```   

    | Arg | Dynamic Device Selection Policy
    |:--- |:---
    | 1   | Fixed Resource Policy (CPU)
    | 2   | Fixed Resource Policy (GPU)
    | 3   | Round Robin Policy
    | 4   | Dynamic Load Policy
    | 5   | Auto Tune Policy

3. Clean the program. (Optional).
   ```
   make clean
   ```

If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
```
make VERBOSE=1
```

### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits, which provides system checks to find missing dependencies and permissions errors. See [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


### On Windows* Using Visual Studio* Version 2019 or Newer

- Build the program using VS2019 or VS2022
    - Right-click on the solution file and open using either VS2019 or VS2022 IDE.
    - Right-click on the project in Solution Explorer and select Set as Startup Project.
    - Select the correct correct configuration from the drop down list in the top menu (5_GPU_optimized has more arguments to choose)
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.
> **Note**: Remember to use Release mode for better performance.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2019" or "x64 Native Tools Command Prompt for VS2022"
     - Run the following command: `MSBuild "nstreams_device_selection.sln" /t:Rebuild /p:Configuration="Release"`

### Application Parameters

You can run individual nstream executables and modify parameters from the command line.

`./<executable_name> <vector length> <policy>`

For example:

```
$ ./2_nstreams_policy 1000 2
```
Where:

    vector length          : The size of the A, B and C vectors.
    Policy                 : Specifies the dynamic device selection policy (only valid for 2_nstreams_policy).

## Example Output

```
Using Static Policy (CPU) to iterate on CPU device with vector length: 10000
11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
...
11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz

 Rate: larger better     (MB/s): 120.042
 Avg time: lower better  (ns):   1.33287e+06
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
