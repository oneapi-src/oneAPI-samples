# Diamond Dependency Sample

Code example demonstrating the usage of [`sycl_ext_oneapi_graph`](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc) extension with queue recording API.

| Area                      | Description
|:---                       |:---
| What you will learn       | How to use SYCL-Graphs extension for optimizing kernel execution.
| Time to complete          | 15 minutes


## Purpose

This code example shows how to record commands submitted to a SYCL `queue` into a `command_graph` object. Once the graph recording is complete, the graph is finalized which means a new `command_graph` object in `graph_state::executable` is created which is ready for submission. Lastly, the graph is submitted en bloc for execution to a queue with a new function `ext_oneapi_graph()`. And can be replayed as many times as needed.



## Prerequisites
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* <br>Windows* 10, 11
| Hardware                          | Intel GPU
| Software                          | Intel® oneAPI DPC++/C++ Compiler


## Key Implementation Details

Key SYCL* concepts demonstrated in the code sample include using command graph extension with buffers and accessors. This sample is using queue recording API.

>**Note**: For comprehensive information about oneAPI programming, see the *[Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/overview.html)*. (Use search or the table of contents to find relevant information quickly.)


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

> **Note**: You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

> **Note**: If you want only specific components or versions of those components, use a [setvars config file](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.


## Build the `Diamond Dependency` Sample for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables or if you have a Unified Directory Layout, see
*[Use the setvars and oneapi-vars Scripts with Linux*](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/use-the-setvars-script-with-linux-or-macos.html)* or *[Use the setvars and oneapi-vars Scripts with Windows*](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/use-the-setvars-script-with-windows.html)*.

### On Linux*

The project uses a standard CMake build configuration system. Ensure the SYCL compiler is used by the configuration either by setting the environment variable `CXX=<compiler>` or passing the configuration flag
`-DCMAKE_CXX_COMPILER=<compiler>` where `<compiler>` is your SYCL compiler's
executable (for example Intel `icpx` or LLVM `clang++`).

1. Change to the sample directory.
2. Build the program.
   ```
   mkdir -p build && cd build
   cmake .. -DCMAKE_CXX_COMPILER=<compiler>
   cmake --build .
   ```

The CMake configuration automatically detects the available SYCL backends and
enables the SPIR/CUDA/HIP targets for the device code, including the corresponding
architecture flags. If desired, these auto-configured cmake options may be overridden
with the following ones:

| OPTION                     | VALUE
|:---                        |:---
| ENABLE_SPIR                | ON or OFF
| ENABLE_CUDA                | ON or OFF
| ENABLE_HIP                 | ON or OFF
| CUDA_COMPUTE_CAPABILITY    | Integer, e.g. `70` meaning capability 7.0 (arch `sm_70`)
| HIP_GFX_ARCH               | String, e.g. `gfx1030`

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/docs/oneapi/user-guide-diagnostic-utility/current/overview.html)* for more information on using the utility.


## Run the `Diamond Dependency` Sample

### On Linux

1. Run the program.
   ```
   ./diamondDependency
   ```

## License

Code samples are licensed under the MIT license. See [License.txt](License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
