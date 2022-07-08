# `Mandelbrot` Sample

Mandelbrot is an infinitely complex fractal pattern that is derived from a
simple formula. This `Mandelbrot` sample demonstrates how to use SYCL*-compliant code for offloading computations to a GPU (or other devices) and further demonstrates how to optimize and improve processing time using parallelism.

| Property                       | Description
|:---                               |:---
| What you will learn               | How to offload the computation to GPU using the Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes

For comprehensive information in oneAPI programming, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). (Use search or the table of contents to find relevant information.)

## Purpose

This `Mandelbrot` sample is a SYCL-compliant application that generates a fractal image by
initializing a matrix of 512 x 512, where the computation at each point (pixel)
is entirely independent of the computation at other points. The sample includes
both parallel and serial calculations of the set, which allows for direct results comparison. The parallel implementation demonstrates the use of
Unified Shared Memory (USM) or buffers. You can modify parameters such as the
number of rows, columns, and iterations to evaluate the difference in
performance and load between USM and buffers.


## Prerequisites
| Property                       | Description
|:---                               |:---
| OS                                | Ubuntu* 18.04 <br>Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler

## Key Implementation Details

The program attempts first to run on an available GPU, and it will fall back to the system CPU if it does not detect a compatible GPU.

The program output displays the compilation device and elapsed render time for the Mandelbrot image, which helps compare different offload implementations based on the complexity of the computation.

The basic SYCL implementation explained in the code includes device selector,
buffer, accessor, kernel, and command groups.

## Build the `Mandelbrot` Sample

### Setting Environment Variables
For working with the Command-Line Interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by
sourcing the `setvars` script every time you open a new terminal window. This practice ensures your compiler, libraries, and tools are ready for development.

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
>For more information on environment variables, see "Use the setvars Script" for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

You can use [Modulefiles scripts](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-modulefiles-with-linux.html) to set up your development environment. The modulefiles scripts work with all Linux shells.

If you wish to fine tune the list of components and the version of those components, use
a [setvars config file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html) to set up your development environment.

### Include Files
The include folder is located on your development system at `%ONEAPI_ROOT%\dev-utilities\latest\include`.

### Using Visual Studio Code*  (Optional)
You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel&reg; oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel&reg; oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*
Perform the following steps:
1. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

   By default, the program creates executables for both USM and buffers. You can build the executables individually with the following:
   - Create buffers executable: `make mandelbrot`
   - Create USM executable: `make mandelbrot_usm`

If an error occurs, you can get more details by running `make` with the `VERBOSE=1` argument:
```
make VERBOSE=1
```

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the Diagnostics Utility for Intel&reg; oneAPI Toolkits, which provides system checks to find missing
dependencies and permissions errors. See [Diagnostics Utility for Intel&reg; oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).


### On Windows* Using Visual Studio*
- Build the program using VS2017 or later.
    - Right-click on the solution file and open using the IDE.
    - Right-click on the project in **Solution Explorer** and select **Rebuild**.
    - From the top menu, select **Debug** > **Start without Debugging**.

- Build the program using MSBuild.
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio version.
     - Run the following command: `MSBuild mandelbrot.sln /t:Rebuild /p:Configuration="Release"`


## Run the Sample
1. Run the program (default uses buffers).
    ```
    make run
    ```
   For USM instead of buffers, use `make run_usm`.

### Modifying Application Parameters

You can modify some parameters in `mandel.hpp`. Adjust the parameters to see how performance varies using the different offload techniques. The configurable parameters are:

|Parameter |Description
|:--- |:---
|`row_size` | Default is 512
|`col_size` |Default is 512
|`max_iterations` |Maximum value is 100.
|`repetitions` |Maximum value is 100.

> **Note**: If either the `col_size` or `row_size` values are below **128**, the output is limited to just text in the output window.

### Example Output
```
Platform Name: Intel(R) OpenCL HD Graphics
  Platform Version: OpenCL 2.1
       Device Name: Intel(R) Gen9 HD Graphics NEO
    Max Work Group: 256
 Max Compute Units: 24

Parallel Mandelbrot set using buffers.
Rendered image output to file: mandelbrot.png (output too large to display in text)
       Serial time: 0.0430331s
     Parallel time: 0.00224131s
Successfully computed Mandelbrot set.
```

## Build the `Mandelbrot` Sample in Intel&reg; DevCloud
If running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

1. Open a terminal on a Linux system.

2. Log in to the Intel&reg; DevCloud.
   ```
   ssh devcloud
   ```

3. Download the samples from GitHub.
   ```
   git clone https://github.com/oneapi-src/oneAPI-samples.git
   ```

4. Change directories to the sample directory.
   ```
   cd ~/oneAPI-samples/DirectProgramming/DPC++/CombinationalLogic/mandelbrot
   ```

### Build and Run the Sample in Batch Mode (optional)
The following instruction describe the optional process of submitting build and run jobs 
in a Portable Bash Script (PBS). A job is a script that is submitted to PBS through the qsub utility. By default, the qsub utility does not inherit the current environment variables or your current working directory. For this reason, it is necessary to submit jobs as scripts that handle the setup of the environment variables. In order to address the working directory issue, you can either use absolute paths or pass the `-d \<dir\>` option to qsub to set the working directory.

### Create the Job Scripts

1. Create a **build.sh** script in a text editor.
   ```
   nano build.sh
   ```

2. Add the following text to the **build.sh** file.
   ```
   source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Save and close the **build.sh** file.

4. Create a **run.sh** script in a text editor.
   ```
   nano run.sh
   ```

5. Add the following text to the **run.sh** file.
   ```
   source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
   cd build
   make run
   ```

6.	Save and close the **run.sh** file.

### Build and Run the Sample

Jobs submitted in batch mode are placed in a queue waiting for the necessary resources (compute nodes) to become available. The jobs are executed on a first-come, first-run basis on the first available node(s) having the requested property or label.

1. Build the sample on a GPU node.
   ```
   qsub -l nodes=1:gpu:ppn=2 -d . build.sh
   ```
   - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node to the job.
   - `-d .` configures the current folder as the working directory for the task.

2. Use the qstat utility to inspect the job progress.
   ```
   watch -n 1 qstat -n -1
   ```
   - The watch `-n 1` command is used to run `qstat -n -1` and display its results every second.

When the build job completes, there will be a **build.sh.oXXXXXX** file in the directory.

> **Note**: Some files are written to the disk when each job terminates.
>- **<script_name>.sh.eXXXX** = the job stderr
> - **<script_name>.sh.oXXXX** = the job stdout
> - where **XXXX** is the job ID printed to the screen after each qsub command.

3. Once the job completes, run the sample on a GPU node.
   ```
    qsub -l nodes=1:gpu:ppn=2 -d . run.sh
   ```
4. Inspect the output of the sample.
   ```
   cat run.sh.oXXXX
   ```
### Example Output in Intel&reg; DevCloud

You should see output similar to the following:
```
Platform Name: Intel(R) OpenCL HD Graphics
  Platform Version: OpenCL 2.1
       Device Name: Intel(R) Gen9 HD Graphics NEO
    Max Work Group: 256
 Max Compute Units: 24

Parallel Mandelbrot set using buffers.
Rendered image output to file: mandelbrot.png (output too large to display in text)
       Serial time: 0.0430331s
     Parallel time: 0.00224131s
Successfully computed Mandelbrot set.
```
5. Optionally, remove the stdout and stderr files and clean-up the project files.
   ```
   rm build.sh.*; rm run.sh.*; make clean
   ```
6. Disconnect from the Intel&reg; DevCloud.
   ```
   exit
   ```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).