# `Monte Carlo Pi` Sample
Monte Carlo Simulation is a broad category of computation that utilizes
statistical analysis to reach a result.

The `Monte Carlo Pi` sample uses the Monte Carlo Procedure to estimate the value of pi.

| Property            | Description
|:---                 |:---
| What you will learn | How to start using the SYCL* compliant reduction extension in the compiler
| Time to complete    | 15 minutes

## Purpose
By inscribing a circle of radius 1 inside a 2x2 square and then sampling a large number of random coordinates falling uniformly within the square, the value of pi can be estimated using the ratio of samples that fall inside the circle divided by the total number of samples.

This method of estimation works for calculating pi because the expected value of the sample ratio is equal to the ratio of the area of a circle divided by the area of a square, so a circle of radius 1 has an area of pi units squared, while a 2x2 square has an area of 4 units squared, yielding a ratio of pi/4. Therefore, to estimate the value of pi, our solution will be four times the sample ratio.

The Monte Carlo procedure for estimating pi is easily parallelized, as each calculation of a random coordinate point can be considered a discrete work item. The computations involved with each work item are entirely independent of one another except for in summing the total number of points inscribed within the circle. This code sample demonstrates how to utilize the SYCL compliant reduction extension in the compiler for this purpose.

The code attempts to execute on an available GPU and will fall back to the system CPU if a compatible GPU is not detected. The device used for the compilation is displayed in the output, along with the elapsed time to complete the computation. 

## Prerequisites
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br>Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel® oneAPI DPC++/C++ Compiler

## Key Implementation Details
The basic SYCL standard implementation explained in the code includes device selector, buffer, accessor, kernel, and reduction.

## Build the `Monte Carlo Pi` Program for CPU and GPU
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Include Files
The include folder is at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system. You might need to use some of the resources from this location to build the sample.

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

### On Linux*
1. Build the program.
    ```
    mkdir build
    cd build
    cmake ..
    make
    ```
If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
### On Windows* 
**Using Visual Studio***

Build the program using **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. From the top menu, select **Debug** > **Start without Debugging**.

**Using MSBuild**
1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild MonteCarloPi.sln /t:Rebuild /p:Configuration="Release"`

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `Monte Carlo Pi` Sample
### Application Parameters
The sample has several configurable parameters in the `monte_carlo_pi.cpp` source file.

|Parameter                  |Description
|:---                       |:---
|`size_n`   | Defines the sample size for the Monte Carlo procedure. 
|`size_wg`  | Defines the size of workgroups inside the kernel code. The number of workgroups is calculated by dividing `size_n` by `size_wg`, so size_n must be greater than or equal to `size_wg`. Increasing `size_n` will increase computation time as well as the accuracy of the pi estimation. Changing `size_wg` will have different performance effects depending on the device used for offloading.
|`img_dimensions` | Defines the size of the output image for data visualization.
|`circle_outline`| Defines the thickness of the circular border in the output image for data visualization. Setting it to zero will remove it entirely.

### On Linux
1. Run the program.
    ```
    make run
    ```
2. Clean the program. (Optional)
    ```
    make clean
    ```
### On Windows
1. Change to the output directory.
2. Run the executable.
   ```
   MonteCarloPi.exe
   ```
### Run the `Monte Carlo Pi` Sample in Intel® DevCloud
If running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

## Example Output
In addition to displaying output similar to the following, the program creates a rendered image plot of the computation to a file: `MonteCarloPi.bmp`.

```
Calculating estimated value of pi...

Running on Intel(R) Gen9 HD Graphics NEO
The estimated value of pi (N = 10000) is: 3.15137

Computation complete. The processing time was 0.446072 seconds.
The simulation plot graph has been written to 'MonteCarloPi.bmp'
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).