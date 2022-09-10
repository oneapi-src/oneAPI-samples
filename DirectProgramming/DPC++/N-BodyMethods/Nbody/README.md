# `Nbody` sample
An N-body simulation is a simulation of a dynamical system of particles, usually under the influence of physical forces, such as gravity. This `Nbody` sample code uses SYCL* standards for CPUs and GPUs.

| Property             | Description
|:---                  |:---
| What you will learn  | How to offload complex computations to GPU
| Time to complete     | 15 minutes

## Purpose
By default, the `Nbody` sample code simulates 16,000 particles over ten integration steps. The position, velocity, and acceleration parameters of each particle are dependent on other (N-1) particles.

This algorithm is highly data parallel, so the code a good candidate to offload to GPU. The code demonstrates how to deal with multiple device kernels, which can be enqueued into a SYCL queue for execution, and the code demonstrates how to handle parallel reductions.

## Prerequisites
| Optimized for     | Description
|:---               |:---
| OS                | Ubuntu* 18.04 <br> Windows* 10
| Hardware          | Skylake with GEN9 or newer
| Software          | Intel® oneAPI DPC++ Compiler

## Key Implementation Details
The basic SYCL* compliant implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

## Set Environment Variables
When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Build the `Nbody` Program for CPU and GPU
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

### Use Visual Studio Code* (VS Code) (Optional)
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
**Using Visual Studio**

Build the program using **Visual Studio 2017** or newer.
1. Change to the sample directory.
2. Right-click on the solution file and open the solution in the IDE.
3. From the top menu, select **Debug** > **Start without Debugging**.

**Using MSBuild**

1. Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019" or whatever is appropriate for your Visual Studio* version.
2. Change to the sample directory.
3. Run the following command: `MSBuild Nbody.sln /t:Rebuild /p:Configuration="Release"`

#### Troubleshooting
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Run the `Nbody` Program
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
   Nbody.exe
   ```

### Run 'Nbody' Sample in Intel® DevCloud (Optional)
When running a sample in the Intel® DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel® oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

#### Build and Run Samples in Batch Mode (Optional)
You can submit build and run jobs through a Portable Bash Script (PBS). A job is a script that submitted to PBS through the `qsub` utility. By default, the `qsub` utility does not inherit the current environment variables or your current working directory, so you might need to submit jobs to configure the environment variables. To indicate the correct working directory, you can use either absolute paths or pass the `-d \<dir\>` option to `qsub`. 

If you choose to use scripts, jobs terminate with writing a couple of files to the disk:
- `<script_name>.sh.eXXXX` is the job stderr
- `<script_name>.sh.oXXXX` is the job stdout

  Here `XXXX` is the job ID, which gets printed to the screen after each `qsub` command. 

You can inspect output of the sample.
```
cat run.sh.oXXXX
```
#### Build and Run on Intel® DevCloud
1. Open a terminal on a Linux* system.
2. Log in to the Intel® DevCloud.
   ```
   ssh devcloud
   ```
3. Download the samples from GitHub.
   ```
   git clone https://github.com/oneapi-src/oneAPI-samples.git
   ```
4. Change to the sample directory.
5. Configure the sample for a GPU node. (This is a single line script.)
	```
	qsub  -I  -l nodes=1:gpu:ppn=2 -d .
	```
   - `-I` (upper case I) requests an interactive session.
   - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node. 
   - `-d .` makes the current folder as the working directory for the task.

> **Note**: To inspect job progress, use the qstat utility.
>   ```
>   watch -n 1 qstat -n -1
>   ```
> The command displays the results every second. The job is complete when no new results display.

6. Perform build steps you would on Linux. (Including optionally cleaning the project.)
7. Run the sample.
8. Disconnect from the Intel® DevCloud.
	```
	exit
	```

## Example Output
### Application Parameters
You can modify the `NBody` sample simulation parameters in `GSimulation.cpp`. Configurable parameters include:

|Parameter       | Defaults
|:---            |:---
|`set_npart`     | Default number of particles is **16000**
|`set_nsteps`    | Default number of integration steps is **10**
|`set_tstep`     | Default time delta is **0.1**
|`set_sfreq`     | Default sample frequency is **1**

### Example Output on Linux
```
===============================
Initialize Gravity Simulation
Target Device: Intel(R) Gen9
nPart = 16000; nSteps = 10; dt = 0.1
------------------------------------------------
s       dt      kenergy     time (s)    GFLOPS
------------------------------------------------
1       0.1     26.405      0.28029     26.488
2       0.2     313.77      0.066867    111.03
3       0.3     926.56      0.065832    112.78
4       0.4     1866.4      0.066153    112.23
5       0.5     3135.6      0.065607    113.16
6       0.6     4737.6      0.066544    111.57
7       0.7     6676.6      0.066403    111.81
8       0.8     8957.7      0.066365    111.87
9       0.9     11587       0.066617    111.45
10      1       14572       0.06637     111.86

# Total Time (s)     : 0.87714
# Average Performance : 112.09 +- 0.56002
===============================
Built target run
```
### Example Output on Intel® DevCloud
```
Scanning dependencies of target run
===============================
 Initialize Gravity Simulation
 nPart = 16000; nSteps = 10; dt = 0.1
------------------------------------------------
 s       dt      kenergy     time (s)    GFLOPS
------------------------------------------------
 1       0.1     26.405      0.43625     17.019
 2       0.2     313.77      0.02133     348.07
 3       0.3     926.56      0.021546    344.59
 4       0.4     1866.4      0.02152     345
 5       0.5     3135.6      0.021458    346
 6       0.6     4737.6      0.021434    346.38
 7       0.7     6676.6      0.02143     346.45
 8       0.8     8957.7      0.021482    345.6
 9       0.9     11587       0.021293    348.68
 10      1       14572       0.021324    348.16

# Total Time (s)     : 0.62911
# Average Performance : 346.36 +- 1.3384
===============================
Built target run
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).