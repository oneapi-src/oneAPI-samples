# `Nbody` sample
An N-body simulation is a simulation of a dynamical system of particles, usually under the influence of physical forces, such as gravity. This nbody sample code is implemented using C++ and DPC++ language for Intel CPU and GPU.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04; Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler;
| What you will learn               | How to offload the computation to GPU using Intel&reg; oneAPI DPC++ Compiler
| Time to complete                  | 15 minutes

## Purpose
Nbody sample code simulates 16000 particles and for ten integration steps. Each particle's position, velocity and acceleration parameters are dependent on other (N-1) particles. This algorithm is highly data parallel and a perfect candidate to offload to GPU. The code demonstrates how to deal with multiple device kernels, which can be enqueued into a DPC++ queue for execution and how to handle parallel reductions.

## Key Implementation Details
The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the Program for CPU and GPU

### Running Samples In DevCloud
Running samples in the Intel DevCloud requires you to specify a compute node. For specific instructions, jump to [Run the Nbody sample on the DevCloud](#run-nbody-on-devcloud)


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

### Include Files
The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### On a Linux* System
Perform the following steps:
1. Build the program using the following `cmake` commands.
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
2. Run the program
    ```
    make run
    ```

3. Clean the program
    ```
    make clean
    ```

### On a Windows* System Using Visual Studio* Version 2017 or Newer
- Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild Hello_World_GPU.sln /t:Rebuild /p:Configuration="Release"`

### Application Parameters
You can modify the NBody simulation parameters from within GSimulation.cpp. The configurable parameters include:
- set_npart(__);
- set_nsteps(__);
- set_tstep(__);
- set_sfreq(__);

Below are the default parameters:

* Number of particles (npart) is 16000
* Number of integration steps (nsteps) is 10
* Time delta (tstep) is 0.1s
* Sample frequency (sfreq) is 1

## Example of Output
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

### Running the Nbody sample in the DevCloud<a name="run-nbody-on-devcloud"></a>
1.  Open a terminal on your Linux system.
2.	Log in to DevCloud.
```
ssh devcloud
```
3.	Download the samples.
```
git clone https://github.com/oneapi-src/oneAPI-samples.git
```

4. Change directories to the Nbody sample directory.
```
cd ~/oneAPI-samples/DirectProgramming/DPC++/N-bodyMethods/Nbody
```
#### Build and run the sample in batch mode
The following describes the process of submitting build and run jobs to PBS.
A job is a script that is submitted to PBS through the qsub utility. By default, the qsub utility does not inherit the current environment variables or your current working directory. For this reason, it is necessary to submit jobs as scripts that handle the setup of the environment variables. In order to address the working directory issue, you can either use absolute paths or pass the -d \<dir\> option to qsub to set the working directory.

#### Create the Job Scripts
1.	Create a build.sh script with your preferred text editor:
```
nano build.sh
```
2.	 Add this text into the build.sh file:
```
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
mkdir build
cd build
cmake ..
make
```

3.	Save and close the build.sh file.

4.	Create a run.sh script with with your preferred text editor:
```
nano run.sh
```

5.	 Add this text into the run.sh file:
```
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
cd build
make run
```
6.	Save and close the run.sh file.

#### Build and run
Jobs submitted in batch mode are placed in a queue waiting for the necessary resources (compute nodes) to become available. The jobs will be executed on a first come basis on the first available node(s) having the requested property or label.
1.	Build the sample on a gpu node.

```
qsub -l nodes=1:gpu:ppn=2 -d . build.sh
```

Note: -l nodes=1:gpu:ppn=2 (lower case L) is used to assign one full GPU node to the job.
Note: The -d . is used to configure the current folder as the working directory for the task.

2.	In order to inspect the job progress, use the qstat utility.
```
watch -n 1 qstat -n -1
```
Note: The watch -n 1 command is used to run qstat -n -1 and display its results every second. If no results are displayed, the job has completed.

3.	After the build job completes successfully, run the sample on a gpu node:
```
qsub -l nodes=1:gpu:ppn=2 -d . run.sh
```
4.	When a job terminates, a couple of files are written to the disk:

    <script_name>.sh.eXXXX, which is the job stderr

    <script_name>.sh.oXXXX, which is the job stdout

    Here XXXX is the job ID, which gets printed to the screen after each qsub command.

5.	Inspect the output of the sample.
```
cat run.sh.oXXXX
```
You should see output similar to this:

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

6.	Remove the stdout and stderr files and clean-up the project files.
```
rm build.sh.*; rm run.sh.*; make clean
```
7.	Disconnect from the Intel DevCloud.
```
exit
```
### Build and run additional samples
Several sample programs are available for you to try, many of which can be compiled and run in a similar fashion to this sample. Experiment with running the various samples on different kinds of compute nodes or adjust their source code to experiment with different workloads.
