# `ISO3DFD OpenMP Offload` Sample

The ISO3DFD sample refers to Three-Dimensional Finite-Difference Wave Propagation in Isotropic Media.  It is a three-dimensional stencil to simulate a wave propagating in a 3D isotropic medium and shows some of the more common challenges and techniques when targeting OMP Offload devices (GPU) in more complex applications to achieve good performance.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler;
| What you will learn               | How to offload the computation to GPU using Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes

Performance number tabulation

| iso3dfd_omp_offload sample            | Performance data
|:---                               	|:---
| Default Baseline version              | 1.0
| Optimized version 1	                | 1.11x
| Optimized version 2	                | 1.48x
| Optimized version 3	                | 1.60x


## Purpose

ISO3DFD is a finite difference stencil kernel for solving the 3D acoustic isotropic wave equation, which can be used as a proxy for propagating a seismic wave. In this sample, kernels are implemented as 16th order in space, with symmetric coefficients, and 2nd order in time scheme without boundary conditions. Using OpenMP Offload, the sample can explicitly run on the GPU to propagate a seismic wave, which is a compute-intensive task.

The code will attempt to find an available GPU or OpenMP Offload capable device and exit if a compatible device is not detected. By default, the output will print the device name where the OpenMP Offload code ran along with the grid computation metrics - flops and effective throughput. For validating results, an OpenMP/CPU-only version of the application will be run on host/CPU, and results will be compared to the OpenMP Offload version.

The code also demonstrates some of the common optimization techniques that can be used to improve 3D-stencil code running on a GPU device.

## Key Implementation Details

The basic OpenMP Offload implementation explained in the code includes the use of the following :
* OpenMP offload target data map construct
* Default Baseline version demonstrates the use of OpenMP offload target parallel for construct with the collapse
* Optimized version 1 demonstrates the use of OpenMP offload teams distribute construct and use of num_teams and thread_limit clause
* Incremental Optimized version 2 demonstrates the use of OpenMP offload teams distribute construct with improved data-access pattern
* Incremental Optimized version 3 demonstrates use of OpenMP CPU threads along with OpenMP offload target construct


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the `ISO3DFD` Program for GPU

### Running Samples In DevCloud
Running samples in the Intel DevCloud requires you to specify a compute node. For specific instructions, jump to [Run the ISO3DFD OpenMP Offload sample in the DevCloud](#run-iso3dfd-omp-on-devcloud)


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
Perform the following steps:
1. Build the program using the following `cmake` commands.
```
$ mkdir build
$ cd build
$ cmake ..
$ make -j
```

> Note: by default, the executable is built with the default baseline version. You can build the kernel with optimized versions with the following:
```
cmake -DUSE_OPT1=1 ..
make -j
```
```
cmake -DUSE_OPT2=1 ..
make -j
```
```
cmake -DUSE_OPT3=1 ..
make -j
```

2. Run the program :
    ```
    make run
    ```

3. Clean the program using:
    ```
    make clean
    ```

## Running the Sample
```
make run
```

### Application Parameters
You can modify the ISO3DFD parameters from the command line.
   * Configurable Application Parameters

	Usage: src/iso3dfd n1 n2 n3 n1_block n2_block n3_block Iterations

 	n1 n2 n3                       	: Grid sizes for the stencil
 	n1_block n2_block n3_block     	: cache block sizes for CPU
                                	: OR TILE sizes for OMP Offload
 	Iterations                     	: No. of timesteps.

### Example of Output with the default baseline version
```
Grid Sizes: 256 256 256
Tile sizes ignored for OMP Offload
--Using Baseline version with omp target with collapse
Memory Usage (MBytes): 230
--------------------------------------
time         : 4.827 secs
throughput   : 347.57 Mpts/s
flops        : 21.2018 GFlops
bytes        : 4.17084 GBytes/s

--------------------------------------

--------------------------------------
Checking Results ...
Final wavefields from OMP Offload device and CPU are equivalent: Success
--------------------------------------
```

### Example of Output with Optimized version 3
```
Grid Sizes: 256 256 256
Tile sizes: 16 8 64
Using Optimized target code - version 3:
--OMP Threads + OMP_Offload with Tiling and Z Window
Memory Usage (MBytes): 230
--------------------------------------
time         : 3.014 secs
throughput   : 556.643 Mpts/s
flops        : 33.9552 GFlops
bytes        : 6.67971 GBytes/s

--------------------------------------

--------------------------------------
Checking Results ...
Final wavefields from OMP Offload device and CPU are equivalent: Success

```
### Running the ISO3DFD OpenMP Offload sample in the DevCloud<a name="run-iso3dfd-omp-on-devcloud"></a>
1.  Open a terminal on your Linux system.
2.	Log in to DevCloud.
```
ssh devcloud
```
3.	Download the samples.
```
git clone https://github.com/oneapi-src/oneAPI-samples.git
```

4. Change directories to the  ISO3DFD OpenMP Offload sample directory.
```
cd ~/oneAPI-samples/DirectProgramming/C++/StructuredGrids/iso3dfd_omp_offload
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
make -j
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
Note: The watch -n 1 command is used to run qstat -n -1 and display its results every second. The **Req’d Time** column will give an estimate for when the job will complete.

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
Grid Sizes: 256 256 256
Tile sizes ignored for OMP Offload
--Using Baseline version with omp target with collapse
Memory Usage (MBytes): 230
--------------------------------------
time         : 9.912 secs
throughput   : 169.262 Mpts/s
flops        : 10.325 GFlops
bytes        : 2.03114 GBytes/s

--------------------------------------

--------------------------------------
Checking Results ...
Final wavefields from OMP Offload device and CPU are equivalent: Success
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
Several sample programs are available for you to try, many of which can be compiled and run in a similar fashion to iso3dfd_omp_offload. Experiment with running the various samples on different kinds of compute nodes or adjust their source code to experiment with different workloads.



