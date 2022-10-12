# `Histogram` Sample
The `Histogram` sample demonstrates a histogram that groups numbers together and
provides the count of a particular number in the input. The code in this sample
uses Intel® oneAPI DPC++ Library (oneDPL) APIs to offload the computation to
selected devices.

## Purpose
This sample creates both dense and sparse histograms using oneDPL APIs, on an
input array of 1,000 elements with values chosen randomly ranging from 0 to 9 (inclusive). To differentiate between sparse and dense histogram, the code ensures one of the values (number 4) never occurs in the input array. One bin will always equal 0.

- For the dense histogram, all the bins (including the zero-size bins) are
  stored.
- For the sparse algorithm, only non-zero sized bins are stored.

>**Note**: For comprehensive information about oneAPI programming, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide). Use search or the table of contents to find relevant information quickly.

## Prerequisites
| Optimized for       | Description
| :---                | :---
| OS                  | Ubuntu* 18.04
| Hardware            | Skylake with GEN9 or newer
| Software            | Intel® oneAPI DPC++ Compiler

## Key Implementation Details
The basic SYCL* implementation explained in the code includes accessor, kernels,
queues, buffers, and some oneDPL library calls.

## Set Environment Variables
When working with the command-line interface (CLI), you should configure the
oneAPI toolkits using environment variables. Set up your CLI environment by
sourcing the `setvars` script every time you open a new terminal window. This
practice ensures that your compiler, libraries, and tools are ready for
development.

## Build the `Histogram` Program for GPU or CPU
> **Note**: If you have not already done so, set up your CLI environment by
> sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source
>   <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the
> setvars Script with Linux* or
> macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)

### Include Files
The include folder is at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your
development system. You might need to use some of the resources from this
location to build the sample.

### Use Visual Studio Code* (VS Code) (Optional)
You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment
    Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel®
    oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment,
see the [Using Visual Studio Code with Intel® oneAPI Toolkits User
Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

### On Linux*
1. Change to the sample directory.
2. Build the program.
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```
If an error occurs, you can get more details by running `make` with the
`VERBOSE=1` argument:
```
make VERBOSE=1
```
#### Troubleshooting
If you receive an error message, troubleshoot the problem using the
**Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility
provides configuration and system checks to help find missing dependencies,
permissions errors, and other issues. See the [Diagnostics Utility for Intel®
oneAPI Toolkits User
Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)
for more information on using the utility.

## Run the `Histogram` Program
### Application Parameters
You can supply any set of input values to the `dense_histogram()` and `sparse_histogram()` functions in the `main.cpp` source file. By default, the input is randomly generated.

### On Linux
1. Run the program.
   ```
   make run
   ```
   Alternatively, you can run the program directly, `./histogram`.

2. Clean the program. (Optional)
   ```
   make clean
   ```

### Run `Histogram` Sample in Intel® DevCloud (Optional)
When running a sample in the Intel® DevCloud, you must specify the compute node
(CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more
information on how to specify compute nodes read, [Launch and manage
jobs](https://devcloud.intel.com/oneapi/documentation/job-submission/) in the
Intel® DevCloud for oneAPI Documentation.

For more information on using Intel® DevCloud, see the Intel&reg; oneAPI Base
Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

#### Build and Run Samples in Batch Mode (Optional)
You can submit build and run jobs through a Portable Bash Script (PBS). A job is
a script that submitted to PBS through the `qsub` utility. By default, the
`qsub` utility does not inherit the current environment variables or your
current working directory, so you might need to submit jobs to configure the
environment variables. To indicate the correct working directory, you can use
either absolute paths or pass the `-d \<dir\>` option to `qsub`. 

If you choose to use scripts, jobs terminate with writing files to the disk:
- `<script_name>.sh.eXXXX`, which is the job stderr
- `<script_name>.sh.oXXXX`, which is the job stdout

Here XXXX is the job ID, which gets printed to the screen after each qsub
command. 

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
   qsub	-I  -l nodes=1:gpu:ppn=2 -d .
   ```
   - `-I` (upper case I) requests an interactive session.
   - `-l nodes=1:gpu:ppn=2` (lower case L) assigns one full GPU node. 
   - `-d .` makes the current folder as the working directory for the task.

> **Note**: To inspect job progress, use the qstat utility.
>   ```
>   watch -n 1 qstat -n -1
>   ```
>  The command displays the results every second. The job is complete when no
>  new results display.

6. Perform build steps you would on Linux. (Including optionally cleaning the
   project.)
7. Run the sample. 
8. Disconnect from the Intel® DevCloud. 
   ```
   exit
   ```

## Example Output
The example output shown below assumes input of the following array.
```
1 1 8 1 8 6 1 0 1 5 5 2 2 8 1 2 1 1 1 6 2 1 1 8 3 6 6 2 2 1 1 8
1 0 0 0 2 2 7 6 5 1 6 1 1 6 1 5 1 0 0 1 1 1 0 5 5 0 7 0 1 6 0 5
7 0 3 0 0 0 0 6 0 2 5 5 6 6 8 7 6 6 8 8 7 7 2 2 0 7 2 2 5 2 7 1
3 0 1 1 0 1 7 2 0 1 5 1 7 0 8 3 1 5 0 6 1 0 8 2 7 2 1 1 1 3 2 5
1 2 5 1 6 3 3 1 3 8 0 1 1 8 2 0 2 0 1 2 0 2 1 8 1 6 0 6 7 1 1 8
3 6 0 7 7 1 6 1 7 6 1 8 3 3 6 3 1 2 7 2 1 0 1 8 7 0 5 5 1 1 3 2
1 3 7 0 3 2 1 1 8 0 1 0 2 5 3 6 7 0 6 2 0 8 8 5 6 3 0 5 7 3 5 0
0 3 7 7 5 6 7 2 7 8 0 0 2 3 0 1 3 1 1 2 7 1 5 1 0 3 7 2 0 3 0 0
6 7 5 0 5 3 0 3 0 0 1 3 2 5 2 3 6 3 5 5 2 0 7 6 3 6 7 6 0 7 6 5
6 0 3 0 2 1 1 0 2 2 1 1 7 3 8 2 5 2 7 7 2 1 3 2 1 1 1 8 6 5 2 3
3 6 1 5 8 2 1 1 2 5 2 0 7 3 3 3 3 8 8 0 1 2 8 2 3 7 0 8 1 2 2 1
6 2 8 5 1 3 5 7 8 0 5 2 1 8 7 0 6 7 8 7 7 5 8 0 3 8 8 2 8 1 7 2
1 6 0 0 7 3 2 2 1 7 0 2 5 7 5 2 3 1 0 2 1 6 2 2 3 1 5 3 0 3 5 0
7 3 1 5 7 6 7 8 2 7 0 7 2 5 7 5 0 6 5 8 3 7 0 7 6 5 8 5 6 2 5 2
5 0 5 1 1 3 1 6 0 8 3 0 0 1 7 2 5 2 0 7 2 0 3 7 3 0 3 0 2 6 0 7
6 5 0 1 8 8 5 8 7 8 1 0 8 0 2 2 2 2 0 2 0 3 0 3 3 3 3 3 7 3 2 0
6 0 3 0 8 0 1 1 6 3 1 3 1 0 6 3 7 1 5 7 8 6 0 0 7 1 1 6 3 2 8 0
2 3 0 1 1 6 3 5 7 7 0 8 2 1 0 7 8 5 2 5 0 0 6 6 5 8 3 8 1 2 7 5
3 2 1 0 8 7 8 1 3 8 1 3 3 1 2 0 5 1 6 3 6 1 0 2 7 3 0 8 1 7 2 5
7 6 8 5 2 7 0 5 6 2 8 7 1 8 7 2 3 2 8 0 3 8 1 1 1 1 7 5 6 0 8 2
6 7 7 8 5 8 2 2 8 2 7 0 1 6 3 5 8 2 3 1 1 2 0 2 3 8 5 7 8 5 1 1
1 8 1 7 5 0 7 1 0 6 3 5 1 6 8 0 6 1 8 7 5 0 8 7 6 2 5 5 5 6 7 7
1 0 5 0 2 3 3 6 0 1 0 1 8 7 0 5 8 6 3 2 2 0 0 1 3 6 5 8 1 3 2 5
1 0 6 3 0 7 7 2 2 8 2 1 1 2 6 3 6 7 5 2 8 6 3 0 1 8 6 0 1 2 6 0
0 1 2 2 8 0 5 1 6 7 0 1 7 6 1 2 2 8 6 8 5 8 8 1 5 1 1 6 6 8 7 6
0 0 0 6 7 3 5 5 8 5 2 6 2 7 8 3 6 1 2 0 1 2 1 6 6 6 2 1 6 7 5 0
5 3 2 3 6 7 6 5 2 2 0 1 0 7 7 6 0 8 1 1 1 8 7 5 3 7 1 0 5 0 3 1
2 5 5 8 1 0 3 5 0 1 8 0 6 0 0 6 3 8 5 2 5 1 5 0 2 0 7 6 8 1 7 1
0 1 0 6 0 1 0 0 1 8 1 7 2 3 3 5 1 8 6 6 1 2 2 2 3 1 8 2 2 6 3 7
6 1 2 6 1 2 6 2 0 5 0 2 7 3 5 8 3 2 3 1 5 6 6 6 7 3 8 0 8 0 5 5
8 5 0 0 6 2 0 6 8 1 6 6 2 0 3 5 3 2 8 6 1 3 3 8 7 0 7 6 7 1 0 6
7 0 5 0 0 5 8 1
```
### Example Output
For the input shown above, the program would output the following results.
```
success for Dense Histogram:
[(0, 161) (1, 170) (2, 136) (3, 108) (4, 0) (5, 105) (6, 110) (7, 108) (8, 102) ]
success for Sparse Histogram:
[(0, 161) (1, 170) (2, 136) (3, 108) (5, 105) (6, 110) (7, 108) (8, 102) ]
```
> **Note**: Your results will differ.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third-party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).