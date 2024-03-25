# `Genetic Algorithms on GPU using Intel® Distribution of Python numba-dpex` Sample

The `Genetic Algorithms on GPU using Intel® Distribution of Python numba-dpex` sample shows how to implement general generic algorithm (GA) and offload computation to GPU using numba-dpex.

| Area                    | Description
| :---                    | :---
| What you will learn     | How to implement genetic algorithm using the Data-parallel Extension for Numba* (numba-dpex)?
| Time to complete        | 8 minutes
| Category                | Code Optimization

>**Note**: The libraries used in this sample are available in Intel® Distribution for Python* as part of the [Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/en-us/oneapi/ai-kit).

## Purpose

In this sample, you will create and run the general genetic algorithm and optimize it to run on GPU using Intel® Distribution for Python* numba-dpex. You will learn what are selection, crossover and mutation, and how to adjust those methods from general genetic algorithm to specific optimization problem which is Traveling Salesman Problem.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 20.04
| Hardware                | GPU
| Software                | Intel® AI Analytics Toolkit (AI Kit)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit-download.html).

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See *[Intel® DevCloud for oneAPI](https://DevCloud.intel.com/oneapi/get_started/)* for information.

## Key Implementation Details

This sample code is implemented for the GPU using Python. The sample assumes you have numba-dpex installed inside a Conda environment, similar to what is installed with the Intel® Distribution for Python*.

>**Note**: Read *[Get Started with the Intel® AI Analytics Toolkit for Linux*](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux/top.html)* to find out how you can achieve performance gains for popular deep-learning and machine-learning frameworks through Intel optimizations.

The sample tutorial contains one Jupyter Notebook and a Python script. You can use either.

### Jupyter Notebook

| Notebook                                                 | Description
|:---                                                      |:---
|`IntelPython_GPU_numba-dpex_Genetic_Algorithm.ipynb`      | Genetic Algorithms on GPU using Intel® Distribution of Python numba-dpex

### Python Scripts

| Script                                                | Description
|:---                                                   |:---
|`IntelPython_GPU_numba-dpex_Genetic_Algorithm.py`      | The script performs Genetic Algorithms on GPU using Intel® Distribution of Python numba-dpex code sample in the command-line interface (CLI)

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `Genetic Algorithms on GPU using Intel® Distribution of Python numba-dpex` Sample

### On Linux*

<!-- > **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*. -->

#### Activate Conda

1. Activate the Conda environment.
   ```
   conda activate base
   ```
   By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it.

   You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

   ```
   conda create --name usr_base --clone base
   conda activate usr_base
   ```

#### Run the Python Script

1. Change to the sample directory.
2. Run the script.
   ```
   python IntelPython_GPU_numba-dpex_Genetic_Algorithm.py
   ```

#### Run the Jupyter Notebook (Optional)

1. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0
   ```
2. Follow the instructions to open the URL with the token in your browser.
3. Locate and select the Notebook.
   ```
   IntelPython_GPU_numba-dpex_Genetic_Algorithm.ipynb
   ```
4. Run every cell in the Notebook in sequence.

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]` at the end of execution. The sample will print out the runtimes and charts of relative performance with numba-dpex and without any optimizations as the baseline. Additionally sample will print best and worst path found in Traveling Salesman problem.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
