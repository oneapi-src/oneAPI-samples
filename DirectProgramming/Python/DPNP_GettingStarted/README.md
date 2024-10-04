# Intel® Python Data Parallel Extension for NumPy Getting Started Sample

The `Intel® Python DPNP Getting Started` sample code shows how to find conjugate gradient using the Intel Python API powered by the [Intel® Python DPNP - Data Parallel Extension for NumPy](https://github.com/IntelPython/dpnp).

| Area                   | Description
| :---                   | :---
| Category               | Getting Started
| What you will learn    | DPNP programming model for Intel GPU
| Time to complete       | 60 minutes
>**Note**: This sample is migrated from Cupy Python sample. See the [ConjugateGradient](https://github.com/cupy/cupy/blob/main/examples/cg/cg.py) sample in the cupy-samples GitHub.


## Purpose
The Data Parallel Extension for NumPy* (dpnp package) - a library that implements a subset of NumPy* that can be executed on any data parallel device. The subset is a drop-in replacement of core NumPy* functions and numerical data types. 

The DPNP is used to offload python code to INTEL GPU's. This is very similar to CUPY API [Comparsion_list](https://intelpython.github.io/dpnp/reference/comparison.html#).   


## Prerequisites

| Optimized for           | Description
| :---                    | :---
| OS                      | Ubuntu* 22.04 (or newer)
| Hardware                | Intel® Gen9 <br>Intel® Gen11 <br>Intel® Data Center GPU Max 
| Software                | Intel® Python Data Parallel Extension for NumPy (DPNP)
> **Note**: [Intel® Python DPNP - Data Parallel Extension for NumPy](https://github.com/IntelPython/dpnp).

## Key Implementation Details

- This get-started sample code is implemented for Intel GPUs using Python language. The example assumes the user has the latest DPNP installed in the environment, similar to what is delivered with the installation of the [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-python-download.html).
  
## Environment Setup

You will need to download and install the following toolkits, tools, and components to use the sample.

**1. Intel Python**


Required Intel Python package: DPNP (Select Intel® Distribution for Python*: Offline on [Get Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-python-download.html) to install)


**2. (Offline Installer) Update the Intel Python base environment**

Load python env:
```
source $PYTHON_INSTALL/env/vars.sh
```
 
**3. (Offline Installer) Check the DPNP version**

```
python -c "import dpnp; print(dpnp.__version__)"
``` 
Note: if the version is 0.15.0 or more continue, otherwise need to upgrade the dpnp version 

**4. Clone the GitHub repository**
<!-- for oneapi-samples: git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/DirectProgramming/<samples-folder>/<individual-sample-folder> -->
<!-- for migrated samples - provide git clone command for individual repo and cd to sample dir --> 
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/DirectProgramming/Python/DPNP_GettingStarted
```


## Run the Sample
>**Note**: Before running the sample, make sure Intel Python is installed.

1. Change to the sample directory.
2. Build the program.
   ```
   $ python cg.py
   ```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
