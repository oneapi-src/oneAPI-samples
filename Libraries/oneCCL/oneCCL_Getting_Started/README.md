# `oneCCL Getting Started` Samples
The CCL sample codes are implemented using C++, C and DPC++ language for CPU and GPU.
By using all reduce collective operation samples, users can understand how to compile oneCCL codes with various oneCCL configurations in Intel oneAPI environment.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04;
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | oneAPI Collective Communications Library (oneCCL), oneAPI DPC++/C++ Compiler, v, GNU Compiler
| What you will learn               | basic oneCCL programming model for both Intel CPU and GPU
| Time to complete                  | 15 minutes

## List of Samples
| C++ API | Collective Operation |
| ------ | ------ |
| sycl_allreduce_test.cpp |[Allreduce](https://spec.oneapi.io/versions/latest/elements/oneCCL/source/spec/collective_operations.html#allreduce) |
| cpu_allreduce_test.cpp/cpu_allreduce_bf16_test.c |[Allreduce](https://spec.oneapi.io/versions/latest/elements/oneCCL/source/spec/collective_operations.html#allreduce) |
>  Notice: Please use Intel® DevCloud for oneAPI as the environment for jupyter notebook samples. \
Users can refer to [DevCloud Getting Started](https://devcloud.intel.com/oneapi/get-started/) for using DevCloud \
Users can use JupyterLab from DevCloud via "One-click Login in", and download samples via "git clone" or the "oneapi-cli" tool \
Once users are in the JupyterLab with download jupyter notebook samples, they can start following the steps without further installation needed.

You can also use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

## Purpose
The samples implement the allreduce collective operation with oneCCL APIs.
The sample users will learn how to compile the code with various oneCCL configurations in the Intel oneAPI environment.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Prerequisites

### CPU

-----

The samples below require the following components, which are part of the [Intel® oneAPI DL Framework Developer Toolkit (DLFD Kit)
](https://software.intel.com/en-us/oneapi/dldev-kit)
*  oneAPI Collective Communications Library (oneCCL)

You can refer to this page [oneAPI](https://software.intel.com/en-us/oneapi) for toolkit installation.


### GPU and CPU

-----

The samples below require the following components, which are part of the [Intel® oneAPI Base Toolkit (Base Kit)](https://software.intel.com/en-us/oneapi/oneapi-kit)
*  oneAPI Collective Communications Library (oneCCL)
*  oneAPI DPC++/C++ Compiler
*  oneAPI DPC++ Library (oneDPL)

The samples also require an OpenCL driver. Please refer [System Requirements](https://software.intel.com/en-us/articles/intel-oneapi-base-toolkit-system-requirements) for OpenCL driver installation.


You can refer to this page [oneAPI](https://software.intel.com/en-us/oneapi) for toolkit installation.


### Using Visual Studio Code* (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the **Generate Launch Configurations** extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.


## Building the samples for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### on a Linux* System

#### CPU only:

- Build the samples  with GCC for CPU only \
  please replace ${ONEAPI_ROOT} for your installation path. \
  ex : /opt/intel/oneapi \
  Don't need to replace {DPCPP_CMPLR_ROOT}
  ```
  source ${ONEAPI_ROOT}/setvars.sh --ccl-configuration=cpu

  cd oneapi-toolkit/oneCCL/oneCCL_Getting_Started
  mkdir build
  cd build
  cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
  make cpu_allreduce_test
  ```
> NOTE: The source file "cpu_allreduce_test.cpp" will be copied from ${INTEL_ONEAPI_INSTALL_FOLDER}/ccl/latest/examples/cpu to build/src/cpu folder.
Users can rebuild the cpu_allreduce_test.cpp by typing "make cpu_allreduce_test" under build folder.

#### GPU and CPU:

- Build the samples  with SYCL for GPU and CPU \
  please replace ${ONEAPI_ROOT} for your installation path. \
  ex : /opt/intel/oneapi \
  Don't need to replace {DPCPP_CMPLR_ROOT}
  ```
  source ${ONEAPI_ROOT}/setvars.sh --ccl-configuration=cpu_gpu_dpcpp

  cd oneapi-toolkit/oneCCL/oneCCL_Getting_Started
  mkdir build
  cd build
  cmake ..  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=dpcpp -DCOMPUTE_BACKEND=dpcpp
  make sycl_allreduce_test
  ```
> NOTE: The source file "sycl_allreduce_test.cpp" will be copied from ${INTEL_ONEAPI_INSTALL_FOLDER}/ccl/latest/examples/sycl to build/src/sycl folder.
Users can rebuild the sycl_allreduce_test.cpp by typing "make sycl_allreduce_test" under build folder.

### Include Files
The include folder is located at ${CCL_ROOT}}\include on your development system".

## Running the Sample

### on a Linux* System

#### CPU only:
- Run the program \
  take cpu_allreduce_test for example. \
  you can apply those steps for all other sample binaries. \
  please replace the {NUMBER_OF_PROCESSES} with integer number accordingly

  ```
  mpirun -n ${NUMBER_OF_PROCESSES} ./out/cpu/cpu_allreduce_test
  ```

  ex:
  ```
  mpirun -n 2 ./out/cpu/cpu_allreduce_test
  ```


#### GPU and CPU:
- Run the program \
  take sycl_allreduce_test for example. \
  you can apply those steps for all other sample binaries. \
  please replace the {NUMBER_OF_PROCESSES} with integer number accordingly

  ```
  mpirun -n ${NUMBER_OF_PROCESSES} ./out/sycl/sycl_allreduce_test gpu|cpu|host|default
  ```

  ex: run on GPU
  ```
  mpirun -n 2 ./out/sycl/sycl_allreduce_test gpu
  ```


### Example of Output

#### on Linux
- Run the program on CPU or GPU following [How to Run Section](#running-the-sample)
- CPU Results

  ```
  Provided device type: cpu
  Running on Intel(R) Core(TM) i7-7567U CPU @ 3.50GHz
  Example passes
  ```
  please note that name of the running device may vary according to your environment


- GPU Results
  ```
  Provided device type: gpu
  Running on Intel(R) Gen9 HD Graphics NEO
  Example passes
  ```
  please note that name of the running device may vary according to your environment

- Enable oneCCL Verbose log

  There are different log levels in oneCCL. Users can refer to [the link](https://oneapi-src.github.io/oneCCL/env-variables.html#ccl-log-level) for different log levels.


  Users can enable oneCCL verbose log by following the command shown below to see more
  runtime information from oneCCL.
  ```
  export CCL_LOG_LEVEL=info
  ```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)
