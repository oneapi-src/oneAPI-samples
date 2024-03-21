## Title
The `guided iso3dfd GPUOptimization` sample demonstrates how to use the Intel&reg; oneAPI Base Toolkit (Base Kit) and tools found in the Base Kit to optimize code for GPU offload. The ISO3DFD sample refers to Three-Dimensional Finite-Difference Wave Propagation in Isotropic Media; it is a three-dimensional stencil to simulate a wave propagating in a 3D isotropic medium.

This sample follows the workflow found in [Optimize Your GPU Application with the Intel&reg; oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/gpu-optimization-workflow.html#gs.101gmt2).

For comprehensive instructions, see the [Intel&reg; oneAPI Programming Guide](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.

| Property                       | Description
|:---                               |:---
| What you will learn               | How to offload the computation to GPU and iteratively optimize the application performance using Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 50 minutes

## Purpose

This sample starts with a CPU oriented application and shows how to use SYCL* and the oneAPI tools to offload regions of the code to the target system GPU. The sample relies heavily on use of the Intel Advisor, which is a design and analysis tool for developing performant code.  We'll use Intel&reg; Advisor to conduct offload modeling and identify code regions that will benefit the most from GPU offload. Once the initial offload is complete, we'll walk through how to develop an optimization strategy by iteratively optimizing the code based on opportunities exposed Intel&reg; Advisor to run roofline analysis.

ISO3DFD is a finite difference stencil kernel for solving the 3D acoustic isotropic wave equation, which can be used as a proxy for propagating a seismic wave. The sample implements kernels as 16th order in space, with symmetric coefficients, and 2nd order in time scheme without boundary conditions.

The sample includes four different versions of the iso3dfd project.

- `1_CPU_only.cpp`: basic serial CPU implementation.
- `2_GPU_basic.cpp`: initial GPU offload version using SYCL.
- `3_GPU_linear.cpp`: first compute optimization by changing indexing patern.
- `4_GPU_optimized.cpp`: additional optimizations for memory bound.


## Prerequisites
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 <br>Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler <br>Intel&reg; Advisor


## Key Implementation Details

The basic SYCL* standards implemented in the code include the use of the following:

- SYCL* local buffers and accessors (declare local memory buffers and accessors to be accessed and managed by each workgroup)
- Code for Shared Local Memory (SLM) optimizations
- SYCL* kernels (including parallel_for function and nd-range<3> objects)
- SYCL* queues (including exception handlers)


## Building the `ISO3DFD` Program for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
>
> Windows:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
>
>For more information on environment variables, see Use the setvars Script for [Linux or macOS](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html), or [Windows](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


> **Note**: For GPU Analysis on Linux* enable collecting GPU hardware metrics by setting the value of dev.i915 perf_stream_paranoidsysctl option to 0 as follows. This command makes a temporary change that is lost after reboot:
>
> `sudo sysctl -w dev.i915.perf_stream_paranoid=0`
>
>To make a permanent change, enter:
>
> `sudo echo dev.i915.perf_stream_paranoid=0 > /etc/sysctl.d/60-mdapi.conf`

### Running Samples in Intel&reg; DevCloud

If running a sample in the Intel&reg; DevCloud, you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit [Get Started Guide](https://devcloud.intel.com/oneapi/get_started/).

### On Linux*
Perform the following steps:
1. Build the program using the following `cmake` commands.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

2. Run the program.
   ```
   $ make run_all
   ```
#### Training Modules

| Modules | Description
|---|---|
|[__ISO3DFD and offload Advsior analysis running on CPU__](01_ISO3DFD_CPU/iso3dfd_Offload_Advisor_Analysis.ipynb) | + Provide performance analysis/projections of the application and run then offload modeling on the CPU version of the application|
|[__ISO3DFD and Implementation using SYCL offloading to a GPU__](02_ISO3DFD_GPU_Basics/iso3dfd_gpu_basice.ipynb) |+ how to offload the most profitable loops in the code on the GPU using SYCL|
|[__ISO3DFD on a GPU and Index computations__](03_ISO3DFD_GPU_Linear/iso3dfd_gpu_linear.ipynb)|+ Write kernels by reducing index calculations by changing how we calculate indices.we can change the 3D indexing to 1D|
|[__ISO3DFD and Implementation using SYCL offloading to a GPU__](iso3dfd_gpu_optimized/3dfd_gpu_basice.ipynb)|+ change the kernel to nd_range; we learn to offload more work in each local work group, which optimizes loading neighboring stencil points from the fast L1 cache.|

#### Content Structure
Each module folder has a Jupyter Notebook file (`*.ipynb`), this can be opened in Jupyter Lab to view the training content, edit code and compile/run. Along with the Notebook file, there is a `lab` and a `src` folder with SYCL source code for samples used in the Notebook. The module folder also has `run_*.sh` files, which can be used in shell terminal to compile and run each sample code.

- `01_{Module_Name}`
  - `lab`
    - `{sample_code_name}.cpp` - _(sample code editable via Jupyter Notebook)_
  - `src`
    - `{sample_code_name}.cpp` - _(copy of sample code)_
  - `01_{Module_Name}.ipynb` - _(Jupyter Notebook with training content and sample codes)_
  - `run_{sample_code_name}.sh` - _(script to compile and run {sample_code_name}.cpp)_
  - `License.txt`
  - `Readme.md`


## Install Directions

The training content can be accessed locally on the computer after installing necessary tools, or you can directly access using Intel DevCloud without any installation necessary.

#### Access using Intel DevCloud

The Jupyter notebooks are tested and can be run on Intel DevCloud without any installation necessary, below are the steps to access these Jupyter notebooks on Intel DevCloud:
1. Register on [Intel DevCloud](https://devcloud.intel.com/oneapi)
2. Login, Get Started and Launch Jupyter Lab
3. Open Terminal in Jupyter Lab and git clone the repo and access the Notebooks

#### Local Installation of oneAPI Tools and JupyterLab

The Jupyter Notebooks can be downloaded locally to computer and accessed:
- Install Intel oneAPI Base Toolkit on local computer: [Installation Guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- Install Jupyter Lab on local computer: [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- git clone the repo and access the Notebooks using Jupyter Lab

#### Local Installation of oneAPI Tools and use command line

The Jupyter Notebooks can be viewed on Github and you can run the code on command line:
- Install Intel oneAPI Base Toolkit on local computer (linux): [Installation Guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- git clone the repo
- open command line terminal and use the `run_*.sh` script in each module to compile and run code.

## License  
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

