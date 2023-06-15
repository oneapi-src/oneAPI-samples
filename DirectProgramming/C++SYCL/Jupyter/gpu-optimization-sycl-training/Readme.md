## Title
 GPU Optimization with SYCL
  
## Requirements
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 18.04, 20 Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler, Jupyter Notebooks, Intel DevCloud
  
## Purpose
The primary focus of this document is GPUs. Each section focuses on different topics to guide you in your path to creating optimized solutions.

Designing high-performance software requires you to think differently than you might normally do when writing software. You need to be aware of the hardware on which your code is intended to run, and the characteristics that control the performance of that hardware. Your goal is to structure the code such that it produces correct answers, but does so in a way that maximizes the hardware’s ability to execute the code.

Also, it familiarizes you with the use of Jupyter notebooks as a front-end for all training exercises. This workshop is designed to be used on the DevCloud and includes details on submitting batch jobs on the DevCloud environment.

At the end of this course, you will be able to:

- Optimize your SYCL code to run faster and efficiently on GPUs.

## Content Details

#### Pre-requisites
- C++ Programming
- SYCL Programming

#### Training Modules

| Modules | Description
|---|---|
|[__Introduction to GPU Optimization__](01_Introduction_to_GPU_Optimization/01_Introduction.ipynb) | + Phases in the Optimization Workflow<br>+ Locality Matters<br>+ Parallelization<br>+ GPU Execution Model Overview|
|[__Thread Mapping and Occupancy__](02_Thread_Mapping_and_Occupancy/02_Thread_Mapping_and_Occupancy.ipynb) |+ nd_range Kernel<br>+ Thread Synchronization<br>+ Mapping Work-groups to Xe-cores for Maximum Occupancy<br>+ Intel® GPU Occupancy Calculator|
|[__Memory Optimizations__](03_Memory_Optimization/03_Memory_Optimization.ipynb) |[Memory Optimization - Buffers](03_Memory_Optimization/031_Memory_Optimization_Buffers.ipynb)<br>+ Buffer Accessor Modes<br>+ Optimizing Memory Movement Between Host and Device<br>+ Avoid Declaring Buffers in a Loop<br>+ Avoid Moving Data Back and Forth Between Host and Device<br>[Memory Optimization - USM](03_Memory_Optimization/032_Memory_Optimization_USM.ipynb)<br>+ Overlapping Data Transfer from Host to Device<br>+ Avoid Copying Unnecessary Block of Data<br>+ Copying Memory from Host to USM Device Allocation|
|[__Kernel Submission__](04_Kernel_Submission/04_Kernel_Submission.ipynb)|+ Kernel Launch<br>+ Executing Multiple Kernels<br>+ Submitting Kernels to Multiple Queues<br>+ Avoid Redundant Queue Construction|
|[__Kernel Programming__](05_Kernel_Programming/05_Kernel_Programming.ipynb)|+ Considerations for Selecting Work-group Size<br>+ Removing Conditional Checks<br>+ Avoiding Register Spills|
|[__Shared Local Memory__](06_Shared_Local_Memory/06_Shared_Local_Memory.ipynb)|+ SLM Size and Work-group Size<br>+ Bank Conflicts<br>+ Using SLM as Cache<br>+ Data Sharing and Work-group Barriers|
|[__Sub-Groups__](07_Sub_Groups/07_Sub_Groups.ipynb)|+ Sub-group Sizes<br>+ Sub-group Size vs. Maximum Sub-group Size<br>+ Vectorization and Memory Access<br>+ Data Sharing|
|[__Atomic Operations__](08_Atomic_Operations/08_Atomic_Operations.ipynb)|+ Data Types for Atomic Operations<br>+ Atomic Operations in Global vs Local Space|
|[__Kernel Reduction__](09_Kernel_Reduction/08_Kernel_Reduction.ipynb)|+ Reduction Using Atomic Operations<br>+ Reduction Using Shared Local Memory<br>+ Reduction Using Sub-Groups <br>+ Reduction Using SYCL Reduction Kernel|

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
