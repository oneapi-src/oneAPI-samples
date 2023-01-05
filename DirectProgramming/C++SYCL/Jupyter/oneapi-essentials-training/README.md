## Title
 oneAPI Essentials
  
## Requirements
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 18.04, 20 Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler, Jupyter Notebooks, Intel DevCloud
  
## Purpose
The Jupyter Notebooks in this training shows challenges of heterogenous programming and how SYCL programming language can solve application development across CPUs, GPUs and FPGAs. 

These training modules teach basics of SYCL programming to offload computation to GPUs and also goes deeper into SYCL concepts for optimization like Sub-Groups, Kernel Reductions, Buffer-Accessor memory model, pointer-based approach using Unified Shared Memory and Task scheduling. 

Modules also teach usage of Intel oneAPI Data Parallel C++ Library to simplify heterogenous programming and tools usage. Debugger tools and Performance analysis tools like Intel VTune Profiler and Intel Advisor.

Also, it familiarizes you with the use of Jupyter notebooks as a front-end for all training exercises. This workshop is designed to be used on the DevCloud and includes details on submitting batch jobs on the DevCloud environment.

At the end of this course you will be able to:

- Understand the challenges of Heterogenous Computing.
- Write a SYCL program that offloads computation to accelerator devices like CPUs, GPUs or FPGAs from any vendor. 
- Perform analysis and profile the SYCL code using Intel oneAPI tools to find performance bottle necks. 

## License  
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Content Details

#### Pre-requisites
- C++ Programming

#### Training Modules

| Modules | Description
|---|---|
|[oneAPI Introduction](01_oneAPI_Intro/oneAPI_Intro.ipynb)| + Introduction and Motivation for SYCL.<br>+ SYCL __Hello World__<br>+ Compiling SYCL and __DevCloud__ Usage
|[SYCL Program Structure](02_SYCL_Program_Structure/SYCL_Program_Structure.ipynb)| + __SYCL Classes__ - device, device_selector, queue, basic kernels and ND-Range kernels, Buffers-Accessor memory model<br>+ SYCL __Code Anatomy__<br>+ Implicit __Dependency__ with Accessors, __Synchronization__ with Host Accessor and Buffer Destruction<br>+ Creating __Custom__ Device Selector<br>+ ___Lab Exercise___: Vector Increment to Vector Add
|[SYCL Unified Shared Memory](03_SYCL_Unified_Shared_Memory/Unified_Shared_Memory.ipynb)|+ What is Unified Shared Memory(USM) and Motivation<br>+ __Implicit and Explicit USM__ code example<br>+ Handling __data dependency__ using depends_on() and ordered queues<br>+ ___Lab Exercise___: Unified Shared Memory
|[SYCL Sub Groups](04_SYCL_Sub_Groups/Sub_Groups.ipynb)| + What is Sub-Groups and Motivation<br>+ Querying for __sub-group info__<br>+ Sub-group __shuffle algorithms__<br>+ Sub-group __group algorithms__<br>+ ___Lab Exercise___: Sub-Groups | 90 min |
|[SYCL Kernel Reductions](08_SYCL_Reduction/Reductions.ipynb)|+ What are Reductions<br>+ Challenges with parallelizing reductions<br>+ __sycl::reduce_over_group__ function for sub-groups and work-groups<br>+ __sycl::reduction__ object in parallel_for<br>+ ___Lab Exercise___: Kernel Reductions 
|[SYCL Buffers and Accessors in depth](09_SYCL_Buffers_And_Accessors_Indepth/SYCL_Buffers_accessors.ipynb)| + Buffers and Accessors<br>+ Buffer properties and usecases<br>+ Create Sub-buffers<br>+ Host accessors and usecases<br>+ ___Lab Exercise___: Buffers and Accessors
|[SYCL Task Scheduling and Data Dependency](10_SYCL_Graphs_Scheduling_Data_management/SYCL_Task_Scheduling_Data_dependency.ipynb)| + Different types of data dependencies<br>+ Execution of graph scheduling<br>+ modes of dependencies in Graphs scheduling<br>+ ___Lab Exercise___: Task Scheduling | 120 min|
|[SYCL Local Memory And Atomics](12_SYCL_Local_Memory_And_Atomics/Local_Memory_Atomics.ipynb)| + Query Local memory type and size<br>+ Local memory and Group barriers<br>+ Local Accessor usage<br>+ Atomic Operations Buffers<br>+ Atomic Operations USM<br>+ ___Lab Exercise___: Atomic Operation | 90 min|
|[Intel® oneAPI DPC++ Library (oneDPL)](07_oneDPL_Library/oneDPL_Introduction.ipynb)| + Introduction to Intel oneAPI DPC++ Library (oneDPL)<br>+ ___Lab Exercise___: Gamma Correction with oneDPL
|[Intel® Advisor](05_Intel_Advisor/offload_advisor.ipynb)| + __Offload Advisor__ Tool usage and command-line options<br>+ ___Lab Exercise___: Generate Offload Advisor Report<br>+ __Roofline Analysis__ and command-line options<br>+ ___Lab Exercise___: Generate Roofline Report
|[Intel® VTune Profiler](06_Intel_VTune_Profiler/Intel_VTune_Profiler.ipynb)| + Intel VTune™ Profiler usage __in Intel DevCloud__ environment using command-line options<br>+ ___Lab Exercise___: VTune Profiling by collecting __gpu_hotspots__ for sample application.
|[Intel® Distribution for GDB on DevCloud](11_Intel_Distribution_for_GDB/gdb_oneapi.ipynb)| + Use the Intel® Distribution for GDB to debug kernels running on GPUs.

##### Learn SYCL Programming
The modules listed above include Introduction to oneAPI, oneAPI Tools, SYCL Programming and Libraries. A sub-set of these modules that only focus on SYCL programming are listed below. Use these modules to just learn SYCL Programming Basics:

| Modules | Skill Level| Description
|---|---|---|
|[SYCL Program Structure](02_SYCL_Program_Structure/SYCL_Program_Structure.ipynb)| basic| You will learn how to write a basic SYCL program that offloads computation to GPU<br>You will learn about SYCL Buffer Memory Model to manage data movement between host and device.
|[SYCL Unified Shared Memory](03_SYCL_Unified_Shared_Memory/Unified_Shared_Memory.ipynb)| basic | You will learn how to write a basic SYCL prgram that uses Unified Shared Memory(USM) which is pointer-based memory model.
|[SYCL Sub Groups](04_SYCL_Sub_Groups/Sub_Groups.ipynb)|intermediate|You will learn how to program to low-level hardware to take advantage of concurrent execution of parallel computation
|[SYCL Kernel Reductions](08_SYCL_Reduction/Reductions.ipynb)|intermediate|You will learn how to write SYCL code to perform reductions effeciently on accelerator devices.
|[SYCL Buffers and Accessors in depth](09_SYCL_Buffers_And_Accessors_Indepth/SYCL_Buffers_accessors.ipynb)| intermediate|You will learn more advanced properties of buffer memory model.
|[SYCL Task Scheduling and Data Dependences](10_SYCL_Graphs_Scheduling_Data_management/SYCL_Task_Scheduling_Data_dependency.ipynb)|intermediate|You will learn how data movement can be controlled in SYCL programs when using Buffers and USM.<br>You will also lean about graph scheduling with buffers memory model
|[SYCL Local Memory and Atomics](12_SYCL_Local_Memory_And_Atomics/LocalMemory_Atomics.ipynb)|intermediate|You will learn how to utilize device's Shared Local Memory to reduce latency in accessing data for kernel computation and Atomic operations to avoid data race conditions.

#### Content Structure

Each module folder has a Jupyter Notebook file (`*.ipynb`), this can be opened in Jupyter Lab to view the training contant, edit code and compile/run. Along with the Notebook file, there is a `lab` and a `src` folder with SYCL source code for samples used in the Notebook. The module folder also has `run_*.sh` files which can be used in shell terminal to compile and run each sample code.

## Install Directions

The training content can be accessed locally on the computer after installing necessary tools, or you can directly access using Intel DevCloud without any installation.

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


