## Title
 
 CUDA To SYCL Migration
  
## Requirements
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 18.04, 20 Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler, Jupyter Notebooks, Intel DevCloud
  
## Purpose
C++ and SYCL* deliver a unified programming model, performance portability, and C++ alignment for applications using accelerators from differnet vendors.

This course will show how CUDA code can be migrated to SYCL code using SYCLomatic Tool, starting with Introduction to SYCL migration, SYCLomatic tool installation, SYCLomatic tool options and usage, compiling SYCL code for different accelerators and tips for migration to SYCL using SYCLomatic.

The different modules have SYCL migration process starting with the most basic CUDA examples and then goes to more complex CUDA projects to CUDA projects using different CUDA features and libraries.

At the end of this course you will be able to:

- Understand the advantages of migrating to SYCL
- Migrate a CUDA application to SYCL application and compile/run on different accelerators like CPU or GPU from different vendors.

## License  
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Content Details

#### Pre-requisites
- CUDA Development Machine
- CUDA Programming
- Basics of C++ SYCL Programming

#### Training Modules

| Modules | Description
|---|---|
|[SYCL Migration - Introduction](00_SYCL_Migration_Introduction/00_SYCL_Migration_Introduction.ipynb)| + CUDA to SYCL Migration Introduction<br>+ SYCLomatic Tool Introduction and Usage<br>+ Migration Workflow Overview
|[SYCLMigration - Simple VectorAdd](01_SYCL_Migration_Simple_VectorAdd/01_SYCL_Migration_Simple_VectorAdd.ipynb)|+ Learn how to migrate a simple single source CUDA code to SYCL.
|[SYCLMigration - Sorting Networks](02_SYCL_Migration_SortingNetworks/02_SYCL_Migration_SortingNetworks.ipynb)|+ Learn how to migrate a CUDA project with multiple sources files that uses Makefile for the project.
|[SYCLMigration - Jacobi Iterative](03_SYCL_Migration_Jacobi_Iterative/03_SYCL_Migration_Jacobi_Iterative.ipynb)|+ Learn how to migrate a CUDA project that using CUDA features to access the GPU hardware like Shared Local Memory, warps and atomics in kernel code.
|[SYCLMigration - Matrix Multiplication with CuBlas library](04_SYCL_Migration_MatrixMul_CuBlas/04_SYCL_Migration_MatrixMul_CuBlas.ipynb)|+ Learn how to migrate a CUDA project that uses CUDA library like cuBLAS.

#### Content Structure

Each module folder has a Jupyter Notebook file (`*.ipynb`), this can be opened in Jupyter Lab to view the training contant and compile/run. Along with the Notebook file, the module folders have sub-folders which has various versions or migrated SYCL source code, the module folder also has `run_*.sh` files which can be used in shell terminal to compile and run migrated SYCL code. The module folder structure is as shown below:

- `<MODULE NAME>`
    - `<MODULE NAME>.ipynb` - Jupyter Notebook with training content
    - `dpct_output` - SYCL code output from SYCLomatic tool
    - `sycl_migrated` - SYCL code modified/fixed to make it compile/run
    - `sycl_migrated_optimized` - SYCL code optimized for performance
    - `run_sycl_migrated.sh` - script to compile/run migrated SYCL code
    - `run_sycl_migrated_optimized.sh`  - script to compile/run migrated optimized SYCL code

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


