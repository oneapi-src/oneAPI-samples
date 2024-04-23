## Title
 SYCL Performance Portability
  
## Requirements
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu 18.04, 20 Windows* 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler, Jupyter Notebooks, Intel Devcloud
  
## Purpose
The Jupyter Notebooks in this training shows challenges of heterogenous programming and why it is important to develop Performance Portable SYCL code. The Notebooks explain how various optimizations can be applied to make the SYCL code performance portable across CPUs and GPUs. Also, it familiarizes you with the use of Jupyter notebooks as a front-end for all training exercises. This workshop is designed to be used on the Devcloud and includes details on submitting batch jobs on the Devcloud environment.

At the end of this course you will be able to:
- write Performance Portable SYCL code that runs on CPUs and GPUs effenciently
- use analysis tools like Intel VTune Profile and Intel Advisor Roofline on SYCL applications

## License  
Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Content Details

#### Pre-requisites
- C++ Programming
- Basics of SYCL Programming

#### Training Modules

| Modules                     | Description
|:---                               |:---
| Introduction to Performance, Portability and Productivity | + Introduction to Performance, Portability and Productivity<br>+ Introduction to oneAPI<br>+ Test Application for Performance Portability<br>+ Analysis for Performance Portability
| Math Kernel Library (oneMKL) and SYCL Basic Parallel Kernel | + Matrix Multiplication with Math Kernel Library (oneMKL)<br>+ Matrix Multiplication with SYCL Basic Parallel Kernel
| ND-Range Implementation for Matrix Multiplication | + Matrix Multiplication with SYCL ND-Range Kernel<br>+ Matrix Multiplication with SYCL ND-Range Kernel using Private Memory
| Local Memory Implementation for Matrix Multiplication | + Matrix Multiplication with SYCL ND-Range Kernel and Shared Local Memory
| Analysis and Optimizing for Performance Portability | + Execution Time Analysis<br>+ Platform and Accelerator Capability<br>+ Impact of Work-group Sizes across different devices<br>+ Optimal Work-Group size for Performance Portability<br>+ Performance Portability Analysis

#### Content Structure

There are Jupyter Notebook files (`*.ipynb`) for each module, these can be opened in Jupyter Lab to view the training contant, edit code and compile/run. Along with the Notebook files, there is a `lab` and a `src` folder with SYCL source code for samples used in the Notebook. The module folder also has `run_*.sh` files which can be used in shell terminal to compile and run each sample code.

## Install Directions

The training content can be accessed locally on the computer after installing necessary tools, or you can directly access using Intel DevCloud without any installation.

#### Local Installation of JupyterLab and oneAPI Tools

The Jupyter Notebooks can be downloaded locally to computer and accessed:
- Install Jupyter Lab on local computer: [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- Install Intel oneAPI Base Toolkit on local computer: [Installation Guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) 
- git clone the repo and access the Notebooks using Jupyter Lab


#### Access using Intel DevCloud

The Jupyter notebooks are tested and can be run on Intel Devcloud without any installation necessary, below are the steps to access these Jupyter notebooks on Intel Devcloud:
1. Register on [Intel Devcloud](https://devcloud.intel.com/oneapi)
2. Login, Get Started and Launch Jupyter Lab
3. Open Terminal in Jupyter Lab and git clone the repo and access the Notebooks
