# oneAPI OpenMP Offload Training Jupyter Notebooks

The the content of this repo is a collection of Jupyter notebooks that were
developed to teach OpenMP Offload.

The Jupyter notebooks are tested and can be run on the Intel Devcloud. Below
are the steps to access these Jupyter notebooks on the Intel Devcloud:

1. Register with the Intel Devcloud at
   https://intelsoftwaresites.secure.force.com/devcloud/oneapi

2. SSH into the Intel Devcloud "terminal"

3. Type the following command to download the oneAPI-essentials series of
   Jupyter notebooks and OpenMP offload notebooks into your devcloud account
   `/data/oneapi_workshop/get_jupyter_notebooks.sh`

| Optimized for         | Description
|:---                   |:---
| OS                    | Linux*
| Hardware              | Skylake with GEN9 or newer
| Software              | Intel&reg; C++ Compiler
| License               | Samples licensed under MIT license.
| What you will learn   | How to offload the computation to GPU using OpenMP with the Intel&reg; C++ Compiler
| Time to complete      | 2 Hours

## Running the Jupyter Notebooks

1. Open "OpenMP Welcome.ipynb" with JupyterLab
2. Start the modules of interest
3. Follow the instructions in each notebook and execute cells when instructed.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# Summary of the Jupyter Notebook Directories

[OpenMP Welcome](OpenMP&#32;Welcome.ipynb)
* Introduce Developer Training Modules
* Describe oneAPI Tool Modules

[Introduction to OpenMP Offload](intro) 
* oneAPI Software Model Overview and Workflow
* HPC Single-Node Workflow with oneAPI
* Simple OpenMP Code Example
* Target Directive Explanation
* _Lab Exercise_: Vector Increment with Target Directive

[Managing Data Transfers](datatransfer) 
* Offloading Data
* Target Data Region
* _Lab Exercise_: Target Data Region
* Mapping Global Variable to Device

[Utilizing GPU Parallelism](parallelism) 
* Device Parallelism
* OpenMP Constructs and Teams
* Host Device Concurrency
* _Lab Exercise_: OpenMP Device Parallelism

[Unified Shared Memory](USM) 
* Allocating Unified Shared Memory
* USM Explicit Data Movement
* _Lab Exercise_: Unified Shared Memory
