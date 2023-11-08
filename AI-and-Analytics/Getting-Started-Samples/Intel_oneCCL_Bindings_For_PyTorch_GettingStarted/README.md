# `oneCCL Bindings for PyTorch* Getting Started` Sample

The oneAPI Collective Communications Library Bindings for PyTorch* (oneCCL Bindings for PyTorch*) holds PyTorch bindings maintained by Intel for the Intel® oneAPI Collective Communications Library (oneCCL).

| Area                  | Description
|:---                   |:---
| What you will learn   | How to get started with oneCCL Bindings for PyTorch*
| Time to complete      | 60 minutes

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 22.04
| Hardware                          | Intel® Xeon® scalable processor family <br> Intel® Data Center GPU
| Software                          | Intel® Extension for PyTorch*


### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Tools**

  You can get the AI Tools from [the product page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html). <br> See [*Get Started with the Intel® AI Tools for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Tools installation information and post-installation steps and scripts.

  oneCCL Bindings for PyTorch* is ready for use once you finish the Intel® AI Tools installation via Offline Installer and have run the post-installation script.


  You can refer to the *[Get Started with the Intel® AI Tools for Linux*](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit)* for post-installation steps and scripts.


## Key Implementation Details

The sample code demonstrates distributed training using oneCCL in PyTorch*. oneCCL is a library for efficient distributed deep learning training that implements such collectives like `allreduce`, `allgather`, and `alltoall`. For more information on oneCCL, refer to the [*oneCCL documentation*](https://oneapi-src.github.io/oneCCL/).

This sample contains a Jupyter Notebook that guides you through the process of running a simple PyTorch* distributed workload on both GPU and CPU by using Intel® AI Tools.

The Jupyter Notebook also demonstrates how to change PyTorch* distributed workloads from CPU to the Intel® Data Center GPU family.

> **Note**: For comprehensive instructions regarding distributed training with oneCCL in PyTorch, see these GitHub repositories:
>
>- [Intel® oneCCL Bindings for PyTorch*](https://github.com/intel/torch-ccl) 
>- [Distributed Training with oneCCL in PyTorch*](https://github.com/intel/optimized-models/tree/master/pytorch/distributed)

## Purpose

From this sample code, you will learn how to perform distributed training with oneCCL in PyTorch*. The `oneCCL_Bindings_GettingStarted.ipynb` Jupyter Notebook targets both CPUs and GPUs using oneCCL Bindings for PyTorch*.


## Run the `oneCCL Bindings for PyTorch* Getting Started` Sample

If you have already set up the PIP or Conda environment and installed AI Tools go directly to Run the Notebook.
### Steps for Intel AI Tools Offline Installer  


> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

1. Read and follow the *Run Scripts and CPU Affinity* instructions at [https://github.com/intel/optimized-models/tree/master/pytorch/distributed#run-scripts--cpu-affinity](https://github.com/intel/optimized-models/tree/master/pytorch/distributed#run-scripts--cpu-affinity).

### Run the Jupyter Notebook

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
   ```
3. Follow the instructions to open the URL with the token in your browser.
4. Locate and select the Notebook.
   ```
   oneCCL_Bindings_GettingStarted.ipynb
   ```
5. Change your Jupyter Notebook kernel to **PyTorch** or **PyTorch-GPU**.
6. Run every cell in the Notebook in sequence.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
