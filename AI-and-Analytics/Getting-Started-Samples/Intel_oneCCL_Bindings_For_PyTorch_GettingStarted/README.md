# `oneCCL Bindings for PyTorch* Getting Started` Sample

The oneAPI Collective Communications Library Bindings for PyTorch* (oneCCL Bindings for PyTorch*) holds PyTorch bindings maintained by Intel for the Intel® oneAPI Collective Communications Library (oneCCL).

| Area                  | Description
|:---                   |:---
| What you will learn   | How to get started with oneCCL Bindings for PyTorch*
| Time to complete      | 60 minutes

## Purpose

From this sample code, you will learn how to perform distributed training with oneCCL in PyTorch*. The `oneCCL_Bindings_GettingStarted.ipynb` Jupyter Notebook targets both CPUs and GPUs using oneCCL Bindings for PyTorch*.

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Ubuntu* 22.04
| Hardware                          | Intel® Xeon® scalable processor family <br> Intel® Data Center GPU
| Software                          | Intel® Extension for PyTorch (IPEX)

> **Note**: AI and Analytics samples are validated on AI Tools Offline Installer. For the full list of validated platforms refer to [Platform Validation](https://github.com/oneapi-src/oneAPI-samples/tree/master?tab=readme-ov-file#platform-validation).

## Key Implementation Details

The sample code demonstrates distributed training using oneCCL in PyTorch*. oneCCL is a library for efficient distributed deep learning training that implements such collectives like `allreduce`, `allgather`, and `alltoall`. For more information on oneCCL, refer to the [*oneCCL documentation*](https://oneapi-src.github.io/oneCCL/).

This sample contains a Jupyter Notebook that guides you through the process of running a simple PyTorch* distributed workload on both GPU and CPU by using Intel® AI Tools.

The Jupyter Notebook also demonstrates how to change PyTorch* distributed workloads from CPU to the Intel® Data Center GPU family.

> **Note**: For comprehensive instructions regarding distributed training with oneCCL in PyTorch, see these GitHub repositories:
>
>- [Intel® oneCCL Bindings for PyTorch*](https://github.com/intel/torch-ccl) 
>- [Distributed Training with oneCCL in PyTorch*](https://github.com/intel/optimized-models/tree/master/pytorch/distributed)


## Run the `oneCCL Bindings for PyTorch* Getting Started` Sample

Go to the section which corresponds to the installation method chosen in [AI Tools Selector](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-tools-selector.html) to see relevant instructions:
* [AI Tools Offline Installer (Validated)](#ai-tools-offline-installer-validated)
* [Docker](#docker)

### AI Tools Offline Installer (Validated)  
1. If you have not already done so, activate the AI Tools bundle base environment. If you used the default location to install AI Tools, open a terminal and type the following
```
source $HOME/intel/oneapi/intelpython/bin/activate
```
If you used a separate location, open a terminal and type the following
```
source <custom_path>/bin/activate
```
2. Clone the GitHub repository and install required packages:
``` 
git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples/AI-and-Analytics/Getting-Started-Samples/Intel_oneCCL_Bindings_For_PyTorch_GettingStarted/
pip install -r requirements.txt
```
3. Launch Jupyter Notebook.
```
jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
```
4. Follow the instructions to open the URL with the token in your browser.
5. Locate and select the Notebook.
     ```
     oneCCL_Bindings_GettingStarted.ipynb
     ```
6. Change your Jupyter Notebook kernel to **PyTorch** or **PyTorch-GPU**.
7. Run every cell in the Notebook in sequence.

### Docker
AI Tools Docker images already have Get Started samples pre-installed. Refer to [Working with Preset Containers](https://github.com/intel/ai-containers/tree/main/preset) to learn how to run the docker and samples.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
