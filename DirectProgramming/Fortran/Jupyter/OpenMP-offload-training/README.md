# oneAPI OpenMP Offload Training Jupyter Notebooks
The content of this repository is a collection of Jupyter Notebooks developed to teach OpenMP* Offload. These Jupyter Notebooks are designed to run on Intel® DevCloud.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to offload complex computations to a GPU using OpenMP and the  Intel® Fortran Compiler
| Time to complete      | 2 Hours

## Prerequisites
| Optimized for         | Description
|:---                   |:---
| OS                    | Linux*
| Hardware              | Skylake with GEN9 or newer
| Software              | Intel® Fortran Compiler <br> Intel® DevCloud

## Access Intel® DevCloud
Download the Jupyter Notebooks on Intel® DevCloud by performing the following steps:

1. If you do not already have an account, request an Intel® DevCloud account at [Create an Intel® DevCloud Account](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).

2. On a Linux* system, open a terminal.

3. SSH into Intel® DevCloud.
   ```
   ssh DevCloud
   ```
   > **Note**: You can find information about configuring your Linux system and connecting to Intel DevCloud at Intel® DevCloud for oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started).

4. Download the oneAPI-essentials series of Jupyter Notebooks and OpenMP offload Notebooks into your Intel® DevCloud account.
   ```
   /data/oneapi_workshop/get_jupyter_notebooks.sh
   ```
   >**Note**: Since this downloads all oneAPI-essential Jupyter Notebooks, this is a one-time task.

## Open the OpenMP Jupyter Notebooks
1. Open a web browser, and navigate to https://devcloud.intel.com. Select **Work with oneAPI**.

2. From Intel® DevCloud for oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started), locate the ***Connect with Jupyter* Lab*** section (near the bottom).

3. Click **Sign in to Connect** button.

4. From the *Intel® oneAPI HPC Toolkit* section, select **View Training Modules**.

5. Select any OpenMP module, and click **Try it in Jupyter**. (JupyterLab should launch with selected Notebook open.)

### Open the Notebooks Directly
1. If the correct Notebook did not open, navigate to the home directory.

2. Navigate to **OpenMP Offload** > **Fortran**.

3. Open the Notebook.
   ```
   OpenMP Welcome.ipynb
   ```
4. Start the module of interest.

5.  Follow the instructions in each Notebook, and execute cells when instructed.

# Summary of the Jupyter Notebook Directories
The *oneAPI OpenMP* Offload Modules in Fortran* contains training and conceptual information for the following areas in the *Developer Training Modules* section of the Notebook.

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

## License
Code samples are licensed under the MIT license. See [License.txt](License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt).
