# Numba Data Parallel Python* Training Jupyter Notebooks

The purpose of this repo is to be the central aggregation, curation, and distribution point for Jupyter Notebooks that are developed in support of Numba data parallel essentials for Python training programs.

These initial hands-on exercises introduce you to concepts of Data Parallel Python*. In addition, these exercises demonstrate how you can execute on multiple devices using Data Parallel essentials for Python, and how to use Numba and Numba-dpex to write parallel code for GPUs.

## Prerequisites

| Optimized for             | Description
|:---                       |:---
| OS                        | Ubuntu* 20.04 (or newer) <br> Windows Subsystem for Linux (WSL)
| Software                  | Intel® oneAPI Base Toolkit (Base Kit) <br> Intel® AI Analytics Toolkit (AI Kit)

The Jupyter Notebooks are tested for and can be run on the Intel® Devcloud for oneAPI.

## Jupyter Notebook Directories and Descriptions

| Notebook Directory and Name                                                          | Notebook Focus
|:---                                                                                  |:---
|[`00_dpex_Prerequisites\Setup_Instructions_Numba.ipynb`](00_dpex_Prerequisites)       | - Setup instructions
|[`01_dpex_Intro\dpex_Intro.ipynb`](01_dpex_Intro)                                     | - Introduction and motivation for Data Parallel Python <br> - Intro to Numba-dpex <br> - @njit Decorator:Explicit and Implicit offload <br> - @dppy.kernel decorator <br> - **Lab Excercise**: Matrix multiplication using numba_dppy
|[`02_dpctl_Intro\dpex_dpCtl.ipynb`](02_dpctl_Intro)                                   | - **Classes** - device, device_selector, queue using dpctl. <br> - USM and memory management using **dpctl**.
|[`03_dpex_Pairwise_Distance\dpex_Pairwise_Distance.ipynb`](03_dpex_Pairwise_Distance) | - Pairwise distance algorithm targeting CPU and GPU using **Numba** JIT decorator <br> - Pairwise distance algorithm targeting GPU using **Kernel** decorator <br> - Pairwise distance algorithm targeting GPU using **NumPy** approach
|[`04_dpex_Black_Sholes\dpex_Black_Sholes.ipynb`](04_dpex_Black_Sholes)                | - Black-Scholes algorithm targeting CPU and GPU using **Numba** JIT decorator <br> - Black-Scholes algorithm targeting GPU using **Kernel** decorator <br> - Black-Scholes algorithm targeting GPU using **NumPy** approach
|[`05_dpex_Kmeans\dpex_Kmeans.ipynb`](05_dpex_Kmeans)                                  | - K-Means algorithm targeting CPU and GPU using **Numba** JIT decorator <br> - K-Means algorithm targeting GPU using **Kernel** decorator <br> - K-Means algorithm targeting GPU using **NumPy** and **Atomics**
|[`06_dpex_GPAIRS\dpex_Gpairs.ipynb`](06_dpex_GPAIRS)                                  | - Gpairs algorithm targeting CPU and GPU using **Numba** JIT decorator <br> - Gpairs algorithm targeting GPU using **Kernel** decorator
|[`07_dpex_KNN\dpex_KNN.ipynb`](07_dpex_KNN)                                           | -  K Nearest Neighbours algorithm using Numba-dpex
|[`08_dpex_reductions\dpex_reductions.ipynb`](08_dpex_reductions)                      | - Reductions and local memory in numba-dpex

>**Note**: Secure the connection while using functions like **Pickle**, and establish an appropriate trust verification mechanism.


## Run the Jupyter Notebooks Locally (on Linux* or WSL)

1. Update the package manager on your system.
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. After the update, reboot your system.
   ```bash
   sudo reboot
   ```
3. Download and install Intel® oneAPI Base Toolkit (Base Kit) and Intel® AI Analytics Toolkit (AI Kit) from the [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html) page.

4. After you complete the installation, refresh the new environment variables. This results in the base conda environment being activated.
   ```bash
   source .bashrc
   ```
5. Configure the variables.
   ```bash
   conda env list
   ```
   > **Note**: if Conda not recognized, source the environment manually.
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```
6. Open a terminal and enter the following command.
   ```bash
   conda create --clone base --name jupyter
   ```
7. List the conda environments.
   ```bash
   conda list
   ```
   You should see two environments now.  The * denotes the active environment.  

8. Activate the new environment.
   ```bash
   conda activate jupyter
   ```
9. Install Jupyterlab.
   ```bash
   conda install -c conda-forge jupyterlab
   ```
10. Clone the oneAPI-samples GitHub repository.

    >**Note**: If Git is not installed, install it now.
    >```bash
    >sudo apt install git
    >```

    ```bash
    git clone https://github.com/oneapi-src/oneAPI-samples.git
    ```

11. From a terminal, start JupyterLab.
    ```bash
    jupyter lab
    ```

12. Make note of the address printed in the terminal, and paste the address into your browser address bar.

13. Once Jupyterlab opens, navigate to the following directory.
    ```bash
    ~/oneAPI-samples/AI-and-Analytics/Jupyter/Numba_DPPY_Essentials_training
    ```

14. Locate and open th `Welcome.ipynb`.


## Run the Jupyter Notebooks on Intel® Devcloud

Use these general steps to access the notebooks on the Intel® Devcloud for oneAPI.

>**Note**: For more information on using Intel® DevCloud, see the Intel® oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started/) page.

1. If you do not already have an account, request an Intel® DevCloud account at [Create an Intel® DevCloud Account](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).

2. Once you get your credentials, open a terminal on a Linux* system.

3. Log in to the Intel® DevCloud.
   ```
   ssh devcloud
   ```
   >**Note**: Alternatively, you can use the Intel [JupyterLab](https://jupyter.oneapi.devcloud.intel.com/hub/login?next=/lab/tree/Welcome.ipynb?reset) to connect with your account credentials.

4. Enter the following command to download the Numba Data Parallel Python series of Jupyter Notebooks into your account.
   ```
   /data/oneapi_workshop/get_jupyter_notebooks.sh
   ```

5. From the navigation panel, navigate through the directory structure and select a Notebook to run.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).