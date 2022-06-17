# Numba Data parallel python training Jupyter notebooks

The purpose of this repo is to be the central aggregation, curation, and
distribution point for Juypter notebooks that are developed in support of
Numba Data parallel essentials for python training programs. These initial hands-on exercises introduce you to concepts of Data Parallel Python. In addition, it familiarizes you how to execute on multiple devices using Data Parallel essentials for Python (dpex), utilize Numba and Numba-dpex to write paralle code on GPU.

The Jupyter notebooks are tested and can be run on the Intel Devcloud. Below
are the steps to access these Jupyter notebooks on the Intel Devcloud:

1. Register with the Intel Devcloud at
   https://intelsoftwaresites.secure.force.com/devcloud/oneapi

2. SSH into the Intel Devcloud "terminal"

3. Type the following command to download the Numba Data parallel Python series of
   Jupyter notebooks into your devcloud account
   `/data/oneapi_workshop/get_jupyter_notebooks.sh`
   
### Running the Jupyter Notebooks locally on a Linux machine OR WSL:
1. Update your system:
   sudo apt update && sudo apt upgrade -y
2. After the update a reboot will be required, enter:
   sudo reboot
3. Download and Install Intel® oneAPI Base Toolkit and Intel® oneAPI AI Analytics Toolkit.
   https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html
4. You can close the terminal and launch a new one and enter:
   source .bashrc
   which will result in the base conda environment being activated.
5. Enter:
   conda env list
   Note: if Conda not recognized enter
   source /opt/intel/oneapi/setvars.sh   
6. Launch a terminal and enter:
    conda create –-clone base –-name <pick something> for example:
    conda create --clone base --name jupyter
7. Conda env list:
    You should see two environments now.  The * denotes the active environment.  
    Activate the new environment:
    Conda activate jupyter    
8. Install Jupyterlab:
   conda install -c conda-forge jupyterlab    
9. Clone the Intel oneAPI Samples Repository, Git will likely not be installed so to install it enter:
    sudo apt install git
    git clone https://github.com/oneapi-src/oneAPI-samples.git    
10. Launch JupyterLab by typing in "jupyter lab" in the terminal. 
11. Make a note of the address printed on the terminal and paste it in the browser window.
12. JupyterLab opens up and navigate to ~/oneAPI-samples/AI-and-Analytics/Jupyter/Numba_DPPY_Essentials_training and double click on "Welcome.ipynb" to get started.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# Organization of the Jupyter Notebook Directories

| Notebook Name | Owner | Description |
|---|---|---|
|[dpex Intro](01_dpex_Intro)|Praveen.K.Kundurthy@intel.com| + Introduction and Motivation for Data parallel python: <br>+ Intro to Numba-dpex.<br>+ @njit Decorator:Explicit and Implicit offload.<br>+ @dppy.kernel decorator. <br>+ _Lab Excercise_: Matrix multiplication using numba_dppy.|
|[DPCTL Intro](02_dpctl_Intro)|Praveen.K.Kundurthy@intel.com|+ __Classes__ - device, device_selector, queue using dpctl. <br>+ USM and memory management using __dpctl__.|
|[dpex Pairwise Distance Algorithm](03_dpex_Pairwise_Distance)|Praveen.K.Kundurthy@intel.com| + Pairwise distance algorithm targeting CPU and GPU using __Numba__ Jit decorator.<br>+Pairwise distance algorithm targeting GPU using __Kernel__ decorator.<br>+ Pairwise distance algorithm targeting GPU using __Numpy__ approach.|
|[dpex Black Scholes Algorithm](04_dpex_Black_Sholes)|Praveen.K.Kundurthy@intel.com|+ Black Scholes algorithm targeting CPU and GPU using __Numba_ Jit decorator.<br>+ Black Scholes algorithm targeting GPU using __Kernel__ decorator.<br>+ Black Scholes algorithm targeting GPU using __Numpy__ approach.|
|dpex K-Means Algorithm](05_dpex_Kmeans)|Praveen.K.Kundurthy@intel.com|<br>+ K-Means algorithm targeting CPU and GPU using __Numba__ Jit decorator.<br>+ K-Means algorithm targeting GPU using __Kernel__ decorator.<br>+ K-Means algorithm targeting GPU using __Numpy__ and __Atomics__.|
|[dpex Gpairs Algorithm](05_dpex_GPAIRS)|Praveen.K.Kundurthy@intel.com|<br>+ Gpairs algorithm targeting CPU and GPU using __Numba__ Jit decorator.<br>+ Gpairs algorithm targeting GPU using __Kernel__ decorator.|

__Note__: Please take care to secure the connection while using functions like __Pickle__ and establish an appropriate trust verification mechanism.
