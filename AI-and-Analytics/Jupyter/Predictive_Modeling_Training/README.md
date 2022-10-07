# Predictive Modeling with XGBoost and the Intel AI Kit Jupyter notebooks

The purpose of this repo is to be the central aggregation, curation, and
distribution point for Juypter notebooks that are developed in support of
the Intel AI Kit. These initial hands-on exercises introduce you to predictive modeling using decision trees, bagging and XGBoost.  

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
3. Download and Install Intel® oneAPI Base Toolkit and Intel® AI Analytics Toolkit (AI Kit).
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
|[Local Setup](00_Local_Setup)|benjamin.j.odom@intel.com| - Howto setup the environment for running on a local machine: <br>- Anaconda Setup <br>- Intel Distribution of Python<br>- Intel AI Kits<br>- Intel Data Science Workstation Kit|
|[Decision Trees](01_Decision_Trees)|benjamin.j.odom@intel.com| - Recognize Decision trees and how to use them for classification problems. <br>- Recognize how to identify the best split and the factors for splitting. <br>- Explain strengths and weaknesses of decision trees. <br>- Explain how regression trees help with classifying continuous values. <br>- Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware.|
|[Bagging](02_Bagging)|benjamin.j.odom@intel.com|- Determine if stratefiedshuffle split is the best approach. <br>- Recognize how to identify the optimal number of trees <br>- Understand the resulting plot of out-of-band errors. <br>- Explore Random Forest vs Extra Random Trees and determine which one worked better. <br>- Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware.|
|[XGBoost](03_XGBoost)|benjamin.j.odom@intel.com|- Utilize XGBoost with Intel's AI KIt <br>- Take advantage of Intel extensions to SciKit Learn by enabling them with XGBoost. <br>- Use Cross Validation technique to find better XGBoost Hyperparameters. <br>- Use a learning curve to estimate the ideal number of trees. <br>- Improve performance by implementing early stopping.|

