# Predictive Modeling with XGBoost and the Intel AI Kit 

The purpose of this repo is to be the central aggregation, curation, and
distribution point for Juypter notebooks that are developed in support of
the Intel AI Kit. These initial hands-on exercises introduce you to predictive modeling using decision trees, bagging and XGBoost.
The notebooks for the exercises are stored under AI_Kit_XGBoost_Predictive_Modeling and the answers to these exercises are stored
under AI_Kit_XGBoost_Predictive_Modeling.complete.

The Jupyter notebooks are tested and can be run on the Intel Devcloud. Below
are the steps to access these Jupyter notebooks on the Intel Devcloud:

## Intel Devcloud

1. <a href="https://devcloud.intel.com/oneapi/get_started/">Register with the Intel Devcloud.</a>
  
2. After you receive your credentials, use SSH via a terminal or <a href="https://jupyter.oneapi.devcloud.intel.com/hub/login?next=/lab/tree/Welcome.ipynb?reset">Jupyter lab to connect.</a>

3. From a terminal, enter the following command to obtain the latest series of Jupyter notebooks into your devcloud account:  *Note if you are setting up    your account for the first time this script will run automatically.
   ```bash
   /data/oneapi_workshop/get_jupyter_notebooks.sh
   ```
4. From the Jupyter file navigation panel on the left select Predictive_Modeling_Training and choose a module.  The notebooks end in `.ipynb `


## Running the Jupyter Notebooks Locally on a Linux Machine OR WSL:
1. Update your system:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
2. After the update a reboot will be required, enter:
   ```bash
   sudo reboot
   ```
3. Download and Install <a href="https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html">Intel® oneAPI Base Toolkit and Intel® AI Analytics Toolkit (AI Kit).</a>
  
4. Post install, you will need to refresh your new environment variables.
   ```bash
   source .bashrc
   ```
5. In order to initialize the oneAPI environment enter:
   ```bash
   source /opt/intel/oneapi/setvars.sh   
   ```
6. Install Jupyterlab--we are cloning our base environment so that we can always get back to a clean start.
   ```bash
   conda create --clone base --name jupyter
    ```
7. Switch to the newly created environment by entering:
    ```bash
    conda activate jupyter
    ```
      
8. Install Jupyterlab:
   ```bash
   conda install -c conda-forge jupyterlab
   ```
9.  Clone the Intel oneAPI Samples Repository, if Git is not installed, enter the following:
    ```bash
    sudo apt install git
    ```
    Now clone the Samples
    ```bash
    git clone https://github.com/oneapi-src/oneAPI-samples.git
    ```
10. Launch JupyterLab by entering:
    ```bash 
    jupyter lab
    ``` 
    in the terminal. 
11. Make a note of the address printed on the terminal and paste it into your browser address bar.
12. Once Jupyterlab opens up, navigate to 
    ```bash
    ~/oneAPI-samples/AI-and-Analytics/Jupyter/Predictive_Modeling_Training
    ```

## License

Code samples are licensed under the MIT license. See <a href="https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt">License.txt</a>
for details.

Third party program Licenses can be found here: <a href="https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt">third-party-programs.txt</a>

# Organization of the Jupyter Notebook Directories

| Notebook Name | Owner | Description |
|---|---|---|
|<a href="https://github.com/bjodom/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling.complete/00_Local_Setup">00_Local_Setup</a>|benjamin.j.odom@intel.com| - Howto setup the environment for running on a local machine: <br>- Anaconda Setup <br>- Intel Distribution of Python<br>- Intel AI Kits<br>- Intel Data Science Workstation Kit|
|<a href="https://github.com/bjodom/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling.complete/01_Decision_Trees">01_Decision_Trees</a>|benjamin.j.odom@intel.com| - Recognize Decision trees and how to use them for classification problems. <br>- Recognize how to identify the best split and the factors for splitting. <br>- Explain strengths and weaknesses of decision trees. <br>- Explain how regression trees help with classifying continuous values. <br>- Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware.|
|<a href="https://github.com/bjodom/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling.complete/02_Bagging">02_Bagging</a>|benjamin.j.odom@intel.com|- Determine if stratefiedshuffle split is the best approach. <br>- Recognize how to identify the optimal number of trees <br>- Understand the resulting plot of out-of-band errors. <br>- Explore Random Forest vs Extra Random Trees and determine which one worked better. <br>- Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware.|
|<a href="https://github.com/bjodom/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling.complete/03_XGBoost">03_XGBoost</a>|benjamin.j.odom@intel.com|- Utilize XGBoost with Intel's AI KIt <br>- Take advantage of Intel extensions to SciKit Learn by enabling them with XGBoost. <br>- Use Cross Validation technique to find better XGBoost Hyperparameters. <br>- Use a learning curve to estimate the ideal number of trees. <br>- Improve performance by implementing early stopping.|
<a href="https://github.com/bjodom/oneAPI-samples/blob/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling.complete/04_oneDal/XGBoost-oneDal.ipynb">04_oneDal</a>|benjamin.j.odom@intel.com|- Utilize XGBoost with Intel's AI KIt <br>- Take advantage of Intel extensions to SciKit Learn by enabling them with XGBoost. <br>- Utilize oneDaal to enhance prediction performance. 
