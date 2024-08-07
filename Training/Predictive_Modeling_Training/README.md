# Predictive Modeling with XGBoost* and the Intel® AI Analytics Toolkit (AI Kit)

The purpose of this repository is to be the central aggregation, curation, and distribution point for Juypter Notebooks that are developed in support of the Intel® AI Analytics Toolkit (AI Kit). These initial hands-on exercises introduce you to predictive modeling using decision trees, bagging, and XGBoost.

The Jupyter Notebooks for the exercises are in the `AI_Kit_XGBoost_Predictive_Modeling` folder, and the answers to these exercises in the `AI_Kit_XGBoost_Predictive_Modeling.complete` folder.

## Prerequisites

| Optimized for             | Description
|:---                       |:---
| OS                        | Ubuntu* 20.04 (or newer) <br> Windows Subsystem for Linux (WSL)
| Software                  | Intel® oneAPI Base Toolkit (Base Kit) <br> Intel® AI Analytics Toolkit (AI Kit)

The Jupyter Notebooks are tested for and can be run on the Intel® Devcloud for oneAPI.

## Jupyter Notebook Directories and Descriptions

The referenced folders and Notebooks are in the [`AI_Kit_XGBoost_Predictive_Modeling`](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling) folder. The `AI_Kit_XGBoost_Predictive_Modeling.complete` folder has the same structure.

| Notebook Directory and Name             | Notebook Focus
|:---                                     |:---
|[`00_Local_Setup\Local_Setup.ipynb`](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling/00_Local_Setup) | - How to setup the environment for running on a local machine <br> - Anaconda setup <br> -  Intel® Distribution for Python* programming language <br> - Intel® AI Analytics Toolkit (AI Kit) <br> - Intel data science workstation kits
|[`01_Decision_Trees\Decision_Trees.ipynb`](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling/01_Decision_Trees) | - Recognize decision trees and how to use them for classification problems <br> - Recognize how to identify the best split and the factors for splitting. <br> - Explain strengths and weaknesses of decision trees <br> - Explain how regression trees help with classifying continuous values <br> - Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware
|[`02_Bagging\Bagging_RF.ipynb`](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling/02_Bagging) | - Determine if stratefiedshuffle split is the best approach <br> - Recognize how to identify the optimal number of trees <br> - Understand the resulting plot of out-of-band errors <br> - Explore Random Forest vs Extra Random Trees and determine which one worked better <br> - Apply Intel® Extension for Scikit-learn* to leverage underlying compute capabilities of hardware
| [`03_XGBoost\XGBoost.ipynb`](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling/03_XGBoost)| - Use XGBoost with the AI Kit <br> - Take advantage of Intel® Extension for Scikit-learn* by enabling them with XGBoost <br> - Use Cross Validation technique to find better XGBoost Hyperparameters <br> - Use a learning curve to estimate the ideal number of trees <br> - Improve performance by implementing early stopping
| [`04_oneDal\XGBoost-oneDal.ipynb`](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter/Predictive_Modeling_Training/AI_Kit_XGBoost_Predictive_Modeling/04_oneDal)| - Utilize XGBoost with the AI KIt <br> - Take advantage of Intel® Extension for Scikit-learn* by enabling them with XGBoost <br> - Use Intel® oneAPI Data Analytics Library (oneDAL) to enhance prediction performance

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

4. After you complete the installation, refresh the new environment variables.
   ```bash
   source .bashrc
   ```

5. Initialize the oneAPI environment enter.
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

6. Install JupyterLab*. (In this case, we are cloning our base environment so that we can always get back to a clean start.)
   ```bash
   conda create --clone base --name jupyter
   ```

7. Switch to the newly created environment.
   ```bash
   conda activate jupyter
   ```

8. Install Jupyterlab.
   ```bash
   conda install -c conda-forge jupyterlab
   ```

9. Clone the oneAPI-samples GitHub repository.

   >**Note**: If Git is not installed, install it now.
   >```bash
   >sudo apt install git
   >```

   ```bash
   git clone https://github.com/oneapi-src/oneAPI-samples.git
   ```

10. From a terminal, start JupyterLab.
    ```bash
    jupyter lab
    ```

11. Make note of the address printed in the terminal, and paste the address into your browser address bar.

12. Once Jupyterlab opens, navigate to the following directory.
    ```bash
    ~/oneAPI-samples/AI-and-Analytics/Jupyter/Predictive_Modeling_Training
    ```

13. From the navigation panel, navigate through the directory structure and select a Notebook to run. (The notebooks have a `.ipynb ` extension.)


## Run the Jupyter Notebooks on Intel® Devcloud (Optional)

Use these general steps to access the notebooks on the Intel® Devcloud for oneAPI.

>**Note**: For more information on using Intel® DevCloud, see the Intel® oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started/) page.

1. If you do not already have an account, request an Intel® DevCloud account at [Create an Intel® DevCloud Account](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).

2. Once you get your credentials, open a terminal on a Linux* system
3. Log in to the Intel® DevCloud.
   ```
   ssh devcloud
   ```
   >**Note**: Alternatively, you can use the Intel [JupyterLab](https://jupyter.oneapi.devcloud.intel.com/hub/login?next=/lab/tree/Welcome.ipynb?reset) to connect with your account credentials.

4. From a terminal, enter the following command to obtain the latest series of Jupyter Notebooks into your Intel® DevCloud account:
   ```bash
   /data/oneapi_workshop/get_jupyter_notebooks.sh
   ```
   > **Note**: If you are setting up your account for the first time this script will run automatically.

5. From the navigation panel, navigate through the directory structure and select a Notebook to run. (The notebooks have a `.ipynb ` extension.)

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).