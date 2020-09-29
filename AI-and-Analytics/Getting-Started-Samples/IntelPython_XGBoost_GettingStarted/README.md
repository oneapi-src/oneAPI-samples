# XGBoost Getting Started 
This code example provides sample code to run Intel's optimized XGBoost. It demonstrates how to use software products that can be found in the [Intel AI Analytics Toolkit powered by oneAPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html). 


## Key implementation details
The example uses Intel's XGBoost published as part of Intel oneAPI AI Analytics Toolkit. The example also illustrates how to setup and train an XGBoost model on datasets for prediction.

## Pre-requirement

XGBoost is ready for use once you finish the Intel AI Analytics Toolkit installation, and have run the post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation, and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

## Activate conda environment With Root Access

Please follow the Getting Started Guide steps (above) to set up your oneAPI environment with the setvars.sh script. Then navigate in linux shell to your oneapi installation path, typically `~/intel/inteloneapi`. Activate the conda environment with the following command:

#### Linux
```
source activate base
(base is default)
```


## Activate conda environment Without Root Access (Optional)

By default, the Intel AI Analytics toolkit is installed in the inteloneapi folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

#### Linux
```
conda create --name user_xgboost --clone base
```

Then activate your conda environment with the following command:

```
source activate user_xgboost
```

## Install Jupyter Notebook 
```
conda install jupyter nb_conda_kernels
```

## How to Build and Run 
1. Go to the code example location<br>
2. Enter command `jupyter notebook` if you have GUI support <br>
or<br>
2a. Enter command `jupyter notebook --no-browser --port=8888` on a remote shell <br>
2b. Open the command prompt where you have GUI support, and forward the port from host to client<br>
2c. Enter `ssh -N -f -L localhost:8888:localhost:8888 <userid@hostname>`<br>
2d. Copy-paste the URL address from the host into your local browser to open the jupyter console<br>
3. Go to `XGBoost_GettingStarted.ipynb` and run each cell to create sythetic data and run xgboost

## License  
This code sample is licensed under MIT license
