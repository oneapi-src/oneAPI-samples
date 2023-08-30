# `Enable distrubted deep learning using Intel® Optimization for Horovod and Tensorflow*` Sample

The `Enable distrubted inference using Intel® Optimization for Horovod and Tensorflow*` sample guides you through the process of how to run inference & training workloads across multi-cards using Intel Optimization for Horovod and TensorFlow* on Intel® dGPU's.


| Area                    | Description
|:---                     |:---
| What you will learn     | Enable distrubted deep learning using Intel Optimization for Horovod and Tensorflow*
| Time to complete        | 10 minutes
| Category                | Code Optimization

## Purpose

Through the implementation of end-to-end deep learning example, this sample demonstrates important concepts:
- The performance benefits of distrubuting deep learning workload among multiple dGPUs

## Prerequisites

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux; Ubuntu* 18.04 or newer
| Hardware                          | Intel® Data Center GPU Max/Flex Series 
| Software                          | Intel® AI Analytics Toolkit (AI Kit)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

- **Jupyter Notebook**

  Install using PIP: `$pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.


### For Intel® Developer Cloud (Beta)

The necessary tools and components are already installed in the environment other than *intel-optimization-for-horovod* package. See [Intel® Developer Cloud for oneAPI](https://github.com/bjodom/idc) for information.

## Key Implementation Detailes

### Jupyter Notebook

| Notebook                                                         | Description
|:---                                                              |:---
|`tensorflow_distributed_inference_with_horovod.ipynb` | Enabling Multi-Card Inference/Training with Intel® Optimizations for Horovod

## Run the distrubuted inference sample using Intel® Optimization for Horovod and Tensorflow: 

### On Linux*

1. Set up oneAPI environment by running setvars.sh script
  Default installation: `source /opt/intel/oneapi/setvars.sh`

  or `source /path/to/oneapi/setvars.sh`

3. Set up conda environment.
   ```
   conda create --name tensorflow_xpu --clone tensorflow-gpu
   conda activate tensorflow_xpu
   ```
4. Install dependencies:
   If you havent already done so, you will need to install *Jupyter notebook* and *Intel® Optimization for Horovod*
   
   ```
   pip install intel-optimization-for-horovod
   ```

   ```
   pip install notebook
   ```

6. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0
   ```
7. Follow the instructions to open the URL with the token in your browser.
8. Locate and select the Notebook.
   ```
   tensorflow_distributed_inference_with_horovod.ipynb
   ````
9. Change your Jupyter Notebook kernel to **tensorflow_xpu**.
10. Run every cell in the Notebook in sequence.


### Run the Sample on Intel® Developer Cloud (Optional)

1. If you do not already have an account, follow the readme to request an Intel® Developer Cloud account at [*Setup an Intel® Developer Cloud Account*](https://github.com/bjodom/idc).
2. On a Linux* system, open a terminal.
3. SSH into Intel® Developer Cloud.
   ```
   ssh idc
   ```
4. Run oneAPI setvars script.
   `source /opt/intel/oneapi/setvars.sh`

5. Activate the prepared `tensorflow_xpu` enviornment.
   ```
   conda activate tensorflow_xpu
   ```
6. Install Intel® Optimizations for Horovod
   ```
   pip install intel-optimization-for-horovod
   ```
   
7. Follow the instructions [here](https://github.com/bjodom/idc#jupyter) to launch a jupyter notebook on the Intel® developer cloud.
8. Locate and select the Notebook.
   ```
   tensorflow_distributed_inference_with_horovod.ipynb
   ````
9. Change the kernel to **tensorflow_xpu**.
10. Run every cell in the Notebook in sequence.


#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
