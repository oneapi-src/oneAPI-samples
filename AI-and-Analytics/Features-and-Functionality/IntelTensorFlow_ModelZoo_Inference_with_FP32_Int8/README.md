# `Intel® AI Reference models for TensorFlow* Inference With FP32 Int8` Sample

The `Intel® AI Reference models for TensorFlow* Inference` sample demonstrates how to run ResNet50 inference on pretrained FP32 and Int8 models included in the Reference models for Intel® Architecture.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to perform TensorFlow* ResNet50 inference on synthetic data using FP32 and Int8 pre-trained models.
| Time to complete      | 30 minutes
| Category              | Code Optimization

## Purpose

The sample intends to help you understand some key concepts:

  - What AI workloads and deep learning models Intel has optimized and validated to run on Intel hardware.
  - How to train and deploy Intel-optimized models.
  - How to start running Intel-optimized models on Intel hardware in the cloud or on bare metal.

> **Disclaimer**: The sample and supplied scripts are not intended for benchmarking Intel platforms. For any performance and/or benchmarking information on specific Intel platforms, visit [https://www.intel.ai/blog](https://www.intel.ai/blog).

## Prerequisites

| Optimized for   | Description
|:---             |:---
| OS              | Ubuntu* 20.04 or higher
| Hardware        | Intel® Core™ Gen10 Processor <br> Intel® Xeon® Scalable Performance processors
| Software        | Intel® AI Reference models, Intel Extension for TensorFlow

### For Local Development Environments

Before running the sample, install the Intel Extension for TensorFlow* via the Intel AI Tools Selector or Offline Installer.   
You can refer to the Intel AI Tools [product page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html) for software installation and the *[Get Started with the Intel® AI Tools for Linux*](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit)* for post-installation steps and scripts.



## Key Implementation Details

The example uses some pretrained models published as part of the [Intel® AI Reference models](https://github.com/IntelAI/models). The example also illustrates how to utilize TensorFlow* runtime settings to maximize CPU performance on ResNet50 workload.


## Run the `Intel® TensorFlow* Model Zoo Inference With FP32 Int8` Sample

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

#### Activate Conda with Root Access

By default, the Intel AI Tools are installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it. However, if you activated another environment, you can return with the following command.
```
conda activate tensorflow
```

#### Activate Conda without Root Access (Optional)

You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

```
conda create --name user_tensorflow --clone tensorflow
conda activate user_tensorflow
```

#### Navigate to Model Zoo

Navigate to the Intel® AI Reference models source directory. By default, it is in your installation path, like `/opt/intel/oneapi/modelzoo`. 

1. View the available Intel® AI Reference models release versions for the AI Tools:
   ```
   ls /opt/intel/oneapi/reference_models
   2.13.0  latest
   ```
2. Navigate to the [Intel® AI Reference models Scripts](https://github.com/IntelAI/models/tree/v2.11.0/benchmarks) GitHub repo to determine the preferred released version to run inference for ResNet50 or another supported topology.
   ```
   cd /opt/intel/oneapi/reference_models/latest
   ```

#### Install Jupyter Notebook

```
conda install -c conda-forge jupyter nb_conda_kernels
```

#### Open Jupyter Notebook

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook
   ```
   > **Note**: If you do not have GUI support, you must open a remote shell and launch the Notebook a different way.
   > 1. Enter a command similar to the following:
   >   ```
   >   jupyter notebook --no-browser --port=8888`
   >   ```
   >2. Open the command prompt where you have GUI support, and forward the port from host to client.
   >3. Enter a command similar to the following:
   >   ```
   >   ssh -N -f -L localhost:8888:localhost:8888 <userid@hostname>
   >   ```
   >4. Copy and paste the URL address from the host into your local browser.

3. Locate and select the Notebook.
   ```
   ResNet50_Inference.ipynb
   ```
4. Change the kernel to **Python [conda env:tensorflow]**.
5. Click the **Run** button to move through the cells in sequence.


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt). 


*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
