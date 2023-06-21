# `Intel® TensorFlow* Model Zoo Inference With FP32 Int8` Sample

The `Intel® TensorFlow* Model Zoo Inference With FP32 Int8` sample demonstrates how to run ResNet50 inference on pretrained FP32 and Int8 models included in the Model Zoo for Intel® Architecture.

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
| Software        | Intel® AI Analytics Toolkit (AI Kit)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

  TensorFlow* or Pytorch* are ready for use once you finish installing and configuring the Intel® AI Analytics Toolkit (AI Kit).

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See [Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get_started/) for information.

## Key Implementation Details

The example uses some pretrained models published as part of the [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models). The example also illustrates how to utilize TensorFlow* and Intel® Math Kernel Library (Intel® MKL) runtime settings to maximize CPU performance on ResNet50 workload.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `Intel® TensorFlow* Model Zoo Inference With FP32 Int8` Sample

### On Linux*

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

By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it. However, if you activated another environment, you can return with the following command.
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

Navigate to the Model Zoo for Intel® Architecture source directory. By default, it is in your installation path, like `/opt/intel/oneapi/modelzoo`. 

1. View the available Model Zoo release versions for the AI Kit:
   ```
   ls /opt/intel/oneapi/modelzoo
   2.11.0  latest
   ```
2. Navigate to the [Model Zoo Scripts](https://github.com/IntelAI/models/tree/v2.11.0/benchmarks) GitHub repo to determine the preferred released version to run inference for ResNet50 or another supported topology.
   ```
   cd /opt/intel/oneapi/modelzoo/latest
   ```

#### Install Jupyter Notebook

```
conda install jupyter nb_conda_kernels
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

### Run the Sample on Intel® DevCloud (Optional)

1. If you do not already have an account, request an Intel® DevCloud account at [*Create an Intel® DevCloud Account*](https://intelsoftwaresites.secure.force.com/DevCloud/oneapi).
2. On a Linux* system, open a terminal.
3. SSH into Intel® DevCloud.
   ```
   ssh DevCloud
   ```
   > **Note**: You can find information about configuring your Linux system and connecting to Intel DevCloud at Intel® DevCloud for oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started).

4. You can specify a CPU node using a single line script.
   ```
   qsub  -I  -l nodes=1:xeon:ppn=2 -d .
   ```

   - `-I` (upper case I) requests an interactive session.
   - `-l nodes=1:xeon:ppn=2` (lower case L) assigns one full GPU node.
   - `-d .` makes the current folder as the working directory for the task.

     |Available Nodes |Command Options
     |:---            |:---
     |GPU	            |`qsub -l nodes=1:gpu:ppn=2 -d .`
     |CPU	            |`qsub -l nodes=1:xeon:ppn=2 -d .`

5. Activate conda. 
` $ conda activate`
6. Follow the instructions to open the URL with the token in your browser.
7. Locate and select the Notebook.
   ```
   ResNet50_Inference.ipynb
   ````
8. Change the kernel to **Python [conda env:tensorflow]**.
9. Run every cell in the Notebook in sequence.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).