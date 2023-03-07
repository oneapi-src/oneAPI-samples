# `TensorFlow* Training Optimizations with Advanced Matrix Extensions Bfloat16` Sample

The `TensorFlow Training Optimizations with Advanced Matrix Extensions Bfloat16` sample demonstrates how to train a DistilBERT model using the Disaster Tweet Classification dataset with the Intel® Optimization for TensorFlow*.

The Intel® Optimization for TensorFlow* enables TensorFlow* with optimizations for performance boost on Intel® hardware. For example, newer optimizations include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

| Area                  | Description
|:---                   |:---
| What you will learn   | Training performance improvements with Intel® AMX BF16
| Time to complete      | 20 minutes
| Category              | Code Optimization

## Purpose

The Intel® Optimization for TensorFlow* gives users the ability to speed up training on Intel® Xeon Scalable processors with lower precision data formats and specialized computer instructions. The bfloat16 (BF16) data format uses half the bit width of floating-point-32 (FP32), lowering the amount of memory needed and execution time to process. You should notice performance optimization with the AMX instruction set when compared to AVX-512.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 18.04 or newer
| Hardware                | Intel® Xeon® Scalable Processor family
| Software                | Intel® Optimization for TensorFlow*

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

- **Jupyter Notebook**

  Install using PIP: `$pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.

- **Additional Packages**

  You will need to install the additional packages in requirements.txt and **Py-cpuinfo**.
  ```
  pip install -r requirements.txt
  python -m pip install py-cpuinfo
  ```

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See *[Intel® DevCloud for oneAPI](https://DevCloud.intel.com/oneapi/get_started/)* for information.

## Key Implementation Details

This code sample trains a DistilBERT model using the Disaster Tweet Classification dataset while using Intel® Optimization for TensorFlow*. The model is trained using FP32 and BF16 precision, including the use of Intel® Advanced Matrix Extensions (AMX) on BF16. AMX is supported on BF16 and INT8 data types starting with 4th Gen Xeon Scalable Processors. The training time will be compared, showcasing the speedup of BF16 and AMX.

>**Note**: Training is not performed using INT8 since using a lower precision will train a model with fewer parameters, which is likely to underfit and not generalize well.

The sample tutorial contains one Jupyter Notebook and a Python script. You can use either.

### Jupyter Notebook

| Notebook                                 | Description
|:---                                      |:---
|`IntelTensorFlow_AMX_BF16_Training.ipynb` | TensorFlow Training Optimizations with Advanced Matrix Extensions Bfloat16

### Python Scripts

| Script                                 | Description
|:---                                    |:---
|`Intel_TensorFlow_AMX_BF16_Training.py` | The script performs training with AMX BF16 and compares the performance against the baseline

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the `TnesorFlow Training Optimizations with Advanced Matrix Extensions Bfloat16` Sample

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

#### Activate Conda

1. Activate the Conda environment.
    ```
    conda activate tensorflow
    ```
2. Activate Conda environment without Root access (Optional).

   By default, the AI Kit is installed in the `/opt/intel/oneapi` folder and requires root privileges to manage it.

   You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

   ```
   conda create --name user_tensorflow --clone tensorflow
   conda activate user_tensorflow
   ```

#### Running the Jupyter Notebook

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0 --port 8888 --allow-root
   ```
3. Follow the instructions to open the URL with the token in your browser.
4. Locate and select the Notebook.
   ```
   IntelTensorFlow_AMX_BF16_Training.ipynb
   ```
5. Change your Jupyter Notebook kernel to corresponding environment.
6. Run every cell in the Notebook in sequence.

#### Running on the Command Line (Optional)

1. Change to the sample directory.
2. Run the script.
   ```
   python Intel_TensorFlow_AMX_BF16_Training.py
   ```


### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.

## Example Output

If successful, the sample displays `[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]`. Additionally, the sample generates performance and analysis diagrams for comparison.

The following image shows approximate performance speed increases using AMX BF16 with auto-mixed precision during training. To see more performance improvement between AVX-512 BF16 and AMX BF16, increase the number of required computations in one batch.

![comparison images](assets/amx_relative_speedup.png)

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)
