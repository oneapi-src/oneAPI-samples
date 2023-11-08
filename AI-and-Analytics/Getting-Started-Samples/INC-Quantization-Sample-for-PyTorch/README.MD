# `Getting Started with Intel® Neural Compressor for Quantization` Sample

The sample is a getting started tutorial for the Intel® Neural Compressor (INC), and demonstrates how to perform INT8 quantization on a Hugging Face BERT model. This sample shows how to achieve performance boosts using Intel hardware.

| Area                  | Description
|:---                   |:---
| What you will learn   | How to quantize a BERT model using Intel® Neural Compressor
| Time to complete      | 20 minutes
| Category              | Code Optimization

## Purpose

Intel® Neural Compressor comes with many options for deep learning model compression, one of them being INT8 Quantization. Quantization help to reduce the size of the model, which enables faster inference. The approach requires a trade-off in reduced accuracy for the reduced size; however, Intel® Neural Compressor provides automated accuracy-driven tuning recipes that will allow you to quantize your model and maintain your model accuracy goals.

The sample starts by loading a BERT model from Hugging Face. After loading the model, we set up an evaluation function that we care about using PyTorch* Dataset and DataLoader classes. Using this evaluation function, Intel® Neural Compressor can perform both post training static and dynamic quantization to achieve the speedups.

## Prerequisites

| Optimized for           | Description
|:---                     |:---
| OS                      | Ubuntu* 20.04 (or newer)
| Hardware                | Intel® Xeon® Scalable processor family
| Software                | Intel® Neural Compressor, Intel Optimization for PyTorch

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Tools**

  You can get the Intel AI Tools from [the product page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html). <br> See [*Get Started with the Intel® AI Tools for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI KitTools installation information and post-installation steps and scripts.

- **Jupyter Notebook**

  Install using PIP: `$pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.


## Key Implementation Details

The sample contains one Jupyter Notebook and one Python script.

### Jupyter Notebook

|Notebook                  |Description
|:---                      |:---
|`quantize_with_inc.ipynb` | Get started tutorial for using Intel® Neural Compressor for PyTorch*

### Python Script

|Script                    |Description
|:---                      |:---
|`dataset.py`              | The script provides a PyTorch* Dataset class that tokenizes text data 


## Run the `Getting Started with Intel® Neural Compressor for Quantization` Sample
```If you have already set up the PIP or Conda environment and installed AI Tools go directly to Run the Notebook.

### Steps for Intel AI Tools Offline Installer
> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

#### Activate Conda

1. Activate the Conda environment.

    ```
    conda activate pytorch
    ```

   By default, the Intel AI Tools are installed in the `/opt/intel/oneapi` folder and require root privileges to manage them.

   You can choose to activate Conda environment without root access. To bypass root access to manage your Conda environment, clone and activate your desired Conda environment using the following commands similar to the following.

#### Run the NoteBook

1. Launch Jupyter Notebook.
   ```
   jupyter notebook --ip=0.0.0.0
   ```
2. Follow the instructions to open the URL with the token in your browser.
3. Locate and select the Notebook.
   ```
   optimize_pytorch_models_with_ipex.ipynb
   ```
4. Change the kernel to **pytorch**.
5. Run every cell in the Notebook in sequence.

#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.   

## Example Output
You should see an image showing the performance comparison and analysis between FP32 and INT8.
>**Note**: The image shown below is an example of a general performance comparison for inference speedup obtained by quantization. (Your results might be different.)

![Performance Numbers](images/inc_speedup.png)


## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
