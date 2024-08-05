# `Intel® Extension for Scikit-learn*: SVC for Adult Data Set` Sample

The `Intel® Extension for Scikit-learn*: SVC for Adult Data Set` sample uses the [Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) to show how to train and predict with an SVC algorithm using Intel® Extension for Scikit-learn*.

| Optimized for          | Description
| :---                   | :---
| What you will learn    | How to get started with Intel® Extension for Scikit-learn*
| Time to complete       | 25 minutes
| Category               | Concepts and Functionality

The sample demonstrates how to use software products that can be found in the [Intel® oneAPI Data Analytics Library (oneDAL)](https://github.com/oneapi-src/oneDAL), [Intel® Extension for Scikit-learn*](https://intel.github.io/scikit-learn-intelex/), and the [Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

## Purpose

In this sample, you will run an SVC algorithm with Intel® Extension for Scikit-learn* and compare its performance against the original stock version of scikit-learn. You will see that patching scikit-learn results in a significant increase in performance over the original scikit-learn while also maintaining the same precision.

The acceleration is achieved through the use of the oneDAL. Patching scikit-learn makes it a well-suited machine learning framework for dealing with real-life problems.

## Prerequisites

| Optimized for          | Description
| :---                   | :---
| OS                     | Ubuntu 20.04 (or newer)
| Hardware               | Intel Atom® Processors <br> Intel® Core™ Processor Family <br> Intel® Xeon® Processor Family <br> Intel® Xeon® Scalable processor family
| Software               | Intel® AI Analytics Toolkit (AI Kit)

### For Local Development Environments

You will need to download and install the following toolkits, tools, and components to use the sample.

- **Intel® AI Analytics Toolkit (AI Kit)**

  You can get the AI Kit from [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit). <br> See [*Get Started with the Intel® AI Analytics Toolkit for Linux**](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-ai-linux) for AI Kit installation information and post-installation steps and scripts.

- **Jupyter Notebook**

  Install using PIP: `$pip install notebook`. <br> Alternatively, see [*Installing Jupyter*](https://jupyter.org/install) for detailed installation instructions.

### For Intel® DevCloud

The necessary tools and components are already installed in the environment. You do not need to install additional components. See [Intel® DevCloud for oneAPI](https://devcloud.intel.com/oneapi/get_started/) for information.

## Key Implementation Details

The sample code is written in Python and it targets CPU architecture. The example assumes you have Intel® Extension for Scikit-learn* installed.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Run the Sample

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

### Open Jupyter Notebook

1. Change to the sample directory.
2. Launch Jupyter Notebook.
   ```
   jupyter notebook
   ```
3. Locate and select the Notebook.
   ```
   Intel_Extension_for_SKLearn_Performance_SVC_Adult.ipynb
   ```
4. Click the **Run** button to move through the cells in sequence.

### Run the Python File

1. Run the script.
   ```
   python Intel_Extension_for_SKLearn_Performance_SVC_Adult.py
   ```
#### Troubleshooting

If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the *[Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)* for more information on using the utility.


### On Intel® DevCloud (Optional)

>**Note**: For more information on using Intel® DevCloud, see the Intel® oneAPI [Get Started](https://devcloud.intel.com/oneapi/get_started/) page.

1. Open a terminal on a Linux* system.
2. Log in to the Intel® DevCloud.
   ```
   ssh devcloud
   ```
3. Change to the sample directory.
4. Perform steps as you would on Linux.
5. Run the sample.
6. Review the output.
7. Disconnect from Intel® DevCloud.
   ```
   exit
   ```
## Example Output

```
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
Execution time with Intel(R) extension for Scikit-learn: 13.88 s
Classification report for SVC trained with Intel(R) extension for Scikit-learn:
              precision    recall  f1-score   support

        -1.0       0.87      0.90      0.88      7414
         1.0       0.64      0.58      0.61      2355

    accuracy                           0.82      9769
   macro avg       0.76      0.74      0.75      9769
weighted avg       0.82      0.82      0.82      9769


Execution time with the original Scikit-learn: 1019.58 s
Classification report for SVC trained with the original scikit-learn:
              precision    recall  f1-score   support

        -1.0       0.87      0.90      0.88      7414
         1.0       0.64      0.58      0.61      2355

    accuracy                           0.82      9769
   macro avg       0.76      0.74      0.75      9769
weighted avg       0.82      0.82      0.82      9769
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).