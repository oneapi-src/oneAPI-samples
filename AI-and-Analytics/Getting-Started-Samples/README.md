# Getting Started Samples for Intel® oneAPI AI Analytics Toolkit

The Intel® oneAPI AI Analytics Toolkit gives data scientists, AI developers, and researchers familiar Python* tools and frameworks to accelerate end-to-end data science and analytics pipelines on Intel® architectures. The components are built using oneAPI libraries for low-level compute optimizations. This toolkit maximizes performance from preprocessing through machine learning, and provides interoperability for efficient model development.

You can find more information at [ AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

Users could understand how to run samples for different components in oneAPI AI Analytics Toolkits with those getting started samples.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# Getting Started Samples

| Compoment      | Folder                                             | Description
| --------- | ------------------------------------------------ | -
| daal4py | [IntelPython_daal4py_GettingStarted](IntelPython_daal4py_GettingStarted)                     | Batch linear regression using the python API package daal4py from oneDAL .
| LPOT | [LPOT-Sample-for-Tensorflow](LPOT-Sample-for-Tensorflow)                     |Quantize a fp32 model into int8, and compare the performance between fp32 and int8 .
| Modin | [IntelModin_GettingStarted](IntelModin_GettingStarted)                     | Run Modin-accelerated Pandas functions and note the performance gain .
| PyTorch | [IntelPyTorch_GettingStarted](IntelPyTorch_GettingStarted) | A simple training example for PyTorch.
| TensorFlow | [IntelTensorFlow_GettingStarted](IntelTensorFlow_GettingStarted)               | A simple training example for TensorFlow.
| XGBoost | [IntelPython_XGBoost_GettingStarted](IntelPython_XGBoost_GettingStarted)                     | Set up and train an XGBoost* model on datasets for prediction.


# Using Samples in Intel oneAPI DevCloud

You can use AI Analytics Toolkit samples in
[Intel oneAPI DevCloud](https://devcloud.intel.com/oneapi/get-started/)
the environment in the following ways:
* Login to a DevCloud system via SSH and
  * use `git clone` to get a full copy of samples repository, or
  * use the `oneapi-cli` tool to download specific sample.
* Launch a JupyterLab server and run Jupyter Notebooks from your web browser.
