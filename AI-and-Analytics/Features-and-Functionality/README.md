# Features and Functionalities for Intel® oneAPI AI Analytics Toolkit

The Intel® oneAPI AI Analytics Toolkit gives data scientists, AI developers, and researchers familiar Python* tools and frameworks to accelerate end-to-end data science and analytics pipelines on Intel® architectures. The components are built using oneAPI libraries for low-level compute optimizations. This toolkit maximizes performance from preprocessing through machine learning, and provides interoperability for efficient model development.

You can find more information at [ AI Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

Users could understand more details of features in oneAPI AI Analytics Toolkits with those features and functionality samples.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# Features and Functionalities Samples

| Compoment      | Folder                                             | Description
| --------- | ------------------------------------------------ | -
| TensorFlow & Model Zoo | [IntelTensorFlow_ModelZoo_Inference_with_FP32_Int8](IntelTensorFlow_ModelZoo_Inference_with_FP32_Int8)               | Run ResNet50 inference on Intel's pretrained FP32 and Int8 model.
| TensorFlow & Model Zoo | [IntelTensorFlow_PerformanceAnalysis](IntelTensorFlow_PerformanceAnalysis) | Analyze the performance difference between Stock Tensorflow and Intel Tensorflow.
| daal4py | [IntelPython_daal4py_DistributedKMeans](IntelPython_daal4py_DistributedKMeans)   | Run a distributed K-Means model with oneDAL daal4py library memory objects.
| daal4py | [IntelPython_daal4py_DistributedLinearRegression](IntelPython_daal4py_DistributedLinearRegression)    | Run a distributed Linear Regression model with oneDAL daal4py library memory objects .
| PyTorch | [IntelPyTorch_Extensions_AutoMixedPrecision](IntelPyTorch_Extensions_AutoMixedPrecision)   | Download, compile, and get started with Intel Extension for PyTorch.
| PyTorch | [IntelPyTorch_TorchCCL_Multinode_Training](IntelPyTorch_TorchCCL_Multinode_Training)   | Perform distributed training with oneCCL in PyTorch.
| TensorFlow | [IntelTensorFlow_Horovod_Multinode_Training](IntelTensorFlow_Horovod_Multinode_Training)   | Get started with scaling out a neural network's training in TensorFlow by using Horovod.

# Using Samples in Intel oneAPI DevCloud

You can use AI Analytics Toolkit samples in
[Intel oneAPI DevCloud](https://devcloud.intel.com/oneapi/get-started/)
the environment in the following ways:
* Login to a DevCloud system via SSH and
  * use `git clone` to get a full copy of samples repository, or
  * use the `oneapi-cli` tool to download specific sample.
* Launch a JupyterLab server and run Jupyter Notebooks from your web browser.
