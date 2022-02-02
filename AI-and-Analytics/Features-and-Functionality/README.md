# Features and Functionalities for Intel® oneAPI AI Analytics Toolkit (AI Kit)

The Intel® oneAPI AI Analytics Toolkit (AI Kit) gives data scientists, AI developers, and researchers familiar Python* tools and frameworks to accelerate end-to-end data science and analytics pipelines on Intel® architectures. The components are built using oneAPI libraries for low-level compute optimizations. This toolkit maximizes performance from preprocessing through machine learning, and provides interoperability for efficient model development.

You can find more information at [ AI Kit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

Users could learn more details of features in oneAPI AI Kit with those features and functionality samples.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# Features and Functionalities Samples

| Compoment      | Folder                                             | Description
| --------- | ------------------------------------------------ | -
| Scikit-learn | [IntelScikitLearn_Extensions_SVC_Adult](IntelScikitLearn_Extensions_SVC_Adult)   | Use Intel® Extension for Scikit-learn to accelerate the training and prediction with SVC algorithm on Adult dataset. Compare the performance of SVC algorithm optimized through Intel® Extension for Scikit-learn against original Scikit-learn.
| daal4py | [IntelPython_daal4py_DistributedLinearRegression](IntelPython_daal4py_DistributedLinearRegression)    | Run a distributed Linear Regression model with oneAPI Data Analytics Library (oneDAL) daal4py library memory objects.
| PyTorch | [IntelPyTorch_Extensions_AutoMixedPrecision](IntelPyTorch_Extensions_AutoMixedPrecision)   | Download, compile, and get started with Intel® Extension for PyTorch*.
| PyTorch | [IntelPyTorch_TorchCCL_Multinode_Training](IntelPyTorch_TorchCCL_Multinode_Training)   | Perform distributed training with oneAPI Collective Communications Library (oneCCL) in PyTorch.
| TensorFlow & Model Zoo | [IntelTensorFlow_ModelZoo_Inference_with_FP32_Int8](IntelTensorFlow_ModelZoo_Inference_with_FP32_Int8)               | Run ResNet50 inference on Intel's pretrained FP32 and Int8 model.
| TensorFlow & Model Zoo | [IntelTensorFlow_PerformanceAnalysis](IntelTensorFlow_PerformanceAnalysis) | Analyze the performance difference between Stock Tensorflow and Intel Tensorflow.
| TensorFlow | [IntelTensorFlow_InferenceOptimization](IntelTensorFlow_InferenceOptimization) |  Optimize a pre-trained model for a better inference performance.
| TensorFlow | [IntelTensorFlow_Horovod_Multinode_Training](IntelTensorFlow_Horovod_Multinode_Training)   | Get started with scaling out a neural network's training in TensorFlow by using Horovod.
| XGBoost | [IntelPython_XGBoost_Performance](IntelPython_XGBoost_Performance) |  Analyze the performance benefit from using Intel optimized XGBoost compared to un-optimized XGBoost 0.81.
| XGBoost | [IntelPython_XGBoost_daal4pyPrediction](IntelPython_XGBoost_daal4pyPrediction) |  Analyze the performance benefit of minimal code changes to port pre-trained XGBoost model to daal4py prediction for much faster prediction than XGBoost prediction..

# Using Samples in Intel® DevCloud for oneAPI
To get started using samples in the DevCloud, refer to [Using AI samples in Intel® DevCloud for oneAPI](https://github.com/intel-ai-tce/oneAPI-samples/tree/devcloud/AI-and-Analytics#using-samples-in-intel-oneapi-devcloud).

