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
| PyTorch | [IntelPyTorch Extensions Inference Optimization](IntelPyTorch_Extensions_Inference_Optimization)   | Applying IPEX Optimizations to a PyTorch workload to gain performance boost.
| PyTorch | [IntelPyTorch TrainingOptimizations AMX BF16](IntelPyTorch_TrainingOptimizations_AMX_BF16)   | Analyze training performance improvements using Intel® Extension for PyTorch with Advanced Matrix Extensions Bfloat16.
| PyTorch | [IntelPyTorch_InferenceOptimizations_AMX_BF16_INT8](IntelPyTorch_InferenceOptimizations_AMX_BF16_INT8)   | Analyze inference performance improvements using Intel® Extension for PyTorch with Advanced Matrix Extensions Bfloat16 and Integer8.
| Numpy, Numba | [IntelPython Numpy Numba dpex kNN](IntelPython_Numpy_Numba_dpex_kNN)   | Optimize k-NN model by numba_dpex operations without sacrificing accuracy.
| XGBoost | [IntelPython XGBoost Performance](IntelPython_XGBoost_Performance) |  Analyze the performance benefit from using Intel optimized XGBoost compared to un-optimized XGBoost 0.81.
| XGBoost | [IntelPython XGBoost daal4pyPrediction](IntelPython_XGBoost_daal4pyPrediction) |  Analyze the performance benefit of minimal code changes to port pre-trained XGBoost model to daal4py prediction for much faster prediction than XGBoost prediction.
| daal4py | [IntelPython daal4py DistributedKMeans](IntelPython_daal4py_DistributedKMeans)    | train and predict with a distributed k-means model using the python API package daal4py powered by the oneAPI Data Analytics Library.
| daal4py | [IntelPython daal4py DistributedLinearRegression](IntelPython_daal4py_DistributedLinearRegression)    | Run a distributed Linear Regression model with oneAPI Data Analytics Library (oneDAL) daal4py library memory objects.
| PyTorch | [IntelPytorch Quantization](IntelPytorch_Quantization)   | Inference performance improvements using Intel® Extension for PyTorch* (IPEX) with feature quantization.
| TensorFlow | [IntelTensorFlow AMX BF16 Training](IntelTensorFlow_AMX_BF16_Training) | Training performance improvements with Intel® AMX BF16.
| TensorFlow | [IntelTensorFlow Enabling Auto Mixed Precision for TransferLearning](IntelTensorFlow_Enabling_Auto_Mixed_Precision_for_TransferLearning) | Enabling auto-mixed precision to use low-precision datatypes, like bfloat16, for transfer learning with TensorFlow*.
| TensorFlow | [IntelTensorFlow InferenceOptimization](IntelTensorFlow_InferenceOptimization) |  Optimize a pre-trained model for a better inference performance.
| TensorFlow & Model Zoo | [IntelTensorFlow ModelZoo Inference with FP32 Int8](IntelTensorFlow_ModelZoo_Inference_with_FP32_Int8)               | Run ResNet50 inference on Intel's pretrained FP32 and Int8 model.
| TensorFlow & Model Zoo | [IntelTensorFlow PerformanceAnalysis](IntelTensorFlow_PerformanceAnalysis) | Analyze the performance difference between Stock Tensorflow and Intel Tensorflow.
| TensorFlow | [IntelTensorFlow Transformer AMX bfloat16 MixedPrecisiong](IntelTensorFlow_Transformer_AMX_bfloat16_MixedPrecision) | Run a transformer classification model with bfloat16 mixed precision.
| Scikit-learn | [IntelScikitLearn Extensions SVC Adult](IntelScikitLearn_Extensions_SVC_Adult)   | Use Intel® Extension for Scikit-learn to accelerate the training and prediction with SVC algorithm on Adult dataset. Compare the performance of SVC algorithm optimized through Intel® Extension for Scikit-learn against original Scikit-learn..

# Using Samples in Intel® DevCloud for oneAPI
To get started using samples in the DevCloud, refer to [Using AI samples in Intel® DevCloud for oneAPI](https://github.com/intel-ai-tce/oneAPI-samples/tree/devcloud/AI-and-Analytics#using-samples-in-intel-oneapi-devcloud).
