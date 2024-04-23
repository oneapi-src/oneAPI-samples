# Features and Functionalities for AI Tools

The Intel AI Tools give data scientists, AI developers, and researchers familiar Python* tools and frameworks to accelerate end-to-end data science and analytics pipelines on Intel® architectures. The components are built using oneAPI libraries for low-level compute optimizations. This toolkit maximizes performance from preprocessing through machine learning, and provides interoperability for efficient model development.

You can find more information at [ AI Tools](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

Users can explore the extensive features of Intel AI Tools through provided feature and functionality samples, offering a deeper understanding of their capabilities.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# Features and Functionalities Samples

|AI Tools preset bundle    | Component      | Folder                                             | Description
|--------------------------| --------- | ------------------------------------------------ | -
|Deep Learning| Intel® Neural Compressor (INC) | [Intel® Neural Compressor (INC) Quantization Aware Training](INC_QuantizationAwareTraining_TextClassification)                     | Fine-tune a BERT tiny model for emotion classification task using Quantization Aware Training and Inference from Intel® Neural Compressor (INC).
|Deep Learning| Intel® Extension for PyTorch (IPEX) | [IntelPyTorch Extensions Inference Optimization](IntelPyTorch_Extensions_Inference_Optimization)   | Apply Intel® Extension for PyTorch (IPEX) to a PyTorch workload to gain performance boost.
|Deep Learning| Intel® Extension for PyTorch (IPEX) | [IntelPyTorch Extensions GPU Inference Optimization with AMP](IntelPyTorch_GPU_InferenceOptimization_with_AMP)   | Use the PyTorch ResNet50 model transfer learning and inference using the CIFAR10 dataset on Intel discrete GPU with Intel® Extension for PyTorch (IPEX).
|Deep Learning| Intel® Extension for PyTorch (IPEX)| [IntelPyTorch_InferenceOptimizations_AMX_BF16_INT8](IntelPyTorch_InferenceOptimizations_AMX_BF16_INT8)   | Analyze inference performance improvements using Intel® Extension for PyTorch (IPEX) with Advanced Matrix Extensions (Intel® AMX) Bfloat16 and Integer8.
|Deep Learning| PyTorch | [IntelPyTorch TrainingOptimizations Intel® AMX BF16](IntelPyTorch_TrainingOptimizations_AMX_BF16)   | Analyze training performance improvements using Intel® Extension for PyTorch (IPEX) with Intel® AMX Bfloat16.
|Data Analytics | Numpy, Numba | [IntelPython Numpy Numba dpex kNN](IntelPython_Numpy_Numba_dpex_kNN)   | Optimize k-NN model by numba_dpex operations without sacrificing accuracy.
|Classical Machine Learning| XGBoost | [IntelPython XGBoost Performance](IntelPython_XGBoost_Performance) |  Analyze the performance benefit from using Intel optimized XGBoost compared to un-optimized XGBoost 0.81.
|Classical Machine Learning| XGBoost | [IntelPython XGBoost daal4pyPrediction](IntelPython_XGBoost_daal4pyPrediction) |  Analyze the performance benefit of minimal code changes to port pre-trained XGBoost model to daal4py prediction for much faster prediction than XGBoost prediction.
|Classical Machine Learning| daal4py | [IntelPython daal4py DistributedKMeans](IntelPython_daal4py_DistributedKMeans)    | Train and predict with a distributed k-means model using the python API package daal4py powered by the oneAPI Data Analytics Library.
|Classical Machine Learning| daal4py | [IntelPython daal4py DistributedLinearRegression](IntelPython_daal4py_DistributedLinearRegression)    | Run a distributed Linear Regression model with oneAPI Data Analytics Library (oneDAL) daal4py library memory objects.
|Deep Learning| PyTorch | [IntelPytorch Interactive Chat Quantization](IntelPytorch_Interactive_Chat_Quantization)   | Create interactive chat based on pre-trained DialoGPT model and add the Intel® Extension for PyTorch (IPEX) quantization to it.
|Deep Learning| PyTorch | [IntelPytorch Quantization](IntelPytorch_Quantization)   | Inference performance improvements using Intel® Extension for PyTorch (IPEX) with feature quantization.
|Deep Learning| TensorFlow | [IntelTensorFlow Intel® AMX BF16 Training](IntelTensorFlow_AMX_BF16_Inference) | Enabling auto-mixed precision to use low-precision datatypes, like bfloat16, for model inference with TensorFlow* .
|Deep Learning| TensorFlow | [IntelTensorFlow Intel® AMX BF16 Training](IntelTensorFlow_AMX_BF16_Training) | Training performance improvements with Intel® AMX BF16.
|Deep Learning| TensorFlow | [IntelTensorFlow Enabling Auto Mixed Precision for TransferLearning](IntelTensorFlow_Enabling_Auto_Mixed_Precision_for_TransferLearning) | Enabling auto-mixed precision to use low-precision datatypes, like bfloat16, for transfer learning with TensorFlow*.
|Deep Learning| Horovod | [IntelTensorFlow Horovod Distributed Deep Learning](IntelTensorFlow_Horovod_Distributed_Deep_Learning) | Run inference & training workloads across multi-cards using Intel Optimization for Horovod and TensorFlow* on Intel® dGPU's.
|Deep Learning| TensorFlow | [IntelTensorFlow InferenceOptimization](IntelTensorFlow_InferenceOptimization) |  Optimize a pre-trained model for a better inference performance.
|Deep Learning| TensorFlow & Intel® AI Reference Models | [IntelTensorFlow Reference Models Inference with FP32 Int8](IntelTensorFlow_ModelZoo_Inference_with_FP32_Int8)               | Run ResNet50 inference on Intel's pretrained FP32 and NT8 model.
|Deep Learning| TensorFlow | [IntelTensorFlow PerformanceAnalysis](IntelTensorFlow_PerformanceAnalysis) | Analyze the performance difference between Stock Tensorflow and Intel Tensorflow.
|Deep Learning| TensorFlow | [IntelTensorFlow Transformer Intel® AMX bfloat16 MixedPrecisiong](IntelTensorFlow_Transformer_AMX_bfloat16_MixedPrecision) | Run a transformer classification model with bfloat16 mixed precision.
|Deep Learning| TensorFlow | [IntelTensorFlow for LLMs](IntelTensorFlow_for_LLMs) | Finetune a GPT-J (LLM) model using the GLUE cola dataset with the Intel® Optimization for TensorFlow*.
|Classical Machine Learning| Scikit-learn | [IntelScikitLearn Extensions SVC Adult](IntelScikitLearn_Extensions_SVC_Adult)   | Use Intel® Extension for Scikit-learn to accelerate the training and prediction with SVC algorithm on Adult dataset. Compare the performance of SVC algorithm optimized through Intel® Extension for Scikit-learn against original Scikit-learn.

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)


