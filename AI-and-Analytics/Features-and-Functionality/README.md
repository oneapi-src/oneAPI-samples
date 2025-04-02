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
|Deep Learning| Intel® Extension for PyTorch (IPEX) | [Optimize PyTorch Models using Intel® Extension for PyTorch* (IPEX)](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/jupyter-notebooks/README.md)  | Apply Intel® Extension for PyTorch (IPEX) to a PyTorch workload to gain performance boost.
|Deep Learning| Intel® Extension for PyTorch (IPEX) | [IntelPyTorch Extensions GPU Inference Optimization with AMP](IntelPyTorch_GPU_InferenceOptimization_with_AMP)   | Use the PyTorch ResNet50 model transfer learning and inference using the CIFAR10 dataset on Intel discrete GPU with Intel® Extension for PyTorch (IPEX).
|Deep Learning| Intel® Extension for PyTorch (IPEX)| [PyTorch Inference Optimizations with Intel® Advanced Matrix Extensions (Intel® AMX) Bfloat16 Integer8](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/jupyter-notebooks/README.md)   | Analyze inference performance improvements using Intel® Extension for PyTorch (IPEX) with Advanced Matrix Extensions (Intel® AMX) Bfloat16 and Integer8.
|Deep Learning| PyTorch | [IntelPyTorch TrainingOptimizations Intel® AMX BF16](IntelPyTorch_TrainingOptimizations_AMX_BF16)   | Analyze training performance improvements using Intel® Extension for PyTorch (IPEX) with Intel® AMX Bfloat16.
|Deep Learning| PyTorch | [Interactive Chat Based on DialoGPT Model Using Intel® Extension for PyTorch* Quantization](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/jupyter-notebooks/README.md)   | Create interactive chat based on pre-trained DialoGPT model and add the Intel® Extension for PyTorch (IPEX) quantization to it.
|Deep Learning| PyTorch | [Optimize PyTorch Models using Intel® Extension for PyTorch* (IPEX) Quantization](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/jupyter-notebooks/README.md)   | Inference performance improvements using Intel® Extension for PyTorch (IPEX) with feature quantization.
|Deep Learning| TensorFlow | [IntelTensorFlow Intel® AMX BF16 Training](IntelTensorFlow_AMX_BF16_Inference) | Enabling auto-mixed precision to use low-precision datatypes, like bfloat16, for model inference with TensorFlow* .
|Deep Learning| TensorFlow | [IntelTensorFlow Intel® AMX BF16 Training](IntelTensorFlow_AMX_BF16_Training) | Training performance improvements with Intel® AMX BF16.
|Deep Learning| TensorFlow | [IntelTensorFlow Enabling Auto Mixed Precision for TransferLearning](IntelTensorFlow_Enabling_Auto_Mixed_Precision_for_TransferLearning) | Enabling auto-mixed precision to use low-precision datatypes, like bfloat16, for transfer learning with TensorFlow*.
|Deep Learning| Horovod | [IntelTensorFlow Horovod Distributed Deep Learning](IntelTensorFlow_Horovod_Distributed_Deep_Learning) | Run inference & training workloads across multi-cards using Intel Optimization for Horovod and TensorFlow* on Intel® dGPU's.
|Deep Learning| TensorFlow | [IntelTensorFlow InferenceOptimization](IntelTensorFlow_InferenceOptimization) |  Optimize a pre-trained model for a better inference performance.
|Deep Learning| TensorFlow & Intel® AI Reference Models | [IntelTensorFlow Reference Models Inference with FP32 Int8](IntelTensorFlow_ModelZoo_Inference_with_FP32_Int8)               | Run ResNet50 inference on Intel's pretrained FP32 and NT8 model.
|Deep Learning| TensorFlow | [IntelTensorFlow PerformanceAnalysis](IntelTensorFlow_PerformanceAnalysis) | Analyze the performance difference between Stock Tensorflow and Intel Tensorflow.
|Deep Learning| TensorFlow | [IntelTensorFlow Transformer Intel® AMX bfloat16 MixedPrecisiong](IntelTensorFlow_Transformer_AMX_bfloat16_MixedPrecision) | Run a transformer classification model with bfloat16 mixed precision.
|Deep Learning| TensorFlow | [IntelTensorFlow for LLMs](IntelTensorFlow_for_LLMs) | Finetune a GPT-J (LLM) model using the GLUE cola dataset with the Intel® Optimization for TensorFlow*.

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)