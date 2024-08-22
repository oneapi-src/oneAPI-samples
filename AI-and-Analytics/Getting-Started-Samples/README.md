# Getting Started Samples for AI Tools

The AI Tools gives data scientists, AI developers, and researchers familiar Python* tools and frameworks to accelerate end-to-end data science and analytics pipelines on Intel® architectures. The components are built using oneAPI libraries for low-level compute optimizations. This toolkit maximizes performance from preprocessing through machine learning, and provides interoperability for efficient model development.

You can find more information at [ AI Tools](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

Users could learn how to run samples for different components in AI Tools with those getting started samples.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# Getting Started Samples

|AI Tools preset | Component      | Folder                                             | Description
|--------------------------| --------- | ------------------------------------------------ | -
|Inference Optimization| Intel® Neural Compressor (INC) | [Intel® Neural Compressor (INC) Sample-for-PyTorch](INC-Quantization-Sample-for-PyTorch)                     | Performs INT8 quantization on a Hugging Face BERT model.
|Inference Optimization| Intel® Neural Compressor (INC) | [Intel® Neural Compressor (INC) Sample-for-Tensorflow](INC-Sample-for-Tensorflow)                     | Quantizes a FP32 model into INT8 by Intel® Neural Compressor (INC) and compares the performance between FP32 and INT8.
|Data Analytics <br/> Classical Machine Learning  | Modin* | [Modin_GettingStarted](Modin_GettingStarted)                     | Run Modin*-accelerated Pandas functions and note the performance gain.
|Data Analytics <br/> Classical Machine Learning | Modin* |[Modin_Vs_Pandas](Modin_Vs_Pandas)| Compares the performance of Intel® Distribution of Modin* and the performance of Pandas.
|Classical Machine Learning| Intel® Optimization for XGBoost* | [IntelPython_XGBoost_GettingStarted](IntelPython_XGBoost_GettingStarted)                     | Set up and trains an XGBoost* model on datasets for prediction.
|Classical Machine Learning| daal4py | [IntelPython_daal4py_GettingStarted](IntelPython_daal4py_GettingStarted)                     | Batch linear regression using the Python API package daal4py from oneAPI Data Analytics Library (oneDAL).
|Deep Learning <br/> Inference Optimization| Intel® Optimization for TensorFlow* | [IntelTensorFlow_GettingStarted](IntelTensorFlow_GettingStarted)               | A simple training example for TensorFlow.
|Deep Learning <br/> Inference Optimization|Intel® Extension of PyTorch | [IntelPyTorch_GettingStarted]([https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/inference/python/jupyter-notebooks](https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/jupyter-notebooks/IPEX_Getting_Started.ipynb)| A simple training example for Intel® Extension of PyTorch.
|Classical Machine Learning| Scikit-learn (OneDAL) | [Intel_Extension_For_SKLearn_GettingStarted](Intel_Extension_For_SKLearn_GettingStarted) | Speed up a scikit-learn application using Intel oneDAL.
|Deep Learning <br/> Inference Optimization|Intel® Extension of TensorFlow | [Intel® Extension For TensorFlow GettingStarted](Intel_Extension_For_TensorFlow_GettingStarted)         | Guides users how to run a TensorFlow inference workload on both GPU and CPU.
|Deep Learning Inference Optimization|oneCCL Bindings for PyTorch | [Intel oneCCL Bindings For PyTorch GettingStarted](Intel_oneCCL_Bindings_For_PyTorch_GettingStarted)         | Guides users through the process of running a simple PyTorch* distributed workload on both GPU and CPU. |

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
