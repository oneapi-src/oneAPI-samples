# Getting Started Samples for AI Tools

The AI Tools gives data scientists, AI developers, and researchers familiar Python* tools and frameworks to accelerate end-to-end data science and analytics pipelines on Intel® architectures. The components are built using oneAPI libraries for low-level compute optimizations. This toolkit maximizes performance from preprocessing through machine learning, and provides interoperability for efficient model development.

You can find more information at [ AI Tools](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

Users could learn how to run samples for different components in AI Tools with those getting started samples.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# Getting Started Samples

<table>
  <tr>
    <th>AI Tools preset</th>
    <th>Component</th>
    <th>Folder</th>
    <th>Description</th>
  </tr>
  <tr>
    <td rowspan="5">Classical Machine Learning</td>
    <td rowspan="2">Modin*</td>
    <td><a href="Modin_GettingStarted">Modin_GettingStarted</a></td>
    <td>Run Modin*-accelerated Pandas functions and note the performance gain.</td>
  </tr>
  <tr>
    <td><a href="Modin_Vs_Pandas">Modin_Vs_Pandas</a></td>
    <td>Compares the performance of Intel® Distribution of Modin* and the performance of Pandas.</td>
  </tr>
  <tr>
    <td>Intel® Optimization for XGBoost*</td>
    <td><a href="IntelPython_XGBoost_GettingStarted">IntelPython_XGBoost_GettingStarted</a></td>
    <td>Set up and trains an XGBoost* model on datasets for prediction.</td>
  </tr>
  <tr>
    <td rowspan="2">Scikit-learn*</td>
    <td><a href="Intel_Extension_For_SKLearn_GettingStarted">Intel_Extension_For_SKLearn_GettingStarted</a></td>
    <td>Speed up a scikit-learn application using Intel oneDAL.</td>
  </tr>
  <tr>
    <td><a href="IntelPython_daal4py_GettingStarted">IntelPython_daal4py_GettingStarted</a></td>
    <td>Batch linear regression using the Python API package daal4py from oneAPI Data Analytics Library (oneDAL).</td>
  </tr>
  <tr>
    <td rowspan="8">Deep Learning</td>
    <td rowspan="2">Intel® Extension of PyTorch</td>
    <td><a href="https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/jupyter-notebooks/README.md">Getting Started with Intel® Extension for PyTorch* (IPEX)</a></td>
    <td>A simple training example for Intel® Extension of PyTorch.</td>
  </tr>
  <tr>
    <td><a href="Intel_oneCCL_Bindings_For_PyTorch_GettingStarted">Intel_oneCCL_Bindings_For_PyTorch_GettingStarted</a></td>
    <td>Guides users through the process of running a simple PyTorch* distributed workload on both GPU and CPU.</td>
  </tr>
  <tr>
    <td rowspan="2">Intel® Neural Compressor (INC)</td>
    <td><a href="Intel® Neural Compressor (INC) Sample-for-PyTorch">Intel® Neural Compressor (INC) Sample-for-PyTorch</a></td>
    <td>Performs INT8 quantization on a Hugging Face BERT model.</td>
  </tr>
  <tr>
    <td><a href="Intel® Neural Compressor (INC) Sample-for-Tensorflow">Intel® Neural Compressor (INC) Sample-for-Tensorflow</a></td>
    <td>Quantizes a FP32 model into INT8 by Intel® Neural Compressor (INC) and compares the performance between FP32 and INT8.</td>
  </tr>
  <tr>
    <td>ONNX Runtime*</td>
    <td><a href="https://onnxruntime.ai/docs/get-started/with-python.html#quickstart-examples-for-pytorch-tensorflow-and-scikit-learn">Quickstart Examples for PyTorch, TensorFlow, and SciKit Learn</a></td>
    <td>Train a model using your favorite framework, export to ONNX format and inference in any supported ONNX Runtime language.</td>
  </tr>
  <tr>
    <td rowspan="2">Intel® Extension of TensorFlow*</td>
    <td><a href="IntelTensorFlow_GettingStarted">IntelTensorFlow_GettingStarted</a></td>
    <td>A simple training example for TensorFlow.</td>
  </tr>
  <tr>
    <td><a href="Intel® Extension For TensorFlow GettingStarted">Intel® Extension For TensorFlow GettingStarted</a></td>
    <td>Guides users how to run a TensorFlow inference workload on both GPU and CPU.</td>
  </tr>
  <tr>
    <td>JAX*</td>
    <td><a href="IntelJAX GettingStarted">IntelJAX GettingStarted</a></td>
    <td>The JAX Getting Started sample demonstrates how to train a JAX model and run inference on Intel® hardware.</td>
  </tr>
</table>

*Other names and brands may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html)
