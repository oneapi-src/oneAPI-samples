# Intel oneAPI Deep Neural Network Library (oneDNN)

Deep Neural Networks Library for Deep Neural Networks (oneDNN) is an open-source performance library for deep learning applications. The library includes basic building blocks for neural networks optimized for Intel Architecture Processors and Intel Processor Graphics. oneDNN is intended for deep learning applications and framework developers interested in improving application performance on Intel CPUs and GPUs

Github: https://github.com/oneapi-src/oneDNN

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# oneDNN Tutorials

| Type      | Name                 | Description                                                  |
| --------- | ----------------------- | ------------------------------------------------------------ |
| Component | [getting_started](tutorial_getting_started.ipynb)  | The sample also includes a Jupyter notebook with step by step instructions on building code with different compilers and runtime configurations oneDNN support. |
| Component | [simple_model](tutorial_simple_model.ipynb)| A Jupyter notebook with step by step instructions on running oneDNN-based application on a GPU. |
| Component | [verbose_jitdump](tutorial_verbose_jitdump.ipynb) | This Jupyter Notebook demonstrates how to use Verbose Mode and JIT Dump to profile oneDNN samples. |
| Component | [analyze_isa_with_dispatcher_control](tutorial_analyze_isa_with_dispatcher_control.ipynb) | This Jupyter Notebook demonstrates how to use CPU Dispatch Control to generate JIT codes among different ISA on CPU and also analyze JIT kernels among ISAs.|
>  Notice : Please use Intel oneAPI DevCloud as the environment for jupyter notebook samples. \
Users can refer to [DevCloud Getting Started](https://devcloud.intel.com/oneapi/get-started/) for using DevCloud \
Users can use JupyterLab from DevCloud via "One-click Login in", and download samples via "git clone" or the "oneapi-cli" tool \
Once users are in the JupyterLab with downloaded jupyter notebook samples, they can start following the steps without further installation needed.
