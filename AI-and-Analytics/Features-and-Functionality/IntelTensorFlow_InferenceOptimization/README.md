# Tutorial : Optimize TensorFlow pre-trained model for inference
This tutorial will guide you how to optimize a pre-trained model for a better inference performance, and also 
analyze the model pb files before and after the inference optimizations.  

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 
| Hardware                          | Intel® Xeon® Scalable processor family or newer
| Software                          | Intel® oneAPI AI Analytics Toolkit
| What you will learn               | Optimize a pre-trained model for a better inference performance
| Time to complete                  | 30 minutes

## Purpose
Show users the importance of inference optimization on performance, and also analyze TF ops difference in pre-trained models before/after the optimizations.

## Key implementation details
### Jupyter Notebooks 
 

| Notebook | Notes|
| ------ | ------ |
|  tutorial_optimize_TensorFlow_pretrained_model.ipynb | Optimize a pre-trained model for a better inference performance, and also analyze the model pb files  |

    
## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Build and Run the Sample

### Pre-requirement

> NOTE: No action is required if users use Intel DevCloud as their environment. 
  Please refer to [Intel oneAPI DevCloud](https://intelsoftwaresites.secure.force.com/devcloud/oneapi) for Intel DevCloud.

 1. **Intel AI Analytics Toolkit**  
       You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation,   
       and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

 2. **Jupyter Notebook**
       Users can install via PIP by `$pip install notebook`.
       Users can also refer to the [installation link](https://jupyter.org/install) for details.



### Running the Sample

1. Launch Jupyter notebook: `$jupyter notebook --ip=0.0.0.0`


2. Follow the instructions to open the URL with the token in your browser
3. Click the `tutorial_optimize_TensorFlow_pretrained_model.ipynb` file
4. Change your Jupyter notebook kernel to "tensorflow" or "intel-tensorflow" 
5. Run through every cell of the notebook one by one



### Example of Output
Users should be able to see some diagrams for performance comparison and analysis.  
One example of performance comparison diagrams:
<br><img src="images/perf_comparison.png" width="500" height="400"><br>

For performance analysis, users can also see pie charts for different Tensorflow* operations in the analyzed pre-trained model pb file.  
One example of model pb file analysis diagrams:
<br><img src="images/saved_model_pie.png" width="800" height="600"><br>


