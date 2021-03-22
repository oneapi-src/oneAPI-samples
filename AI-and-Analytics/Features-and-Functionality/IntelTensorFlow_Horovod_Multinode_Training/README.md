# `Distributed TensorFlow with Horovod` Sample
Today's modern computer systems are becoming heavily distributed. It is important to capitalize on scaling techniques to maximize the efficiency and performance of neural networks training, a resource-intensive process.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04 
| Hardware                          | Intel® Xeon® Scalable processor family or newer
| Software                          | Intel® oneAPI AI Analytics Toolkit
| What you will learn               | How to get scale-out (distribute) the training of a model on multiple compute nodes
| Time to complete                  | 10 minutes

## Purpose
This sample code shows how to get started with scaling out a neural network's training in TensorFlow on multiple compute nodes in a cluster. The sample uses  [Horovod](https://github.com/horovod/horovod)*, a distributed deep learning training framework, to facilitate the task of distributing the workload. Horovod's core principles are based on MPI concepts such as size, rank, local rank, allreduce, allgather and, broadcast.

Intel-optimized Tensorflow is available as part of Intel® AI Analytics Toolkit. For more information on the optimizations and performance data, see this blog post [TensorFlow* Optimizations on Modern Intel® Architecture](https://software.intel.com/content/www/us/en/develop/articles/tensorflow-optimizations-on-modern-intel-architecture.html).

## Key implementation details

 - The training dataset comes from Keras*'s built-in dataset.
 - The dataset will be split based on the number of MPI ranks 
 - We load Horovod and initialize the framework using `hvd.init()`
 - We then wrap the optimizer around Horovod's distributed optimizer with `opt = hvd.DistributedOptimizer(opt)`
 - The appropriate hooks are configured, and we make sure that only rank 0 writes the checkpoints
    
Runtime settings for `OMP_NUM_THREADS`, `KMP_AFFINITY`, and `Inter/Intra-op` Threads are set within the script. You can read more about these settings in this dedicated document: [Maximize TensorFlow Performance on CPU: Considerations and Recommendations for Inference Workloads](https://software.intel.com/en-us/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference) 
    
## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Build and Run the Sample

### Running Samples In DevCloud (Optional)
If running a sample in the Intel DevCloud, please follow the below steps to build the python environment. Remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the [Intel® oneAPI Base Toolkit Get Started Guide] (https://devcloud.intel.com/oneapi/get-started/base-toolkit/) 

### Pre-requirement

TensorFlow is ready for use once you finish the Intel AI Analytics Toolkit installation and have run the post-installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.


### Sourcing the oneAPI AI Analytics Toolkit environment variables

By default, the Intel AI Analytics toolkit is installed in the `/opt/intel/oneapi` folder. The toolkit may be loaded by sourcing the `setvars.sh` script on a Linux shell. Notice the flag `--ccl-configuration=cpu_icc`. By default, the `ccl-configuration` is set to `cpu_gpu_dpcpp`. However, since we are distributing our TensorFlow workload on multiple CPU nodes, we are configuring the Horovod installation to use CPUs. 

```
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc
```

### Creating a TensorFlow environment with Horovod

Let's proceed with creating a conda environment with the Intel-optimized TensorFlow and horovod installed. Execute the following commands:

```
conda create --name tensorflow_horovod 
conda activate tensorflow_horovod 
```

Find the path where the `tensorflow_horovod` conda environment has been created. 

```
conda install -c "/opt/intel/oneapi/conda_channel" -p <path_of_tensorflow_horovod_env>/tensorflow_horovod -y -q conda python=3.7 numpy intel-openmp tensorflow --offline
```

Before running the sample, you will need to install the 3rd-party [Horovod](https://github.com/horovod/horovod) framework. Proceed with installing Horovod with the follwing command:
```
env HOROVOD_WITHOUT_MPI=1 HOROVOD_CPU_OPERATIONS=CCL HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 python -m pip install --upgrade --force-reinstall --no-cache-dir horovod
```

## Running the Sample

To execute the script on one machine without invoking Horovod, type the following command in the terminal with Python installed:
```
    python TensorFlow_Multinode_Training_with_Horovod.py
```

To run the script with Horovod, we invoke MPI:
```
    horovodrun -np 2 TensorFlow_Multinode_Training_with_Horovod.py
```

In the example above, we run the script on two MPI threads but on the same node. To use multiple nodes, we pass the `-hosts` flag, where host1 and host2 are the hostnames of two nodes on your cluster. 

Example:

```
    horovodrun -n 2 -H host1,host2 TensorFlow_Multinode_Training_with_Horovod.py
```


### Example of Output
With successful execution, it will print out the following results:

```
[...]
I0930 19:50:23.496505 140411946298240 basic_session_run_hooks.py:606] Saving checkpoints for 0 into ./checkpoints/model.ckpt.
INFO:tensorflow:loss = 2.2964332, step = 1
I0930 19:50:25.164514 140410195934080 basic_session_run_hooks.py:262] loss = 2.2964332, step = 1
INFO:tensorflow:loss = 2.2851412, step = 1
I0930 19:50:25.186133 140411946298240 basic_session_run_hooks.py:262] loss = 2.2851412, step = 1
INFO:tensorflow:loss = 1.1958275, step = 101 (19.356 sec)
I0930 19:50:44.521067 140410195934080 basic_session_run_hooks.py:260] loss = 1.1958275, step = 101 (19.356 sec)
INFO:tensorflow:global_step/sec: 5.16882
[...]
============================
Number of tasks:  2
Total time is: 96.9508
```


