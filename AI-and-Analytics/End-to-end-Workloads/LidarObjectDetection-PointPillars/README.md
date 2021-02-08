# `PointPillars` Sample
This sample performs 3D object detection and classification using data (point cloud) from a LIDAR sensor as input. The Intel® oneAPI implementation is based on the paper 'PointPillars: Fast Encoders for Object Detection from Point Clouds' [1] and the implementation in [2]. It shows how Intel® oneAPI kernels (using SYCL and DPCPP) can be used in combination with the Intel® Distribution of OpenVINO™ toolkit for CNN inference.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer / Intel Xe Graphics
| Software                          | Intel® oneAPI DPC++/C++ Compiler, Intel® Distribution of OpenVINO™ toolkit
| What you will learn               | How to combine Intel® Distribution of OpenVINO™ toolkit and Intel® oneAPI to offload the computation of a complex workload to one of Intel's supported accelerators (e.g. GPU or CPU)
| Time to complete                  | 15 minutes

## Purpose
PointPillars is an AI algorithm, that uses LIDAR point clouds to detect and classify 3D objects in the sensor environment. For this purpose, the algorithm consists of 5 main steps. First a pre-processing of the point cloud is performed. This is realized with the help of kernels implemented in oneAPI. Afterward, the preprocessed data is used by a so-called Pillar Feature Extraction (PFE) CNN to create a 2D image-like representation of the sensor environment. For the inference, this sample uses the Intel® Distribution of OpenVINO™ toolkit. It follows another oneAPI processing step before a second CNN inference for the so-called Region Proposal Network (RPN) is executed using the OpenVINO™ toolkit. Finally, the output data (object list) is post-processed and filtered, which is again performed in oneAPI kernels.


By default the application will use 'host' as the Intel® oneAPI execution device and CPU for Intel® Distribution of OpenVINO™ toolkit inferencing part. The Intel® oneAPI execution device and inferencing device are displayed in the output along with elapsed time of each of the five steps described above. For more details refer to section: [Execution Options for the Sample Program](#execution-options-for-the-sample-program).

## Key Implementation Details
The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.
This sample also demonstrates how to combine custom SYCL kernels (for CPU and GPU) with Intel® Distribution of OpenVINO™ toolkit including memory transfers to create comprehensive processing pipeline using LIDARs, from point cloud to object detection.

## License  
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)


## Building the `PointPillars` Sample Program for CPU and GPU
Currently, only Linux platforms are supported. It is recommended to use Ubuntu 18.04.

### Requirements
To build and run the PointPillars sample, the following libraries have to be installed:
1. Intel® Distribution of OpenVINO™ toolkit (at least 2021.1)
2. Intel® oneAPI Base Toolkit (at least 2021.1)
3. Boost (including `boost::program_options` library)
4. Optional: If the sample should be run on an Intel GPU, it might be necessary to upgrade the corresponding drivers. Therefore, please consult the following page: https://github.com/intel/compute-runtime/releases/   


### Build process
Perform the following steps:
1. Prepare the environment to be able to use the Intel® Distribution of OpenVINO™ toolkit and oneAPI
``` 
$ source /opt/intel/openvino_2021/bin/setupvars.sh
$ source /opt/intel/oneapi/setvars.sh
$ export TBB_DIR=/opt/intel/openvino_2021/inference_engine/external/tbb
```

2. Download the PFE and RPN models in ONNX format
``` 
$ mkdir -p data/model
$ cd data/model
$ wget https://github.com/k0suke-murakami/kitti_pretrained_point_pillars/raw/master/pfe.onnx
$ wget https://github.com/k0suke-murakami/kitti_pretrained_point_pillars/raw/master/rpn.onnx
$ cd ../..
```

2. Build the program using the following `cmake` commands. 
``` 
$ mkdir build && cd build
$ cmake ..
$ make
```

## Running the `PointPillars` Sample Program
After a successful build, the sample program can be run as follows:
```
./example.exe
```
The input data for the sample program is the example.pcd located in the /data folder. It contains an artificial point cloud from a simulated LIDAR sensor of the CARLA simulator [3]. The corresponding scene looks as follows:
![SampleScene](data/image.png)
The example.pcd file is a point cloud in ASCII using the PCL-format, which renders to (using the pcl_viewer tool):
![SamplePointCloud](data/pointcloud.png)
It is worth noting that there are three cars in this scene, one of which is very far away and thus not yet properly covered in the LIDAR scan. Then, there is a black car inside the intersection, which is also well visible in the LIDAR data, and a hidden car, right behind the black one. Hence, `PointPillars` should detect at least one car.

Successful execution of the sample program results in at least one detected object and an output similar to:
```
Starting PointPillars
   PreProcessing - 20ms
   AnchorMask - 10ms
   PFE Inference - 91ms
   Scattering - 50ms
   RPN Inference - 107ms
   Postprocessing - 13ms
Done
Execution time: 296ms

1 cars detected
Car: Probability = 0.622569 Position = (24.8561, 12.5615, -0.00771689) Length = 2.42855 Width = 3.61396
```

## Execution Options for the Sample Program
The sample program provides a few command-line options, which can be accessed using the 'help' option
```
./example.exe --help
```

Furthermore, it is possible to specify the execution device. For using multi-threaded CPU execution, please use:
```
./example.exe --cpu
```
For single-threaded execution on the host system, please use:
```
./example.exe --host
```
And to use an Intel® DG1 or integrated graphics, please use:
```
./example.exe --gpu
```
These options can also be used in combination, e.g.:
```
./example.exe --cpu --gpu --host
```
---

## References
[1] [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

[2] [Autoware package for Point Pillars](https://github.com/Autoware-AI/core_perception/tree/master/lidar_point_pillars)

[3] [Open-source simulator for autonomous driving research](http://carla.org/)

## Notes
OpenVINO is a trademark of Intel Corporation or its subsidiaries