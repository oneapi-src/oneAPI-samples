# oneAPI training Jupyter notebooks

The purpose of this repo is to be the central aggregation, curation, and
distribution point for Juypter notebooks that are developed in support of
oneAPI training programs (e.g., oneAPI Essentials Series).

The Jupyter notebooks are tested and can be run on the Intel Devcloud. Below
are the steps to access these Jupyter notebooks on the Intel Devcloud:

1. Register with the Intel Devcloud at
   https://intelsoftwaresites.secure.force.com/devcloud/oneapi

2. SSH into the Intel Devcloud "terminal"

3. Type the following command to download the oneAPI-essentials series of
   Jupyter notebooks and OpenMP offload notebooks into your devcloud account
   `/data/oneapi_workshop/get_jupyter_notebooks.sh`

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# Organization of the Jupyter Notebook Directories

Notebook Name: Owner
* Descriptions
*

[oneAPI_Intro](01_oneAPI_Intro): Praveen.K.Kundurthy@intel.com
* Introduction and Motivation for oneAPI and DPC++
* DPC++ __Hello World__
* Compiling DPC++ and __DevCloud__ Usage
* _Lab Excercise_: Vector Increment to Vector Add

[DPCPP_Program_Structure](02_DPCPP_Program_Structure): Praveen.K.Kundurthy@intel.com
* __Classes__ - device, device_selector, queue, basic kernels and ND-Range kernels, Buffers-Accessor memory model
* DPC++ __Code Anotomy__
* Implicit __Dependency__ with Accessors, __Synchronization__ with Host Accessor and Buffer Destruction
* Creating __Custom__ Device Selector
* _Lab Exercise_: Complex Multiplication

[DPCPP_Unified_Shared_Memory](03_DPCPP_Unified_Shared_Memory): Rakshith.Krishnappa@intel.com
* What is Unified Shared Memory(USM) and Motivation
* __Implicit and Explicit USM__ code example
* Handling __data dependency__ using depends_on() and ordered queues
* _Lab Exercise_: Solving data dependency with USM

[DPCPP_Sub_Groups](04_DPCPP_Sub_Groups): Rakshith.Krishnappa@intel.com
* What is Sub-Goups and Motivation
* Quering for __sub-group info__
* Sub-group __collectives__
* Sub-group __shuffle operations__

[Intel_Advisor](05_Intel_Advisor): Praveen.K.Kundurthy@intel.com
* __Offload Advisor__ Tool usage and command-line options
* __Roofline Analysis__ and command-line options

[Intel®_VTune™_Profiler](06_Intel_VTune_Profiler): Rakshith.Krishnappa@intel.com
* Intel VTune Profiler usage __in Intel DevCloud__ environment using command-line options
* _Lab Excercise_: VTune Profiling by collecting __gpu_hotspots__ for [iso3dfd](https://github.com/intel/HPCKit-code-samples/tree/master/Compiler/iso3dfd_dpcpp) sample application

[Intel oneAPI DPC++ Library (oneDPL)](07_DPCPP_Library): Praveen.K.Kundurthy@intel.com
* Introduction to oneAPI DPC++ Library
* _Lab Excercise_: Gamma Correction with oneDPL
