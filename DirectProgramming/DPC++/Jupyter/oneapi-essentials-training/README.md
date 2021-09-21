# oneAPI training Jupyter notebooks
This repository is both standalone and a GitLab sub-module to the larger oneAPI training repo at https://gitlab.devtools.intel.com/ecosystem-dev-programs/oneapi-training-development.

The purpose of this repo is to be the central aggregation, curation, and distribution point for Juypter notebooks that are developed in support of oneAPI training programs (e.g. oneAPI Essentials Series).  These are the hands-on components to be used in conjunction with training slides.

# The organization of the Jupyter notebook directories is a follows:

| Notebook Name | Owner | Description |
|---|---|---|
|[oneAPI_Intro](01_oneAPI_Intro)|Praveen.K.Kundurthy@intel.com| + Introduction and Motivation for oneAPI and DPC++.<br>+ DPC++ __Hello World__<br>+ Compiling DPC++ and __DevCloud__ Usage<br>+ ___Lab Exercise___: Vector Increment to Vector Add |
|[DPCPP_Program_Structure](02_DPCPP_Program_Structure)|Praveen.K.Kundurthy@intel.com| + __Classes__ - device, device_selector, queue, basic kernels and ND-Range kernels, Buffers-Accessor memory model<br>+ DPC++ __Code Anotomy__<br>+ Implicit __Dependency__ with Accessors, __Synchronization__ with Host Accessor and Buffer Destruction<br>+ Creating __Custom__ Device Selector<br>+ ___Lab Exercise___: Complex Multiplication |
|[DPCPP_Unified_Shared_Memory](03_DPCPP_Unified_Shared_Memory)|Rakshith.Krishnappa@intel.com| + What is Unified Shared Memory(USM) and Motivation<br>+ __Implicit and Explicit USM__ code example<br>+ Handling __data dependency__ using depends_on() and ordered queues<br>+ ___Lab Exercise___: Solving data dependency with USM |
|[DPCPP_Sub_Groups](04_DPCPP_Sub_Groups)|Rakshith.Krishnappa@intel.com| + What is Sub-Groups and Motivation<br>+ Quering for __sub-group info__<br>+ Sub-group __collectives__<br>+ Sub-group __shuffle operations__ |
|[Intel_Advisor](05_Intel_Advisor)|Praveen.K.Kundurthy@intel.com| + __Offload Advisor__ Tool usage and command-line options<br>+ __Roofline Analysis__ and command-line options |
|[Intel_VTune_Profiler](06_Intel_VTune_Profiler)|Rakshith.Krishnappa@intel.com| + Intel VTune Profiler usage __in Intel DevCloud__ environment using command-line options<br>+ ___Lab Exercise___: VTune Profiling by collecting __gpu_hotspots__ for [iso3dfd](https://github.com/intel/HPCKit-code-samples/tree/master/Compiler/iso3dfd_dpcpp) sample application. |
|[Intel oneAPI DPC++ Library (oneDPL)](07_DPCPP_Library)|Praveen.K.Kundurthy@intel.com| + Introduction to DPC++ Library<br>+ ___Lab Exercise___: Gamma Correction with oneDPL |
|[DPCPP_Reduction](08_DPCPP_Reduction)|Rakshith.Krishnappa@intel.com| + What are Reductions<br>+ Challenges with parallelizing reductions<br>+ __ONEAPI::reduce__ function for sub-groups and work-groups<br>+ __ONEAPI::reduction__ object in parallel_for |
|[Intel Distribution for GDB](11_Intel_Distribution_for_GDB)|karl.qi@intel.com| + Debug oneAPI Applications with the Intel Distribution for GDB __in Intel DevCloud__ environment<br>+  ___Lab Exercise___: Debug a simple array transform DPCPP program |