# oneAPI training Jupyter notebooks
The purpose of this repo is to be the central aggregation, curation, and distribution point for Juypter notebooks that are developed in support of oneAPI training programs (e.g. oneAPI Essentials Series).  These are the hands-on components to be used in conjunction with training slides.

# The organization of the Jupyter notebook directories is a follows:

| Notebook Name | Owner | Description |
|---|---|---|
|[oneAPI_Intro](01_oneAPI_Intro)|Praveen.K.Kundurthy@intel.com| + Introduction and Motivation for oneAPI and DPC++.<br>+ DPC++ __Hello World__<br>+ Compiling DPC++ and __DevCloud__ Usage<br>+ ___Lab Excercise___: Vector Increment to Vector Add |
|[DPCPP_Program_Structure](02_DPCPP_Program_Structure)|Praveen.K.Kundurthy@intel.com| + __Classes__ - device, device_selector, queue, basic kernels and ND-Range kernels, Buffers-Accessor memory model<br>+ DPC++ __Code Anotomy__<br>+ Implicit __Dependency__ with Accessors, __Synchronization__ with Host Accessor and Buffer Destruction<br>+ Creating __Custom__ Device Selector<br>+ ___Lab Exercise___: Complex Multiplication |
|[DPCPP_Unified_Shared_Memory](03_DPCPP_Unified_Shared_Memory)|Rakshith.Krishnappa@intel.com| + What is Unified Shared Memory(USM) and Motivation<br>+ __Implicit and Explicit USM__ code example<br>+ Handling __data dependency__ using depends_on() and ordered queues<br>+ ___Lab Exercise___: Solving data dependency with USM |
|[DPCPP_Sub_Groups](04_DPCPP_Sub_Groups)|Rakshith.Krishnappa@intel.com| + What is Sub-Goups and Motivation<br>+ Quering for __sub-group info__<br>+ Sub-group __collectives__<br>+ Sub-group __shuffle operations__ |
|[Intel_Advisor](05_Intel_Advisor)|Praveen.K.Kundurthy@intel.com| + __Offload Advisor__ Tool usage and command-line options<br>+ __Roofline Analysis__ and command-line options |
|[Intel_VTune_Profiler](06_Intel_VTune_Profiler)|Rakshith.Krishnappa@intel.com| + Intel VTune™ Profiler usage __in Intel DevCloud__ environment using command-line options<br>+ ___Lab Excercise___: VTune Profiling by collecting __gpu_hotspots__ for [iso3dfd](https://github.com/intel/HPCKit-code-samples/tree/master/Compiler/iso3dfd_dpcpp) sample application. |
|[Intel oneAPI DPC++ Library (oneDPL)](07_DPCPP_Library)|Praveen.K.Kundurthy@intel.com| + Introduction to Intel oneAPI DPC++ Library (oneDPL)<br>+ ___Lab Excercise___: Gamma Correction with oneDPL |
|[DPCPP_Reduction](08_DPCPP_Reduction)|Rakshith.Krishnappa@intel.com| + What are Reductions<br>+ Challenges with parallelizing reductions<br>+ __ONEAPI::reduce__ function for sub-groups and work-groups<br>+ __ONEAPI::reduction__ object in parallel_for |
|[Explore Buffers and Accessors in depth](09_DPCPP_Buffers_And_Accessors_Indepth)|Praveen.K.Kundurthy@intel.com| + Bufers and Accessors<br>+ Buffer properties and usecases<br>+ Create Sub-buffers<br>+ Host accessors and usecases |
|[SYCL Task Scheduling and Data Dependences](10_DPCPP_Graphs_Scheduling_Data_management)|Praveen.K.Kundurthy@intel.com| + Different types of data dependences<br>+ Execution of graph scheduling<br>+ modes of dependences in Graphs scheduling |


