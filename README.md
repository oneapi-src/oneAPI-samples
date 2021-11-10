## Introduction

The oneAPI samples repository provides code samples for Intel oneAPI toolkits.<br><br>We recommend checking out a specific stable release version of the repository. The version of the repository you fetch should match the version of the oneAPI compiler you are using. [View available stable releases](https://github.com/oneapi-src/oneAPI-samples/tags). 
The latest versions (2022.1.0) of code samples on the master branch are not guaranteed to be stable.
 ### Sample Details

The oneAPI sample repository is organized as follows:

* [AI-and-Analytics:](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics)
  * [End-to-End-Workloads](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/End-to-end-Workloads)
  * [Features-and-Functionality](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality)
  * [Getting-Started-Samples](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples)
* [DirectProgramming](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming)
  * [C++](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2B)
  * [DPC++](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B)
  * [DPC++FPGA](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA)
    * [Reference Designs](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA/ReferenceDesigns)
    * [Tutorials](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA/Tutorials)
  * [Fortran](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA)
* [Libraries](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries)
  * [oneCCl](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneCCL)
  * [oneDAL](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDAL)
  * [oneDNN](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDNN)
  * [oneDPL](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDPL)
  * [oneMKL](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneMKL)
  * [oneTBB](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneTBB)
  * [oneVPL](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneVPL)
* [Publications](https://github.com/oneapi-src/oneAPI-samples/Publications/)
  * [Data Parallel C++](https://github.com/oneapi-src/oneAPI-samples/Publications/Data_Parallel_C%2B%2B)
* [Tools](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/)
  * [Advisor](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Advisor)
  * [Application Debugger](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/ApplicationDebugger)
  * [Benchmark](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Benchmark)
  * [IoT Connections Tools](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/IoTConnectionsTools)
  * [Migration](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Migration)
  * [Socwatch](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Socwatch)
  * [Trace](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/Trace)
  * [UEFI debug](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/UEFI%20debug)
  * [VTune Profiler](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/VTuneProfiler)
)

## Known Issues or Limitations

### On Windows Platform

- If you are using Visual Studio 2019, Visual Studio 2019 version 16.4.0 or newer is required.
- Windows support for the FPGA code samples is limited to the FPGA emulator and optimization reports. Compile targets for FPGA hardware are provided on Linux only. See any FPGA code sample for more details.
- If you encounter a compilation error when building a sample program, such as the example error below, the directory path of the sample may be too long. The workaround is to move the sample to a temp directory.
    - Example error: *Error MSB6003 The specified task executable dpcpp.exe could not be run .......

## Additional Resources
- Samples in [Alphabetical order w/ device target](https://github.com/oneapi-src/CODESAMPLESLIST.md/)
- Samples by [Change History](https://github.com/oneapi-src/oneAPI-samples/CHANGELOGS.md)

## Contributing

See [CONTRIBUTING wiki](https://github.com/oneapi-src/oneAPI-samples/blob/master/CONTRIBUTING.md) for more information.



## New Code Samples

|Version Introduced   |Sample Name|Description|
 |-----------------------|-------------------------------------------|---------------|
|2022.1.0|[Adaptive Noise Reduction](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC++FPGA/ReferenceDesigns/anr)|A highly optimized adaptive noise reduction (ANR) algorithm on an FPGA.|
|2022.1.0|[Printf](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC++FPGA/Tutorials/Features/printf)|This FPGA tutorial explains how to use the printf() to print in a DPC++ FPGA program|
|2022.1.0|[Scheduler Target FMAX](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC++FPGA/Tutorials/Features/scheduler_target_fmax)|Explain the scheduler_target_fmax_mhz attribute and its effect on the performance of Intel® FPGA kernels|

Total Samples: 161


## Deleted Code Samples

|Version Introduced|Version Deleted|Sample Name|Description|Path|
 |---|---|------|---------------|------|
| 2021.1.Gold | 2021.3.0 | Use Library | Removed for 2021.4 - Remove the tutorial use_library due to HLS/OCL library support being removed in oneAPI 2021.4. Also note that RTL libraries has an issue (with a known workaround) so that part of the flow is also removed, but we will likely have a KDB to showcase the workaround. The fix for this is expected in 2022.1. Please use 2021.3 if you either need HLS/OCL library support in SYCL, or if you need continued access to RTL libraries. | [2021.3.0](https://github.com/oneapi-src/oneAPI-samples/releases/tag/2021.3.0) Path: DirectProgramming/DPC++FPGA/Tutorials/Tools/use_library|


## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

Report Generated on:  October 20, 2021
