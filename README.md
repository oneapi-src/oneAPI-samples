## Introduction

The oneAPI samples repository provides code samples for Intel oneAPI toolkits.<br><br>We recommend checking out a specific stable release version of the repository to [View available stable releases](https//github.com/oneapi-src/oneAPI-samples/tags). 
The latest versions (2021.4.0) of code samples on the master branch are not guaranteed to be stable.
 ### Sample Details

The oneAPI sample repository is organized as follows:

* AI-and-Analytics:
  * End-to-End-Workloads
  * Features-and-Functionality
  * Getting-Started-Samples
* DirectProgramming
  * C++
  * DPC++
  * DPC++FPGA
    * Reference Designs
    * Tutorials
  * Fortran
* Libraries
  * oneCCl
  * oneDAL
  * oneDNN
  * oneDPL
  * oneMKL
  * oneTBB
  * oneVPL
* Publications
  * Data Parallel C++
* Tools
  * Advisor
  * ApplicationDebugger
  * Benchmark/STREAM
  * IoTConnectionsTools
  * Migration
  * Socwatch
  * Trace
  * UEFI debug
  * VTuneProfiler
)
To view an alphabetized list of all samples with descriptions by:
- Samples by [Change History](https://github.com/oneapi-src/oneAPI-samples/CHANGELOGS.md)
- Samples by with [Device Target](https://github.com/oneapi-src/CODESAMPLESLIST.md/)

### On Windows Platform

- If you are using Visual Studio 2019, Visual Studio 2019 version 16.4.0 or newer is required.
- Windows support for the FPGA code samples is limited to the FPGA emulator and optimization reports. Compile targets for FPGA hardware are provided on Linux only. See any FPGA code sample for more details.
- If you encounter a compilation error when building a sample program, such as the example error below, the directory path of the sample may be too long. The workaround is to move the sample to a temp directory.
- Example error: *Error MSB6003 The specified task executable dpcpp.exe could not be run .......

## Known Issues or Limitations

## Contributing

See [CONTRIBUTING wiki](https://github.com/oneapi-src/oneAPI-samples/blob/master/CONTRIBUTING.md) for more information.



## New Code Samples

|Code Sample    |Supported Intel&reg;   Architecture(s)|Description|
 |-----------------------|-------------------------------------------|---------------|
|2021.4.0|[Merge Sort](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC++FPGA/ReferenceDesigns/merge_sort)|A Reference design demonstrating merge sort on an Intel® FPGA|
|2021.4.0|[Private Copies](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC++FPGA/Tutorials/Features/private_copies)|An Intel® FPGA tutorial demonstrating how to use the private_copies attribute to trade off the resource use and the throughput of a DPC++ FPGA program|
|2021.4.0|[Stall Enable](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC++FPGA/Tutorials/Features/stall_enable)|An Intel® FPGA tutorial demonstrating the use_stall_enable_clusters attribute|
## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)