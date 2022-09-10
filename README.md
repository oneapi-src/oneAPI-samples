# oneAPI Samples

The oneAPI-samples repository contains samples for the [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html).

The version of the repository you use should match the version of the Intel® oneAPI Toolkit you have installed, particularly for the compilers.

The latest versions of code samples on the master branch are not guaranteed to be stable. Use a [stable release version](https://github.com/oneapi-src/oneAPI-samples/tags) of the repository.

## Repository Structure

The oneAPI-sample repository is organized by high-level categories.

* [AI-and-Analytics](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics)
  * [End-to-End-Workloads](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/End-to-end-Workloads)
  * [Features-and-Functionality](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality)
  * [Getting-Started-Samples](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples)
  * [Jupyter](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Jupyter)
* [DirectProgramming](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming)
  * [C++](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2B)
  * [DPC++](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B)
  * [DPC++FPGA](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA)
  * [Fortran](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA)
* [Libraries](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries)
* [Publications](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications)
  * [Data Parallel C++](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications/DPC%2B%2B)
* [RenderingToolkit](https://github.com/oneapi-src/oneAPI-samples/tree/master/RenderingToolkit)
* [Tools](https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/)


## Known Issues and Limitations

### Windows

- If you are using Microsoft Visual Studio 2019, you must use Microsoft Visual Studio 2019 version 16.4.0 or newer.
- Windows support for the FPGA code samples is limited to the **FPGA emulator** and **optimization reports**. Only Linux supports **FPGA hardware** compilation. See any FPGA code sample README.md for more details.
- If you encounter `Error MSB6003 The specified task executable ... could not be run...` when building a sample program, it might be due to the length of the directory path. Move the sample to a temp directory with a shorter path and recompile.

## Contribute

See [CONTRIBUTING.md](https://github.com/oneapi-src/oneAPI-samples/blob/master/CONTRIBUTING.md) and the [*Contributing a New Sample*](https://github.com/oneapi-src/oneAPI-samples/wiki/Contributing-a-New-Sample) wiki page for more information.

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).