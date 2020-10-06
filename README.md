# [Stable Releases](https://github.com/oneapi-src/oneAPI-samples/tags)

We recommend checking out a release version of the repository. Releases are available [here](https://github.com/oneapi-src/oneAPI-samples/tags).

The latest versions of code samples on the master branch are not guaranteed to be stable.

# Index

|Code Sample    |Supported Intel(r)   Architecture(s)    	|Description 	| 
|-----------------------|-------------------------------------------|---------------|
|DirectPrograming/ |
|../DPC++/CombinationalLogic/Mandelbrot 	|GPU, CPU  	|Example of a fractal in   mathematics 	|
|../DPC++/CombinationalLogic/Sepia-filter    	|GPU, CPU  	|Color image conversion using 1D   range    	|
|../DPC++/DenseLinearAlgebra/Complex_mult    	|GPU, CPU  	|Complex number Multiplication    	|
|../DPC++/DenseLinearAlgebra/Matrix_mul 	|GPU, CPU  	|Simple program that multiplies   two large matrices in parallel using DPC++, OpenMP and MKL  	|
|../DPC++/DenseLinearAlgebra/Simple-add 	|FPGA, GPU, CPU 	|Simple Add program	|
|../DPC++/DenseLinearAlgebra/Vector-add 	|FPGA, GPU, CPU 	|Simple Vector add program   	|
|../DPC++/GraphTraversal/Bitonic-sort   	|GPU, CPU  	|Implementation of bitonic sort   using DPC++.   	|
|../DPC++/ParallelPatterns/Dpc_reduce   	|GPU, CPU  	|A simple program that calculates   pi,  implemented using C++ and DPC++. 	|
|../DPC++/SpectralMethods/Discrete-cosine-transform    	|GPU, CPU  	|Image processing algorithm used   in JPEG compression	|
|../DPC++/StructuredGrids/1d_HeatTransfer    	|GPU, CPU  	|A simulation of one dimensional   heat transfer process using DPC++.	|
|../DPC++/StructuredGrids/ISO2DFD_DPCPP 	|GPU, CPU  	|A simple finite difference   stencil kernel for solving 2D acoustic isotropic wave equation using DPC++	|
|../DPC++/StructuredGrids/ISO3DFD_DPCPP 	|GPU, CPU  	|A finite difference stencil   kernel for solving 3D acoustic isotropic wave equation using DPC++  	|
|../DPC++/StructuredGrids/Particle-diffusion 	|GPU, CPU  	|A simple implementation of a   Monte Carlo simulation of the diffusion of water molecules in tissue    	|
|../DPC++FPGA/ReferenceDesigns/crr                  |FPGA |High-performance CRR binomial tree option pricing model using DPC++ on FPGA|
|../DPC++FPGA/ReferenceDesigns/gzip                 |FPGA |High-performance GZIP compression using DPC++ on FPGA|
|../DPC++FPGA/ReferenceDesigns/qrd                  |FPGA |High-performance QR decomposition of matrices using DPC++ on FPGA|
|../DPC++FPGA/Tutorials/                            |
|...../GettingStarted/fpga_compile                   |FPGA |Tutorial introducing how to compile DPC++ for FPGA |
|...../GettingStarted/fast_recompile                 |FPGA |Tutorial introducing host-only recompile to save DPC++ development time on FPGA |
|...../Tools/use_library                             |FPGA |Tutorial showing how to use cross-language libraries in DPC++ on FPGA |
|...../Tools/system_profiling                        |FPGA |Tutorial showing how to use the OpenCL Intercept Layer to profile DPC++ designs running on FPGA |
|...../DesignPatterns/double_buffering               |FPGA |Tutorial demonstrating how to overlap kernel execution with buffer transfers and host processing |
|...../DesignPatterns/n_way_buffering                |FPGA |Tutorial demonstrating an extension of double buffering to n-way buffering |
|...../DesignPatterns/onchip_memory_cache            |FPGA |Tutorial demonstrating the caching of on-chip memory to reduce loop initiation interval on FPGA |
|...../DesignPatterns/pipe_array                     |FPGA |Tutorial demonstrating how to create an array of pipes  |
|...../DesignPatterns/compute_units                  |FPGA |Tutorial demonstrating how to efficiently make multiple copies of a kernel, called compute units  |
|...../DesignPatterns/remove_loop_carried_dependency |FPGA |Tutorial demonstrating a technique to optimize performance by removing loop carried dependencies |
|...../DesignPatterns/triangular_loop                |FPGA |Tutorial demonstrating an advanced FPGA optimization technique for triangular loops |
|...../DesignPatterns/shannonization                 |FPGA |Tutorial demonstrating an optimization for removing computation from the critical path |
|...../Features/fpga_reg                             |FPGA |Tutorial demonstrating the use of the DPC++ FPGA power user extension intel::fpga_reg |
|...../Features/kernel_args_restrict                 |FPGA |Tutorial demonstrating how to avoid performance penalties due to kernel argument aliasing |
|...../Features/loop_coalesce                        |FPGA |Tutorial demonstrating the DPC++ FPGA loop_coalesce attribute |
|...../Features/loop_ivdep                           |FPGA |Tutorial demonstrating the use of the loop ivdep attribute |
|...../Features/loop_unroll                          |FPGA |Tutorial demonstrating the DPC++ unroll pragma and its performance trade-offs on FPGA |
|...../Features/max_concurrency                      |FPGA |Tutorial demonstrating the DPC++ FPGA max_concurrency attribute |
|...../Features/max_interleaving                     |FPGA |Tutorial demonstrating the DPC++ FPGA max_interleaving attribute |
|...../Features/memory_attributes                    |FPGA |Tutorial demonstrating how to use DPC++ FPGA memory attributes |
|...../Features/pipes                                |FPGA |Tutorial demonstrating the DPC++ FPGA pipes extension to transfer data between kernels |
|...../Features/speculated_iterations                |FPGA |Tutorial demonstrating the DPC++ FPGA speculated_iterations attribute |
|../C++/CombinationalLogic/Mandelbrot   	|CPU  	|Demonstrates how to accelerate   Mandelbrot performance with SIMD and parallelization using OpenMP*.   	|
|../C++/CompilerInfrastructure/Intrinsics    	|CPU  	|Shows how to utilize the   intrinsics supported by C++ compiler in a variety of applications.	|
|../C++/GraphTraversal/Mergesort   	|CPU  	|Shows how to accelerate scalar   merge sort program using OpenMP tasks   	|
|Libraries |
|../oneDPL/Gamma-correction 	|GPU, CPU  	|gamma correction using Parallel   STL 	|
|../oneDPL/Stable_sort_by_key    	|GPU, CPU  	|stable sort by key using   counting_iterator and zip_iterator  	|
|../oneVPL/hello-decode	|CPU  	|shows how to use oneVPL to   perform a simple video decode	|
|../oneVPL/hello-encode	|CPU  	|shows how to use oneVPL to   perform a simple video encode	|
|Tools |
|../ApplicationDebugger/Debugger/array-transform    	|GPU, CPU  	|Array transform   	|
|../IoTConnectionTools/Analog-in	|CPU  	|Analog   pin input example using Eclipse* MRAA  	|
|../IoTConnectionTools/Digital   In  	|CPU  	|GPIO   pin input example using Eclipse* MRAA    	|
|../IoTConnectionTools/Digital   Out 	|CPU  	|GPIO   pin output example using Eclipse* MRAA   	|
|../IoTConnectionTools/Hello   IoT World  	|CPU  	|Basic   example that prints the compiler used during build	|
|../IoTConnectionTools/Interrupt	|CPU  	|Interrupt   Service Routine example using Eclipse* MRAA   	|
|../IoTConnectionTools/Onboard   Blink    	|CPU  	|Built-in   LED blink for common IoT boards using Eclipse* MRAA 	|
|../IoTConnectionTools/PWM 	|CPU  	|Pulse   Width Modulation pin output using Eclipse* MRAA   	|
|../IoTConnectionTools/Up2   LEDs    	|CPU  	|Built-in   LED example for UP* Squared using Eclipse* MRAA	|
|../SystemDebug/System Debug   Sample Build    	|UEFI 	|Basic example that   showcases the features of the Intel® System Debugger	|

# License
 
The code samples are licensed under MIT license
 
# Known issues or limitations
 
## On Windows Platform
1. If you are using Visual Studio 2019, Visual Studio 2019 version 16.4.0 or newer is required.
2. To build samples on Windows, the required Windows SDK is ver. 10.0.17763.0. 
3. Now you should be able to build the code sample.
4. For beta, FPGA samples support Windows through FPGA-emulator.
5. If you encounter a compilation error like below when building a sample program, one reason is that the directory path of the sample is too long; the work around is to move the sample to a directory like "c:\temp\sample_name".
  * Error MSB6003 The specified task executable "dpcpp.exe" could not be run .......

