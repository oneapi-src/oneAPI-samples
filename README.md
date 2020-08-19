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
|../DPC++FPGA/ReferenceDesigns/crr                                    |FPGA |High-performance CRR binomial tree option pricing model using DPC++ on FPGA|
|../DPC++FPGA/ReferenceDesigns/gzip                                   |FPGA |High-performance GZIP compression using DPC++ on FPGA|
|../DPC++FPGA/ReferenceDesigns/qrd                                    |FPGA |High-performance QR decomposition of matrices using DPC++ on FPGA|
|../DPC++FPGA/Tutorials/GettingStarted/fpga_compile                   |FPGA |Tutorial introducing how to compile DPC++ for FPGA |
|../DPC++FPGA/Tutorials/GettingStarted/fast_recompile                 |FPGA |Tutorial introducing host-only recompile to save DPC++ development time on FPGA |
|../DPC++FPGA/Tutorials/Tools/use_library                             |FPGA |Tutorial showing how to use cross-language libraries in DPC++ on FPGA |
|../DPC++FPGA/Tutorials/Tools/system_profiling                        |FPGA |Tutorial showing how to use the OpenCL Intercept Layer to profile DPC++ designs running on FPGA |
|../DPC++FPGA/Tutorials/DesignPatterns/double_buffering               |FPGA |Tutorial demonstrating how to overlap kernel execution with buffer transfers and host processing |
|../DPC++FPGA/Tutorials/DesignPatterns/n_way_buffering                |FPGA |Tutorial demonstrating an extension of double buffering to n-way buffering |
|../DPC++FPGA/Tutorials/DesignPatterns/onchip_memory_cache            |FPGA |Tutorial demonstrating the caching of on-chip memory to reduce loop initiation interval on FPGA |
|../DPC++FPGA/Tutorials/DesignPatterns/pipe_array                     |FPGA |Tutorial demonstrating how to create an array of pipes  |
|../DPC++FPGA/Tutorials/DesignPatterns/remove_loop_carried_dependency |FPGA |Tutorial demonstrating a technique to optimize performance by removing loop carried dependencies |
|../DPC++FPGA/Tutorials/DesignPatterns/triangular_loop                |FPGA |Tutorial demonstrating an advanced FPGA optimization technique for triangular loops |
|../DPC++FPGA/Tutorials/Features/fpga_reg                             |FPGA |Tutorial demonstrating the use of the DPC++ FPGA power user extension intel::fpga_reg |
|../DPC++FPGA/Tutorials/Features/kernel_args_restrict                 |FPGA |Tutorial demonstrating how to avoid performance penalties due to kernel argument aliasing |
|../DPC++FPGA/Tutorials/Features/loop_coalesce                        |FPGA |Tutorial demonstrating the DPC++ FPGA loop_coalesce attribute |
|../DPC++FPGA/Tutorials/Features/loop_ivdep                           |FPGA |Tutorial demonstrating the use of the loop ivdep attribute |
|../DPC++FPGA/Tutorials/Features/loop_unroll                          |FPGA |Tutorial demonstrating the DPC++ unroll pragma and its performance trade-offs on FPGA |
|../DPC++FPGA/Tutorials/Features/max_concurrency                      |FPGA |Tutorial demonstrating the DPC++ FPGA max_concurrency attribute |
|../DPC++FPGA/Tutorials/Features/memory_attributes                    |FPGA |Tutorial demonstrating how to use DPC++ FPGA memory attributes |
|../DPC++FPGA/Tutorials/Features/pipes                                |FPGA |Tutorial demonstrating the DPC++ FPGA pipes extension to transfer data between kernels |
|../DPC++FPGA/Tutorials/Features/speculated_iterations                |FPGA |Tutorial demonstrating the DPC++ FPGA speculated_iterations attribute |
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

#License
 
The code samples are licensed under MIT license
 
#Known issues or limitations
 
##On Windows Platform
1. If you are using Visual Studio 2019, Visual Studio 2019 version 16.3.0 or newer is required.
2. To build samples on Windows, the required Windows SDK is ver. 10.0.17763.0. 

  1. If the SDK is not installed, use the following instructions below to avoid build failure: 

    1. Open the "code sample's" .sln from within Visual Studio 2017 or 2019,
    2. Right-click on the project name in "Solution Explorer" and select "Properties"
  
  2. The project property dialog opens. 
  
    1. Select the "General" tab on the left,
    2. Select on the right side of the dialog box "Windows SDK Version"(2nd Item".
    3. Click on the drop-down icon to select a version that is installed on your system.
    4. click on [Ok] to save.

3. Now you should be able to build the code sample.
4. For beta, FPGA samples support Windows through FPGA-emulator.
5. If you encounter a compilation error like below when building a sample program, one reason is that the directory path of the sample is too long; the work around is to move the sample to a directory like "c:\temp\sample_name".
  * Error MSB6003 The specified task executable "dpcpp-cl.exe" could not be run ......

